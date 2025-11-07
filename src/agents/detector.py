"""Detector Agent - Initial triage and threat classification."""

import re
import json
import time
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from ..agents.state import AgenticGuardState
from ..config.prompts import DETECTOR_PROMPT, MALICIOUS_PATTERNS, ALL_MALICIOUS_PATTERNS
from ..utils.llm_factory import get_llm


class DetectorAgent:
    """Fast triage agent for initial threat detection.

    Combines rule-based pattern matching with LLM classification
    for optimal speed and accuracy. Targets <100ms latency.
    """

    def __init__(self, llm: Optional[Any] = None):
        """Initialize the Detector Agent.

        Args:
            llm: Optional LLM instance. If not provided, uses factory default.
        """
        self.llm = llm or get_llm(model_type="fast", temperature=0, max_tokens=500)
        # Compile patterns by category
        self.pattern_categories = {}
        for category, patterns in MALICIOUS_PATTERNS.items():
            self.pattern_categories[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
        # Also keep flattened list for backward compatibility
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in ALL_MALICIOUS_PATTERNS]

    def _check_patterns(self, prompt: str) -> tuple[bool, list[str], dict[str, int]]:
        """Check prompt against known malicious patterns.

        Args:
            prompt: User prompt to check

        Returns:
            Tuple of (has_malicious_patterns, matched_patterns, category_counts)
        """
        matched_patterns = []
        category_counts = {}

        # Check patterns by category for better threat assessment
        for category, patterns in self.pattern_categories.items():
            category_matches = 0
            for pattern in patterns:
                if pattern.search(prompt):
                    matched_patterns.append(pattern.pattern)
                    category_matches += 1
            if category_matches > 0:
                category_counts[category] = category_matches

        return len(matched_patterns) > 0, matched_patterns, category_counts

    def _classify_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Use LLM to classify the prompt when pattern matching is inconclusive.

        Args:
            prompt: User prompt to classify

        Returns:
            Classification result from LLM
        """
        try:
            formatted_prompt = DETECTOR_PROMPT.format(user_prompt=prompt)
            messages = [
                SystemMessage(content="You are a security detection agent. Respond only with valid JSON."),
                HumanMessage(content=formatted_prompt)
            ]

            response = self.llm.invoke(messages)

            # Extract content - handle both string and object responses
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Convert to string if not already (handles Mock objects)
            if not isinstance(content, str):
                content = str(content)

            # Parse JSON response - handle cases where JSON might be wrapped in markdown
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)

            # Validate required fields
            required_fields = ["threat_level", "threat_type", "confidence", "reasoning"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            return result

        except Exception as e:
            # Fallback classification on error - be more conservative
            # If LLM fails, assume it's likely a benign prompt unless patterns were found
            return {
                "threat_level": "SAFE",
                "threat_type": "null",
                "confidence": 0.3,
                "reasoning": f"Classification error, defaulting to safe: {str(e)}"
            }

    def invoke(self, state: AgenticGuardState) -> AgenticGuardState:
        """Process the state and detect threats.

        Args:
            state: Current state of the workflow

        Returns:
            Updated state with detection results
        """
        start_time = time.time()
        user_prompt = state["user_prompt"]

        # First, check against known patterns
        has_patterns, matched_patterns, category_counts = self._check_patterns(user_prompt)

        # Determine threat level based on pattern categories
        if has_patterns:
            # Check for high-severity categories
            high_severity = any(cat in category_counts for cat in
                               ["instruction_override", "system_manipulation", "jailbreak", "harmful_content"])

            # If high-severity patterns detected OR multiple patterns
            if high_severity or len(matched_patterns) >= 2:
                threat_level = "MALICIOUS"
                threat_type = self._determine_threat_type_from_categories(category_counts)

                # Higher confidence for more patterns and high-severity categories
                base_confidence = 0.8 if high_severity else 0.7
                confidence = min(base_confidence + (len(matched_patterns) * 0.05), 1.0)
                reasoning = f"Detected {len(matched_patterns)} malicious patterns in categories: {', '.join(category_counts.keys())}"

            # Single medium-severity pattern - likely SUSPICIOUS
            elif any(cat in category_counts for cat in ["exfiltration", "roleplay"]):
                # Use LLM to confirm since these can have false positives
                llm_result = self._classify_with_llm(user_prompt)

                # If LLM agrees it's malicious, upgrade it
                if llm_result["threat_level"] == "MALICIOUS":
                    threat_level = "MALICIOUS"
                    confidence = min(llm_result["confidence"] + 0.1, 1.0)
                else:
                    threat_level = "SUSPICIOUS"
                    confidence = max(llm_result["confidence"], 0.6)

                threat_type = self._determine_threat_type_from_categories(category_counts)
                reasoning = f"Medium-severity patterns detected: {', '.join(category_counts.keys())}"

            else:
                # Shouldn't happen but fallback to LLM
                llm_result = self._classify_with_llm(user_prompt)
                threat_level = llm_result["threat_level"]
                threat_type = llm_result["threat_type"]
                confidence = llm_result["confidence"]
                reasoning = llm_result["reasoning"]

        else:
            # No patterns found - rely on LLM
            llm_result = self._classify_with_llm(user_prompt)
            threat_level = llm_result["threat_level"]
            threat_type = llm_result["threat_type"]
            confidence = llm_result["confidence"]
            reasoning = llm_result["reasoning"]

        # Update state
        state["threat_level"] = threat_level
        state["threat_type"] = threat_type
        state["detector_confidence"] = confidence
        state["attack_patterns"] = matched_patterns if has_patterns else []

        # Add to agent trace - for LangGraph annotated fields, return new items only
        state["agent_trace"] = [f"detector (level={threat_level}, confidence={confidence:.2f})"]

        # Track processing time
        processing_time = time.time() - start_time
        state["processing_time"] = processing_time

        return state

    def _determine_threat_type_from_categories(self, category_counts: dict[str, int]) -> str:
        """Determine threat type based on matched pattern categories.

        Args:
            category_counts: Dictionary of category matches

        Returns:
            Threat type classification
        """
        # Priority order for threat type determination
        if "instruction_override" in category_counts or "system_manipulation" in category_counts:
            return "injection"
        elif "jailbreak" in category_counts or "roleplay" in category_counts:
            return "jailbreak"
        elif "exfiltration" in category_counts:
            return "exfiltration"
        elif "harmful_content" in category_counts:
            return "harmful_request"
        else:
            return "injection"  # Default for safety
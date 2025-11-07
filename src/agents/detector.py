"""Detector Agent - Initial triage and threat classification."""

import re
import json
import time
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from ..agents.state import AgenticGuardState
from ..config.prompts import DETECTOR_PROMPT, MALICIOUS_PATTERNS
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
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in MALICIOUS_PATTERNS]

    def _check_patterns(self, prompt: str) -> tuple[bool, list[str]]:
        """Check prompt against known malicious patterns.

        Args:
            prompt: User prompt to check

        Returns:
            Tuple of (has_malicious_patterns, matched_patterns)
        """
        matched_patterns = []
        for pattern in self.compiled_patterns:
            if pattern.search(prompt):
                matched_patterns.append(pattern.pattern)

        return len(matched_patterns) > 0, matched_patterns

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
            # Fallback classification on error
            return {
                "threat_level": "SUSPICIOUS",
                "threat_type": "null",
                "confidence": 0.5,
                "reasoning": f"Classification error: {str(e)}"
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
        has_patterns, matched_patterns = self._check_patterns(user_prompt)

        if has_patterns and len(matched_patterns) >= 2:
            # High confidence detection based on multiple pattern matches
            threat_level = "MALICIOUS"
            threat_type = self._determine_threat_type(matched_patterns)
            confidence = min(0.9 + (len(matched_patterns) * 0.02), 1.0)
            reasoning = f"Multiple malicious patterns detected: {', '.join(matched_patterns[:3])}"

        elif has_patterns:
            # Single pattern match - use LLM for confirmation
            llm_result = self._classify_with_llm(user_prompt)

            # Boost confidence if pattern and LLM agree
            if llm_result["threat_level"] == "MALICIOUS":
                threat_level = "MALICIOUS"
                confidence = min(llm_result["confidence"] + 0.1, 1.0)
            else:
                threat_level = llm_result["threat_level"]
                confidence = llm_result["confidence"]

            threat_type = llm_result["threat_type"]
            reasoning = f"Pattern match + LLM analysis: {llm_result['reasoning']}"

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

        # Add to agent trace
        if "agent_trace" not in state:
            state["agent_trace"] = []
        state["agent_trace"].append(f"detector (level={threat_level}, confidence={confidence:.2f})")

        # Track processing time
        processing_time = time.time() - start_time
        state["processing_time"] = processing_time

        return state

    def _determine_threat_type(self, patterns: list[str]) -> str:
        """Determine threat type based on matched patterns.

        Args:
            patterns: List of matched pattern strings

        Returns:
            Threat type classification
        """
        patterns_str = " ".join(patterns).lower()

        if any(word in patterns_str for word in ["ignore", "disregard", "forget", "override"]):
            return "injection"
        elif any(word in patterns_str for word in ["jailbreak", "dan", "developer", "roleplay"]):
            return "jailbreak"
        elif any(word in patterns_str for word in ["print", "reveal", "repeat", "system"]):
            return "exfiltration"
        else:
            return "injection"  # Default to injection for unclassified patterns
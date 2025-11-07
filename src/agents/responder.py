"""Response Agent - Generates structured security responses."""

import time
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from ..agents.state import AgenticGuardState
from ..config.prompts import RESPONDER_PROMPT
from ..utils.llm_factory import get_llm


class ResponseAgent:
    """Generates final security response and recommendations.

    This agent creates human-readable explanations and compiles
    the final security response object.
    """

    def __init__(self, llm: Optional[Any] = None):
        """Initialize the Response Agent.

        Args:
            llm: Optional LLM instance. Defaults to factory balanced model.
        """
        self.llm = llm or get_llm(
            model_type="balanced",
            temperature=0.3,  # Slightly creative for better explanations
            max_tokens=500
        )

    def _determine_action(self, threat_level: str, confidence: float) -> str:
        """Determine recommended action based on threat level and confidence.

        Args:
            threat_level: Detected threat level
            confidence: Final confidence score

        Returns:
            Recommended action
        """
        if threat_level == "MALICIOUS" and confidence > 0.8:
            return "block"
        elif threat_level == "MALICIOUS" or (threat_level == "SUSPICIOUS" and confidence > 0.7):
            return "flag"
        else:
            return "allow"

    def _generate_explanation(self, state: AgenticGuardState, action: str) -> str:
        """Generate human-readable explanation using LLM.

        Args:
            state: Current workflow state
            action: Recommended action

        Returns:
            Human-readable explanation
        """
        try:
            # Format the prompt with state information
            formatted_prompt = RESPONDER_PROMPT.format(
                threat_level=state["threat_level"],
                threat_type=state.get("threat_type", "null"),
                final_confidence=state["final_confidence"],
                attack_patterns=", ".join(state.get("attack_patterns", [])[:3]) or "None detected",
                recommended_action=action
            )

            messages = [
                SystemMessage(content="You are a security response agent. Provide clear, concise explanations."),
                HumanMessage(content=formatted_prompt)
            ]

            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            # Fallback explanation on error
            return self._generate_fallback_explanation(state, action)

    def _generate_fallback_explanation(self, state: AgenticGuardState, action: str) -> str:
        """Generate fallback explanation without LLM.

        Args:
            state: Current workflow state
            action: Recommended action

        Returns:
            Fallback explanation
        """
        threat_level = state["threat_level"]
        threat_type = state.get("threat_type", "unknown")
        confidence = state["final_confidence"]

        if threat_level == "MALICIOUS":
            return (
                f"A {threat_type} attack was detected with high confidence ({confidence:.0%}). "
                f"The prompt contains malicious patterns that attempt to compromise system security. "
                f"Action: {action.upper()} - This prompt should be {action}ed to prevent potential harm. "
                f"Recommended action: {action} this request."
            )
        elif threat_level == "SUSPICIOUS":
            return (
                f"Suspicious activity detected that may indicate a {threat_type} attempt. "
                f"While not definitively malicious, the prompt shows concerning patterns. "
                f"Action: {action.upper()} - Recommend manual review for safety."
            )
        else:
            return (
                f"No significant security threats detected in this prompt. "
                f"The content appears to be a normal user query. "
                f"Action: {action.upper()} - Safe to process normally."
            )

    def _compile_response(self, state: AgenticGuardState, action: str, explanation: str) -> Dict[str, Any]:
        """Compile the final security response object.

        Args:
            state: Current workflow state
            action: Recommended action
            explanation: Human-readable explanation

        Returns:
            Complete security response
        """
        response = {
            "threat_detected": state["threat_level"] != "SAFE",
            "threat_level": state["threat_level"],
            "threat_type": state.get("threat_type", "null"),
            "confidence_score": state["final_confidence"],
            "recommended_action": action,
            "explanation": explanation,
            "attack_patterns": state.get("attack_patterns", []),
            "embedding_similarity": state.get("embedding_similarity"),
            "processing_time_seconds": state.get("processing_time", 0),
            "agent_trace": state.get("agent_trace", [])
        }

        # Add semantic analysis details if available
        if "semantic_analysis" in state and state["semantic_analysis"]:
            if not state["semantic_analysis"].get("skipped", False):
                response["analysis_details"] = {
                    "attack_techniques": state["semantic_analysis"].get("attack_techniques", []),
                    "attacker_goal": state["semantic_analysis"].get("attacker_goal", "Unknown"),
                    "sophistication": state["semantic_analysis"].get("sophistication", 0)
                }

        return response

    def invoke(self, state: AgenticGuardState) -> AgenticGuardState:
        """Process the state and generate the final response.

        Args:
            state: Current state of the workflow

        Returns:
            Updated state with security response
        """
        start_time = time.time()

        # Check if validation passed
        if not state.get("schema_valid", False):
            # Create error response
            state["security_response"] = {
                "error": "Validation failed",
                "validation_errors": state.get("validation_errors", []),
                "threat_detected": False,
                "threat_level": "UNKNOWN",
                "recommended_action": "flag",
                "explanation": "Unable to process due to validation errors. Manual review required."
            }
            state["recommended_action"] = "flag"

            # Update agent trace
            if "agent_trace" not in state:
                state["agent_trace"] = []
            state["agent_trace"].append("responder (error - validation failed)")

            return state

        # Determine recommended action
        action = self._determine_action(state["threat_level"], state["final_confidence"])

        # Generate explanation
        explanation = self._generate_explanation(state, action)

        # Compile final response
        security_response = self._compile_response(state, action, explanation)

        # Update state
        state["security_response"] = security_response
        state["recommended_action"] = action

        # Update agent trace
        if "agent_trace" not in state:
            state["agent_trace"] = []
        state["agent_trace"].append(f"responder (action={action})")

        # Update processing time
        processing_time = time.time() - start_time
        state["processing_time"] = state.get("processing_time", 0) + processing_time

        # Update the processing time in the response
        state["security_response"]["processing_time_seconds"] = state["processing_time"]
        state["security_response"]["agent_trace"] = state["agent_trace"]

        return state
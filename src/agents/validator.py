"""Validator Agent - Schema validation and confidence aggregation."""

import time
from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from ..agents.state import AgenticGuardState


class SecurityResponseSchema(BaseModel):
    """Pydantic schema for validating the security response."""

    threat_detected: bool = Field(..., description="Whether a threat was detected")
    threat_level: str = Field(..., pattern="^(SAFE|SUSPICIOUS|MALICIOUS)$")
    threat_type: Optional[str] = Field(None, pattern="^(injection|jailbreak|exfiltration|null)$")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    recommended_action: str = Field(..., pattern="^(block|flag|allow)$")
    explanation: str = Field(..., min_length=10, max_length=1000)
    attack_patterns: List[str] = Field(default_factory=list)
    embedding_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time_seconds: float = Field(..., ge=0.0)
    agent_trace: List[str] = Field(default_factory=list)


class ValidatorAgent:
    """Validates state and aggregates confidence scores.

    This agent is deterministic (no LLM calls) and ensures
    data quality and proper confidence scoring.
    """

    def __init__(self):
        """Initialize the Validator Agent."""
        # Weights for confidence aggregation
        self.weights = {
            "detector_confidence": 0.5,
            "embedding_similarity": 0.3,
            "pattern_score": 0.2
        }

    def _calculate_pattern_score(self, patterns: List[str]) -> float:
        """Calculate a score based on detected patterns.

        Args:
            patterns: List of detected attack patterns

        Returns:
            Pattern score between 0.0 and 1.0
        """
        if not patterns:
            return 0.0

        # More patterns = higher score, with diminishing returns
        pattern_count = len(patterns)
        score = min(pattern_count * 0.2, 1.0)
        return score

    def _aggregate_confidence(self, state: AgenticGuardState) -> float:
        """Aggregate confidence scores using weighted combination.

        Uses IQR-based approach to handle outliers and provide
        robust confidence estimation.

        Args:
            state: Current workflow state

        Returns:
            Aggregated confidence score
        """
        scores = []
        weights = []

        # Detector confidence
        if "detector_confidence" in state and state["detector_confidence"] is not None:
            scores.append(state["detector_confidence"])
            weights.append(self.weights["detector_confidence"])

        # Embedding similarity (if available)
        if "embedding_similarity" in state and state["embedding_similarity"] is not None:
            scores.append(state["embedding_similarity"])
            weights.append(self.weights["embedding_similarity"])

        # Pattern score
        pattern_score = self._calculate_pattern_score(state.get("attack_patterns", []))
        scores.append(pattern_score)
        weights.append(self.weights["pattern_score"])

        # Handle edge cases
        if not scores:
            return 0.5  # Default neutral confidence

        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Calculate weighted average
        weighted_avg = np.average(scores, weights=weights)

        # Apply IQR-based adjustment for robustness
        if len(scores) > 1:
            q1, q3 = np.percentile(scores, [25, 75])
            iqr = q3 - q1

            # Adjust confidence if there's high variance
            if iqr > 0.3:  # High disagreement between scores
                # Move confidence toward center (more uncertain)
                weighted_avg = weighted_avg * 0.8 + 0.5 * 0.2

        return float(min(max(weighted_avg, 0.0), 1.0))

    def _validate_required_fields(self, state: AgenticGuardState) -> tuple[bool, List[str]]:
        """Validate that all required fields are present in state.

        Args:
            state: Current workflow state

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        required_fields = [
            "user_prompt",
            "threat_level",
            "detector_confidence",
            "processing_time"
        ]

        for field in required_fields:
            if field not in state or state[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate field values
        if "threat_level" in state:
            if state["threat_level"] not in ["SAFE", "SUSPICIOUS", "MALICIOUS"]:
                errors.append(f"Invalid threat_level: {state['threat_level']}")

        if "detector_confidence" in state:
            if not (0 <= state["detector_confidence"] <= 1):
                errors.append(f"Invalid detector_confidence: {state['detector_confidence']}")

        return len(errors) == 0, errors

    def _validate_schema(self, response_data: dict) -> tuple[bool, List[str]]:
        """Validate response data against the schema.

        Args:
            response_data: Response data to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            SecurityResponseSchema(**response_data)
            return True, []
        except ValidationError as e:
            errors = [str(err) for err in e.errors()]
            return False, errors

    def invoke(self, state: AgenticGuardState) -> AgenticGuardState:
        """Process the state and perform validation.

        Args:
            state: Current state of the workflow

        Returns:
            Updated state with validation results
        """
        start_time = time.time()

        # Validate required fields
        fields_valid, field_errors = self._validate_required_fields(state)

        # Calculate aggregated confidence
        final_confidence = self._aggregate_confidence(state)

        # Determine if we can build a valid response
        can_build_response = fields_valid and len(field_errors) == 0

        # Update state
        state["schema_valid"] = can_build_response
        state["validation_errors"] = field_errors
        state["final_confidence"] = final_confidence

        # Prepare data for schema validation (preview)
        if can_build_response:
            response_preview = {
                "threat_detected": state["threat_level"] != "SAFE",
                "threat_level": state["threat_level"],
                "threat_type": state.get("threat_type", "null"),
                "confidence_score": final_confidence,
                "recommended_action": self._determine_action(state["threat_level"], final_confidence),
                "explanation": "Pending detailed analysis by response agent.",  # Will be filled by Response Agent
                "attack_patterns": state.get("attack_patterns", []),
                "embedding_similarity": state.get("embedding_similarity"),
                "processing_time_seconds": state.get("processing_time", 0),
                "agent_trace": state.get("agent_trace", [])
            }

            # Validate the response structure
            schema_valid, schema_errors = self._validate_schema(response_preview)
            if not schema_valid:
                state["schema_valid"] = False
                state["validation_errors"].extend(schema_errors)

        # Update agent trace - for LangGraph annotated fields, return new items only
        state["agent_trace"] = [
            f"validator (confidence={final_confidence:.2f}, valid={state['schema_valid']})"
        ]

        # Update processing time
        processing_time = time.time() - start_time
        state["processing_time"] = state.get("processing_time", 0) + processing_time

        return state

    def _determine_action(self, threat_level: str, confidence: float) -> str:
        """Determine recommended action based on threat level and confidence.

        Args:
            threat_level: Detected threat level
            confidence: Aggregated confidence score

        Returns:
            Recommended action
        """
        if threat_level == "MALICIOUS" and confidence > 0.8:
            return "block"
        elif threat_level == "MALICIOUS" or (threat_level == "SUSPICIOUS" and confidence > 0.7):
            return "flag"
        else:
            return "allow"
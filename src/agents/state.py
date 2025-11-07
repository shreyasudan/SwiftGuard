"""State definition for AgenticGuard multi-agent workflow."""

from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator


class AgenticGuardState(TypedDict):
    """State passed between agents in the workflow.

    This state object flows through all agents and accumulates
    information as each agent processes the user prompt.
    """

    # Input
    user_prompt: str

    # Detector outputs
    threat_level: str  # "SAFE", "SUSPICIOUS", "MALICIOUS"
    threat_type: Optional[str]  # "injection", "jailbreak", "exfiltration", None
    detector_confidence: float

    # Analyzer outputs (optional, skipped for SAFE prompts)
    semantic_analysis: Optional[Dict[str, Any]]
    attack_patterns: List[str]
    embedding_similarity: Optional[float]

    # Validator outputs
    schema_valid: bool
    validation_errors: List[str]
    final_confidence: float

    # Response outputs
    security_response: Dict[str, Any]
    recommended_action: str  # "block", "flag", "allow"

    # Metadata
    processing_time: float
    agent_trace: Annotated[Sequence[str], operator.add]  # Track agent execution
    messages: Annotated[Sequence[BaseMessage], operator.add]  # LLM message history
"""LangGraph workflow orchestration for AgenticGuard."""

import os
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from ..agents.state import AgenticGuardState
from ..agents.detector import DetectorAgent
from ..agents.analyzer import AnalyzerAgent
from ..agents.validator import ValidatorAgent
from ..agents.responder import ResponseAgent
from ..utils.embeddings import EmbeddingAnalyzer
from ..utils.llm_factory import get_llm, get_embedding_model


# Load environment variables
load_dotenv()


class AgenticGuardWorkflow:
    """Orchestrates the multi-agent workflow for adversarial prompt detection."""

    def __init__(
        self,
        detector_llm: Optional[Any] = None,
        analyzer_llm: Optional[Any] = None,
        responder_llm: Optional[Any] = None,
        embeddings_model: Optional[Any] = None
    ):
        """Initialize the workflow with all agents.

        Args:
            detector_llm: Optional LLM for detector agent
            analyzer_llm: Optional LLM for analyzer agent
            responder_llm: Optional LLM for responder agent
            embeddings_model: Optional embeddings model
        """
        # Initialize embedding analyzer
        embedding_analyzer = EmbeddingAnalyzer(embeddings_model)

        # Initialize agents
        self.detector = DetectorAgent(detector_llm)
        self.analyzer = AnalyzerAgent(analyzer_llm, embedding_analyzer)
        self.validator = ValidatorAgent()
        self.responder = ResponseAgent(responder_llm)

        # Build the workflow graph
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Compiled LangGraph application
        """
        # Create the graph
        workflow = StateGraph(AgenticGuardState)

        # Add nodes for each agent
        workflow.add_node("detector", self.detector.invoke)
        workflow.add_node("analyzer", self.analyzer.invoke)
        workflow.add_node("validator", self.validator.invoke)
        workflow.add_node("responder", self.responder.invoke)

        # Define edges (flow between agents)
        workflow.set_entry_point("detector")

        # Detector -> Analyzer
        workflow.add_edge("detector", "analyzer")

        # Analyzer -> Validator
        workflow.add_edge("analyzer", "validator")

        # Validator -> Responder
        workflow.add_edge("validator", "responder")

        # Responder -> END
        workflow.add_edge("responder", END)

        # Compile the graph
        return workflow.compile()

    def run(self, user_prompt: str) -> Dict[str, Any]:
        """Run the workflow on a user prompt.

        Args:
            user_prompt: User input to analyze

        Returns:
            Security response dictionary
        """
        # Initialize state
        initial_state = {
            "user_prompt": user_prompt,
            "agent_trace": [],
            "messages": [],
            "processing_time": 0.0
        }

        # Run the workflow
        start_time = time.time()

        try:
            # Execute the workflow
            result = self.app.invoke(initial_state)

            # Extract the security response
            if "security_response" in result:
                return result["security_response"]
            else:
                # Fallback response if something went wrong
                return {
                    "error": "Workflow execution failed",
                    "threat_detected": False,
                    "threat_level": "UNKNOWN",
                    "recommended_action": "flag",
                    "explanation": "Unable to process prompt due to workflow error.",
                    "processing_time_seconds": time.time() - start_time
                }

        except Exception as e:
            # Error response
            return {
                "error": str(e),
                "threat_detected": False,
                "threat_level": "ERROR",
                "recommended_action": "flag",
                "explanation": f"System error occurred: {str(e)}",
                "processing_time_seconds": time.time() - start_time
            }


def run_agentic_guard(user_prompt: str, mode: str = "precision") -> Dict[str, Any]:
    """Main entry point for AgenticGuard.

    Args:
        user_prompt: User input to analyze
        mode: Execution mode - "classic" (fast) or "precision" (accurate)

    Returns:
        Security response dictionary with threat assessment
    """
    # Configure LLMs based on mode
    if mode == "classic":
        # Fast mode - use smaller models
        detector_llm = get_llm(model_type="fast", temperature=0, max_tokens=500)
        analyzer_llm = get_llm(model_type="fast", temperature=0, max_tokens=800)
        responder_llm = get_llm(model_type="fast", temperature=0.3, max_tokens=300)
    else:
        # Precision mode - use larger models
        detector_llm = get_llm(model_type="fast", temperature=0, max_tokens=500)
        analyzer_llm = get_llm(model_type="powerful", temperature=0, max_tokens=1000)
        responder_llm = get_llm(model_type="balanced", temperature=0.3, max_tokens=500)

    # Create and run workflow
    workflow = AgenticGuardWorkflow(
        detector_llm=detector_llm,
        analyzer_llm=analyzer_llm,
        responder_llm=responder_llm
    )

    return workflow.run(user_prompt)


# Convenience function for direct import
def analyze_prompt(prompt: str, mode: str = "precision") -> Dict[str, Any]:
    """Analyze a prompt for adversarial content.

    Args:
        prompt: User prompt to analyze
        mode: "classic" for speed, "precision" for accuracy

    Returns:
        Security analysis results
    """
    return run_agentic_guard(prompt, mode)
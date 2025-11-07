"""Analyzer Agent - Deep semantic analysis of flagged prompts."""

import json
import time
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from ..agents.state import AgenticGuardState
from ..config.prompts import ANALYZER_PROMPT
from ..utils.embeddings import EmbeddingAnalyzer
from ..utils.llm_factory import get_llm


class AnalyzerAgent:
    """Performs deep semantic analysis on potentially adversarial prompts.

    This agent is conditionally executed - skipped for SAFE prompts.
    Uses embeddings and LLM analysis for comprehensive threat assessment.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        embedding_analyzer: Optional[EmbeddingAnalyzer] = None
    ):
        """Initialize the Analyzer Agent.

        Args:
            llm: Optional LLM instance. Defaults to factory balanced model.
            embedding_analyzer: Optional embedding analyzer instance.
        """
        self.llm = llm or get_llm(model_type="balanced", temperature=0, max_tokens=1000)
        self.embedding_analyzer = embedding_analyzer or EmbeddingAnalyzer()

    def _should_skip(self, state: AgenticGuardState) -> bool:
        """Determine if analysis should be skipped.

        Args:
            state: Current workflow state

        Returns:
            True if analysis should be skipped
        """
        return state.get("threat_level") == "SAFE"

    def _perform_llm_analysis(self, state: AgenticGuardState) -> Dict[str, Any]:
        """Perform deep analysis using LLM.

        Args:
            state: Current workflow state

        Returns:
            Analysis results from LLM
        """
        try:
            formatted_prompt = ANALYZER_PROMPT.format(
                user_prompt=state["user_prompt"],
                threat_level=state["threat_level"],
                threat_type=state.get("threat_type", "null")
            )

            messages = [
                SystemMessage(content="You are a security analysis agent. Respond only with valid JSON."),
                HumanMessage(content=formatted_prompt)
            ]

            response = self.llm.invoke(messages)

            # Parse JSON response
            result = json.loads(response.content)

            # Validate required fields
            required_fields = ["attack_techniques", "attacker_goal", "sophistication", "specific_patterns", "semantic_analysis"]
            for field in required_fields:
                if field not in result:
                    result[field] = self._get_default_value(field)

            return result

        except Exception as e:
            # Return default analysis on error
            return {
                "attack_techniques": ["unknown"],
                "attacker_goal": "Unable to determine",
                "sophistication": 5,
                "specific_patterns": [],
                "semantic_analysis": f"Analysis error: {str(e)}"
            }

    def _get_default_value(self, field: str) -> Any:
        """Get default value for a missing field.

        Args:
            field: Field name

        Returns:
            Default value for the field
        """
        defaults = {
            "attack_techniques": [],
            "attacker_goal": "Unknown",
            "sophistication": 5,
            "specific_patterns": [],
            "semantic_analysis": "Analysis incomplete"
        }
        return defaults.get(field, None)

    def invoke(self, state: AgenticGuardState) -> AgenticGuardState:
        """Process the state and perform deep analysis.

        Args:
            state: Current state of the workflow

        Returns:
            Updated state with analysis results
        """
        start_time = time.time()

        # Check if we should skip analysis
        if self._should_skip(state):
            # Set default values for skipped analysis
            state["semantic_analysis"] = {"skipped": True, "reason": "Prompt classified as SAFE"}
            state["embedding_similarity"] = None

            # Update agent trace
            if "agent_trace" not in state:
                state["agent_trace"] = []
            state["agent_trace"].append("analyzer (skipped - SAFE)")

            # Update processing time
            state["processing_time"] = state.get("processing_time", 0) + (time.time() - start_time)

            return state

        # Compute embedding similarity
        try:
            similarity = self.embedding_analyzer.compute_similarity(state["user_prompt"])
            most_similar = self.embedding_analyzer.find_most_similar(state["user_prompt"], top_k=1)
        except Exception as e:
            print(f"Embedding analysis failed: {e}")
            similarity = 0.0
            most_similar = []

        # Perform LLM analysis
        llm_analysis = self._perform_llm_analysis(state)

        # Extract and combine attack patterns
        existing_patterns = state.get("attack_patterns", [])
        new_patterns = llm_analysis.get("specific_patterns", [])
        combined_patterns = list(set(existing_patterns + new_patterns))

        # Update state with analysis results
        state["semantic_analysis"] = {
            "attack_techniques": llm_analysis["attack_techniques"],
            "attacker_goal": llm_analysis["attacker_goal"],
            "sophistication": llm_analysis["sophistication"],
            "detailed_analysis": llm_analysis["semantic_analysis"],
            "most_similar_adversarial": most_similar[0] if most_similar else None
        }
        state["attack_patterns"] = combined_patterns
        state["embedding_similarity"] = similarity

        # Update agent trace
        if "agent_trace" not in state:
            state["agent_trace"] = []
        state["agent_trace"].append(
            f"analyzer (similarity={similarity:.2f}, sophistication={llm_analysis['sophistication']})"
        )

        # Update processing time
        processing_time = time.time() - start_time
        state["processing_time"] = state.get("processing_time", 0) + processing_time

        return state
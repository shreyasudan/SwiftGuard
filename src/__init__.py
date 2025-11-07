"""AgenticGuard - Multi-agent adversarial prompt detection system."""

from .orchestration import run_agentic_guard, analyze_prompt
from .agents.state import AgenticGuardState

__version__ = "1.0.0"

__all__ = [
    "run_agentic_guard",
    "analyze_prompt",
    "AgenticGuardState"
]
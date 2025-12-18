import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.models import EvaluationResult, ResearchReport

if TYPE_CHECKING:
    from haiku.rag.config.models import AppConfig
    from haiku.rag.graph.agui.emitter import AGUIEmitter


class HumanDecision(BaseModel):
    """Human decision input for interactive research."""

    action: Literal[
        "search", "synthesize", "modify_questions", "add_questions", "chat", "research"
    ]
    questions: list[str] | None = None
    message: str | None = None
    research_question: str | None = None


@dataclass
class ResearchDeps:
    """Dependencies for research graph execution."""

    client: HaikuRAG
    agui_emitter: "AGUIEmitter[ResearchState, ResearchReport] | None" = None
    semaphore: asyncio.Semaphore | None = None
    human_input_queue: asyncio.Queue[HumanDecision] | None = None
    interactive: bool = False

    def emit_log(self, message: str, state: "ResearchState | None" = None) -> None:
        """Emit a log message through AG-UI events."""
        if self.agui_emitter:
            self.agui_emitter.log(message)
            if state:
                self.agui_emitter.update_state(state)


class ResearchState(BaseModel):
    """Research graph state model."""

    model_config = {"arbitrary_types_allowed": True}

    context: ResearchContext = Field(
        description="Shared research context with questions and QA responses"
    )
    iterations: int = Field(default=0, description="Current iteration number")
    max_iterations: int = Field(default=3, description="Maximum allowed iterations")
    confidence_threshold: float = Field(
        default=0.8, description="Confidence threshold for completion", ge=0.0, le=1.0
    )
    max_concurrency: int = Field(
        default=1, description="Maximum concurrent search operations", ge=1
    )
    last_eval: EvaluationResult | None = Field(
        default=None, description="Last evaluation result"
    )
    search_filter: str | None = Field(
        default=None, description="SQL WHERE clause to filter search results"
    )

    @classmethod
    def from_config(
        cls,
        context: ResearchContext,
        config: "AppConfig",
        max_iterations: int | None = None,
        confidence_threshold: float | None = None,
    ) -> "ResearchState":
        """Create a ResearchState from an AppConfig.

        Args:
            context: The ResearchContext containing the question
            config: The AppConfig object
            max_iterations: Override max iterations (None uses config default)
            confidence_threshold: Override threshold (None uses config, 0.0 disables check)
        """
        return cls(
            context=context,
            max_iterations=max_iterations
            if max_iterations is not None
            else config.research.max_iterations,
            confidence_threshold=confidence_threshold
            if confidence_threshold is not None
            else config.research.confidence_threshold,
            max_concurrency=config.research.max_concurrency,
        )

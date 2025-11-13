import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.models import (
    EvaluationResult,
    InsightAnalysis,
    ResearchReport,
)

if TYPE_CHECKING:
    from haiku.rag.config.models import AppConfig
    from haiku.rag.graph.agui.emitter import AGUIEmitter


@dataclass
class ResearchDeps:
    """Dependencies for research graph execution."""

    client: HaikuRAG
    agui_emitter: "AGUIEmitter[ResearchState, ResearchReport] | None" = None
    semaphore: asyncio.Semaphore | None = None

    def emit_log(self, message: str, state: "ResearchState | None" = None) -> None:
        """Emit a log message through AG-UI events.

        Args:
            message: The message to log
            state: Optional state to include in state update
        """
        if self.agui_emitter:
            self.agui_emitter.log(message)
            if state:
                self.agui_emitter.update_state(state)


class ResearchState(BaseModel):
    """Research graph state model.

    Fully JSON-serializable Pydantic model suitable for AG-UI state synchronization.
    """

    model_config = {"arbitrary_types_allowed": True}

    context: ResearchContext = Field(
        description="Shared research context with questions, insights, and gaps"
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
    last_analysis: InsightAnalysis | None = Field(
        default=None, description="Last insight analysis"
    )

    @classmethod
    def from_config(
        cls, context: ResearchContext, config: "AppConfig"
    ) -> "ResearchState":
        """Create a ResearchState from an AppConfig.

        Args:
            context: The ResearchContext containing the question and settings
            config: The AppConfig object (uses config.research for state parameters)

        Returns:
            A configured ResearchState instance
        """
        return cls(
            context=context,
            max_iterations=config.research.max_iterations,
            confidence_threshold=config.research.confidence_threshold,
            max_concurrency=config.research.max_concurrency,
        )

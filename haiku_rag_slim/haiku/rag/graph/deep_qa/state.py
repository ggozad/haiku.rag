import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.deep_qa.dependencies import DeepQAContext
from haiku.rag.graph.deep_qa.models import DeepQAAnswer

if TYPE_CHECKING:
    from haiku.rag.config.models import AppConfig
    from haiku.rag.graph.agui.emitter import AGUIEmitter


@dataclass
class DeepQADeps:
    client: HaikuRAG
    agui_emitter: "AGUIEmitter[DeepQAState, DeepQAAnswer] | None" = None
    semaphore: asyncio.Semaphore | None = None


class DeepQAState(BaseModel):
    """Deep QA state for multi-agent question answering."""

    model_config = {"arbitrary_types_allowed": True}

    context: DeepQAContext = Field(description="Shared QA context")
    max_sub_questions: int = Field(
        default=3, description="Maximum number of sub-questions"
    )
    max_iterations: int = Field(
        default=2, description="Maximum number of QA iterations"
    )
    max_concurrency: int = Field(
        default=1, description="Maximum parallel sub-question searches"
    )
    iterations: int = Field(default=0, description="Current iteration number")

    @classmethod
    def from_config(cls, context: DeepQAContext, config: "AppConfig") -> "DeepQAState":
        """Create a DeepQAState from an AppConfig.

        Args:
            context: The DeepQAContext containing the question and settings
            config: The AppConfig object (uses config.qa for state parameters)

        Returns:
            A configured DeepQAState instance
        """
        return cls(
            context=context,
            max_sub_questions=config.qa.max_sub_questions,
            max_iterations=config.qa.max_iterations,
            max_concurrency=config.qa.max_concurrency,
        )

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from haiku.rag.agents.research.dependencies import ResearchContext
from haiku.rag.client import HaikuRAG

if TYPE_CHECKING:
    from haiku.rag.config.models import AppConfig


@dataclass
class ResearchDeps:
    """Dependencies for research graph execution."""

    client: HaikuRAG
    semaphore: asyncio.Semaphore | None = None


class ResearchState(BaseModel):
    """Research graph state model."""

    model_config = {"arbitrary_types_allowed": True}

    context: ResearchContext = Field(
        description="Shared research context with questions and QA responses"
    )
    iterations: int = Field(default=0, description="Current iteration number")
    max_iterations: int = Field(default=3, description="Maximum allowed iterations")
    max_concurrency: int = Field(
        default=1, description="Maximum concurrent search operations", ge=1
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
    ) -> "ResearchState":
        """Create a ResearchState from an AppConfig.

        Args:
            context: The ResearchContext containing the question
            config: The AppConfig object
            max_iterations: Override max iterations (None uses config default)
        """
        return cls(
            context=context,
            max_iterations=max_iterations
            if max_iterations is not None
            else config.research.max_iterations,
            max_concurrency=config.research.max_concurrency,
        )

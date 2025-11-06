import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.research.dependencies import ResearchContext
from haiku.rag.research.models import EvaluationResult, InsightAnalysis
from haiku.rag.research.stream import ResearchStream

if TYPE_CHECKING:
    from haiku.rag.config.models import AppConfig


@dataclass
class ResearchDeps:
    client: HaikuRAG
    console: Console | None = None
    stream: ResearchStream | None = None
    semaphore: asyncio.Semaphore | None = None

    def emit_log(self, message: str, state: "ResearchState | None" = None) -> None:
        if self.console:
            self.console.print(message)
        if self.stream:
            self.stream.log(message, state)


@dataclass
class ResearchState:
    context: ResearchContext
    iterations: int = 0
    max_iterations: int = 3
    confidence_threshold: float = 0.8
    max_concurrency: int = 1
    last_eval: EvaluationResult | None = None
    last_analysis: InsightAnalysis | None = None

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

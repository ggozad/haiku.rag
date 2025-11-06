import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.qa.deep.dependencies import DeepQAContext

if TYPE_CHECKING:
    from haiku.rag.config.models import AppConfig


@dataclass
class DeepQADeps:
    client: HaikuRAG
    console: Console | None = None
    semaphore: asyncio.Semaphore | None = None

    def emit_log(self, message: str, state: "DeepQAState | None" = None) -> None:
        if self.console:
            self.console.print(message)


@dataclass
class DeepQAState:
    context: DeepQAContext
    max_sub_questions: int = 3
    max_iterations: int = 2
    max_concurrency: int = 1
    iterations: int = 0

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

import asyncio
from dataclasses import dataclass

from haiku.rag.client import HaikuRAG
from haiku.rag.qa.deep.dependencies import DeepQAContext
from rich.console import Console


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

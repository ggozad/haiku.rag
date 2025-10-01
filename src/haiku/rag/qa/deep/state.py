from dataclasses import dataclass

from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.qa.deep.dependencies import DeepQAContext


@dataclass
class DeepQADeps:
    client: HaikuRAG
    console: Console | None = None

    def emit_log(self, message: str, state: "DeepQAState | None" = None) -> None:
        if self.console:
            self.console.print(message)


@dataclass
class DeepQAState:
    context: DeepQAContext
    max_sub_questions: int = 3
    max_iterations: int = 2
    max_concurrency: int = 3
    iterations: int = 0

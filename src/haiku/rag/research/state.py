from dataclasses import dataclass

from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.research.dependencies import ResearchContext
from haiku.rag.research.models import EvaluationResult, InsightAnalysis
from haiku.rag.research.stream import ResearchStream


@dataclass
class ResearchDeps:
    client: HaikuRAG
    console: Console | None = None
    stream: ResearchStream | None = None

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
    max_concurrency: int = 1
    confidence_threshold: float = 0.8
    last_eval: EvaluationResult | None = None
    last_analysis: InsightAnalysis | None = None

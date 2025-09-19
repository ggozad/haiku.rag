from dataclasses import dataclass, field

from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.research.dependencies import ResearchContext
from haiku.rag.research.models import EvaluationResult


@dataclass
class ResearchDeps:
    client: HaikuRAG
    console: Console | None = None


@dataclass
class ResearchState:
    question: str
    context: ResearchContext
    sub_questions: list[str] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 3
    max_concurrency: int = 1
    confidence_threshold: float = 0.8
    last_eval: EvaluationResult | None = None

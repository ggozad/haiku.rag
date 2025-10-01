from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field
from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.models import SearchAnswer


@runtime_checkable
class GraphContext(Protocol):
    """Protocol for graph context objects."""

    original_question: str
    sub_questions: list[str]
    qa_responses: list[SearchAnswer]

    def add_qa_response(self, qa: SearchAnswer) -> None: ...


class BaseGraphDeps(BaseModel):
    """Base dependencies for graph nodes."""

    model_config = {"arbitrary_types_allowed": True}

    client: HaikuRAG = Field(description="RAG client for document operations")
    console: Console | None = None

    def emit_log(self, message: str) -> None:
        if self.console:
            self.console.print(message)

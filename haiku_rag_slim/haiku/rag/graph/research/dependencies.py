from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import SearchResult

if TYPE_CHECKING:
    from haiku.rag.graph.research.models import SearchAnswer


class ResearchContext(BaseModel):
    """Context shared across research agents."""

    original_question: str = Field(description="The original research question")
    sub_questions: list[str] = Field(
        default_factory=list, description="Decomposed sub-questions"
    )
    qa_responses: list[Any] = Field(
        default_factory=list, description="Structured QA pairs used during research"
    )

    def add_qa_response(self, qa: "SearchAnswer") -> None:
        """Add a structured QA response."""
        self.qa_responses.append(qa)


class ResearchDependencies(BaseModel):
    """Dependencies for research agents with multi-agent context."""

    model_config = {"arbitrary_types_allowed": True}

    client: HaikuRAG = Field(description="RAG client for document operations")
    context: ResearchContext = Field(description="Shared research context")
    search_results: list[SearchResult] = Field(
        default_factory=list, description="Search results for citation resolution"
    )

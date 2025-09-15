"""Shared dependencies for multi-agent research workflow."""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from haiku.rag.client import HaikuRAG

if TYPE_CHECKING:
    from haiku.rag.research.base import SearchResult


class ResearchContext(BaseModel):
    """Context shared across research agents."""

    original_question: str = Field(description="The original research question")
    sub_questions: list[str] = Field(
        default_factory=list, description="Decomposed sub-questions"
    )
    search_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Accumulated search results"
    )
    qa_responses: list[dict[str, Any]] = Field(
        default_factory=list, description="Question-answer pairs with sources"
    )
    insights: list[str] = Field(
        default_factory=list, description="Key insights discovered"
    )
    gaps: list[str] = Field(
        default_factory=list, description="Identified information gaps"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list, description="Generated follow-up questions"
    )

    def add_search_result(self, query: str, results: list["SearchResult"]) -> None:
        """Add search results to context."""
        self.search_results.append(
            {
                "query": query,
                "results": results,
            }
        )

    def add_qa_response(
        self, question: str, answer: str, sources: list["SearchResult"]
    ) -> None:
        """Add a QA response with its source documents."""
        self.qa_responses.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
            }
        )

    def add_insight(self, insight: str) -> None:
        """Add a key insight."""
        if insight not in self.insights:
            self.insights.append(insight)

    def add_gap(self, gap: str) -> None:
        """Identify an information gap."""
        if gap not in self.gaps:
            self.gaps.append(gap)


class ResearchDependencies(BaseModel):
    """Dependencies for research agents with multi-agent context."""

    model_config = {"arbitrary_types_allowed": True}

    client: HaikuRAG = Field(description="RAG client for document operations")
    context: ResearchContext = Field(description="Shared research context")

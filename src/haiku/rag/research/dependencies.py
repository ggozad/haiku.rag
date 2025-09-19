from pydantic import BaseModel, Field
from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.research.models import SearchAnswer


class ResearchContext(BaseModel):
    """Context shared across research agents."""

    original_question: str = Field(description="The original research question")
    sub_questions: list[str] = Field(
        default_factory=list, description="Decomposed sub-questions"
    )
    qa_responses: list[SearchAnswer] = Field(
        default_factory=list, description="Structured QA pairs used during research"
    )
    insights: list[str] = Field(
        default_factory=list, description="Key insights discovered"
    )
    gaps: list[str] = Field(
        default_factory=list, description="Identified information gaps"
    )

    def add_qa_response(self, qa: SearchAnswer) -> None:
        """Add a structured QA response (minimal context already included)."""
        self.qa_responses.append(qa)

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
    console: Console | None = None

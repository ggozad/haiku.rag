from pydantic import BaseModel, Field

from haiku.rag.agents.research.models import Citation


class QAResult(BaseModel):
    """Result from the QA toolset."""

    question: str = Field(description="The question that was answered")
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(
        default=1.0,
        description="Confidence score for this answer (0-1)",
        ge=0.0,
        le=1.0,
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations supporting the answer",
    )

    @property
    def sources(self) -> list[str]:
        """Source names for display."""
        return list(
            dict.fromkeys(c.document_title or c.document_uri for c in self.citations)
        )


class AnalysisResult(BaseModel):
    """Result from the analysis toolset (RLM execution)."""

    answer: str = Field(description="The answer produced by analysis")
    code_executed: bool = Field(
        default=True,
        description="Whether code was executed to produce this answer",
    )

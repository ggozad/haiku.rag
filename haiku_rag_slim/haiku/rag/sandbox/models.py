from pydantic import BaseModel, Field

from haiku.rag.store.models.citation import Citation


class AnalysisResult(BaseModel):
    """Result from analysis execution with resolved citations.

    Executed code is tracked on ``AnalysisState.executions`` (populated by the
    analysis capability's code-execution tool). Consumers that need the program
    should pull it from capability state."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)

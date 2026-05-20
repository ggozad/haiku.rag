from pydantic import BaseModel, Field

from haiku.rag.store.models.citation import Citation


class AnalysisResult(BaseModel):
    """Result from analysis execution with resolved citations.

    Executed code is tracked on ``AnalysisState.executions`` (populated by the
    analysis skill's ``execute_code`` tool). Consumers that need the program
    should pull it from the skill state."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)

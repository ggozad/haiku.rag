"""Common models used across different graph implementations."""

from pydantic import BaseModel, Field, field_validator


class ResearchPlan(BaseModel):
    """A structured research plan with sub-questions to explore."""

    sub_questions: list[str] = Field(
        ...,
        description="Specific questions to research, phrased as complete questions",
    )

    @field_validator("sub_questions")
    @classmethod
    def validate_sub_questions(cls, v: list[str]) -> list[str]:
        if len(v) < 1:
            raise ValueError("Must have at least 1 sub-question")
        if len(v) > 12:
            raise ValueError("Cannot have more than 12 sub-questions")
        return v


class SearchAnswer(BaseModel):
    """Answer from a search operation with sources."""

    query: str = Field(..., description="The question that was answered")
    answer: str = Field(..., description="The comprehensive answer to the question")
    context: list[str] = Field(
        default_factory=list,
        description="Relevant snippets that directly support the answer",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source URIs or titles that contributed to this answer",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for this answer (0-1)",
        ge=0.0,
        le=1.0,
    )

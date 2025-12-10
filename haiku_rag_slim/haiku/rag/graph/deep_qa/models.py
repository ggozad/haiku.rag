from pydantic import BaseModel, Field

from haiku.rag.graph.common.models import Citation


class DeepQAEvaluation(BaseModel):
    is_sufficient: bool = Field(
        description="Whether we have sufficient information to answer the question"
    )
    reasoning: str = Field(description="Explanation of the sufficiency assessment")
    new_questions: list[str] = Field(
        description="Additional sub-questions needed if insufficient",
        default_factory=list,
    )


class DeepQAAnswer(BaseModel):
    """Final deep QA answer with resolved citations."""

    answer: str = Field(description="The comprehensive answer to the question")
    citations: list[Citation] = Field(
        default_factory=list, description="Resolved citations for the answer"
    )

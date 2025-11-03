from pydantic import BaseModel, Field


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
    answer: str = Field(description="The comprehensive answer to the question")
    sources: list[str] = Field(
        description="Document titles or URIs used to generate the answer",
        default_factory=list,
    )

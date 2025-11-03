from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    main_question: str
    sub_questions: list[str]


class SearchAnswer(BaseModel):
    query: str = Field(description="The search query that was performed")
    answer: str = Field(description="The answer generated based on the context")
    context: list[str] = Field(
        description=(
            "Only the minimal set of relevant snippets (verbatim) that directly "
            "support the answer"
        )
    )
    sources: list[str] = Field(
        description=(
            "Document titles (if available) or URIs corresponding to the"
            " snippets actually used in the answer (one per snippet; omit if none)"
        ),
        default_factory=list,
    )


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

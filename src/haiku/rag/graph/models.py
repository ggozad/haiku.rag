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

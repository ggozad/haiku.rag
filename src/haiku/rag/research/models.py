from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    main_question: str
    sub_questions: list[str]


class SearchAnswer(BaseModel):
    """Structured output for the SearchSpecialist agent."""

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


class EvaluationResult(BaseModel):
    """Result of analysis and evaluation."""

    key_insights: list[str] = Field(
        description="Main insights extracted from the research so far"
    )
    new_questions: list[str] = Field(
        description="New sub-questions to add to the research (max 3)",
        max_length=3,
        default=[],
    )
    confidence_score: float = Field(
        description="Confidence level in the completeness of research (0-1)",
        ge=0.0,
        le=1.0,
    )
    is_sufficient: bool = Field(
        description="Whether the research is sufficient to answer the original question"
    )
    reasoning: str = Field(
        description="Explanation of why the research is or isn't complete"
    )


class ResearchReport(BaseModel):
    """Final research report structure."""

    title: str = Field(description="Concise title for the research")
    executive_summary: str = Field(description="Brief overview of key findings")
    main_findings: list[str] = Field(
        description="Primary research findings with supporting evidence"
    )
    conclusions: list[str] = Field(description="Evidence-based conclusions")
    limitations: list[str] = Field(
        description="Limitations of the current research", default=[]
    )
    recommendations: list[str] = Field(
        description="Actionable recommendations based on findings", default=[]
    )
    sources_summary: str = Field(
        description="Summary of sources used and their reliability"
    )

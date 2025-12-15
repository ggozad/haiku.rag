from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Result of research sufficiency evaluation."""

    is_sufficient: bool = Field(
        description="Whether the research is sufficient to answer the original question"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level in the completeness of research (0-1)",
    )
    reasoning: str = Field(
        description="Explanation of why the research is or isn't complete"
    )
    new_questions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="New sub-questions to add to the research (max 3)",
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

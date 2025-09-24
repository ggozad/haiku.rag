import re
from enum import Enum

from pydantic import BaseModel, Field, model_validator

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _make_slug(text: str, prefix: str) -> str:
    """Generate a lowercase slug with the given prefix as fallback."""

    base = _SLUG_RE.sub("-", text.lower()).strip("-")
    if not base:
        base = prefix
    # Trim overly long slugs but keep enough entropy for readability
    return base[:48]


class InsightStatus(str, Enum):
    OPEN = "open"
    VALIDATED = "validated"
    TENTATIVE = "tentative"


class GapSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InsightRecord(BaseModel):
    """Structured insight with provenance and lifecycle metadata."""

    id: str | None = Field(
        default=None,
        description="Stable slug identifier for the insight (auto-generated if omitted)",
    )
    summary: str = Field(description="Concise description of the insight")
    status: InsightStatus = Field(
        default=InsightStatus.OPEN,
        description="Lifecycle status for the insight",
    )
    supporting_sources: list[str] = Field(
        default_factory=list,
        description="Source identifiers backing the insight",
    )
    originating_questions: list[str] = Field(
        default_factory=list,
        description="Research sub-questions that produced this insight",
    )
    notes: str | None = Field(
        default=None,
        description="Optional elaboration or caveats for the insight",
    )

    @model_validator(mode="after")
    def _set_defaults(self) -> "InsightRecord":
        if not self.id:
            self.id = _make_slug(self.summary, "insight")
        self.id = self.id.lower()
        self.supporting_sources = list(dict.fromkeys(self.supporting_sources))
        self.originating_questions = list(dict.fromkeys(self.originating_questions))
        return self


class GapRecord(BaseModel):
    """Structured representation of an identified research gap."""

    id: str | None = Field(
        default=None,
        description="Stable slug identifier for the gap (auto-generated if omitted)",
    )
    description: str = Field(description="Concrete statement of what is missing")
    severity: GapSeverity = Field(
        default=GapSeverity.MEDIUM,
        description="Severity of the gap for answering the main question",
    )
    blocking: bool = Field(
        default=True,
        description="Whether this gap blocks a confident answer",
    )
    resolved: bool = Field(
        default=False,
        description="Flag indicating if the gap has been resolved",
    )
    resolved_by: list[str] = Field(
        default_factory=list,
        description="Insight IDs or notes explaining how the gap was closed",
    )
    supporting_sources: list[str] = Field(
        default_factory=list,
        description="Sources confirming the gap status (e.g., evidence of absence)",
    )
    notes: str | None = Field(
        default=None,
        description="Optional clarification about the gap or follow-up actions",
    )

    @model_validator(mode="after")
    def _set_defaults(self) -> "GapRecord":
        if not self.id:
            self.id = _make_slug(self.description, "gap")
        self.id = self.id.lower()
        self.resolved_by = list(dict.fromkeys(self.resolved_by))
        self.supporting_sources = list(dict.fromkeys(self.supporting_sources))
        return self


class InsightAnalysis(BaseModel):
    """Output of the insight aggregation agent."""

    highlights: list[InsightRecord] = Field(
        default_factory=list,
        description="New or updated insights discovered this iteration",
    )
    gap_assessments: list[GapRecord] = Field(
        default_factory=list,
        description="New or updated gap records based on current evidence",
    )
    resolved_gaps: list[str] = Field(
        default_factory=list,
        description="Gap identifiers or descriptions considered resolved",
    )
    new_questions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three follow-up sub-questions to pursue next",
    )
    commentary: str = Field(
        description="Short narrative summary of the incremental findings",
    )


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
    gaps: list[str] = Field(
        description="Concrete information gaps that remain", default_factory=list
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

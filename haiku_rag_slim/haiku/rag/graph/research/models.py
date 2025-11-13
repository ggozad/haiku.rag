import uuid
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def _deduplicate_list(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    return list(dict.fromkeys(items))


class InsightStatus(str, Enum):
    OPEN = "open"
    VALIDATED = "validated"
    TENTATIVE = "tentative"


class GapSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrackedRecord(BaseModel):
    """Base model for tracked entities with sources and metadata."""

    model_config = {"validate_assignment": True}

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for the record",
    )
    supporting_sources: list[str] = Field(
        default_factory=list,
        description="Source identifiers backing this record",
    )
    notes: str | None = Field(
        default=None,
        description="Optional elaboration or caveats",
    )

    @field_validator("supporting_sources", mode="before")
    @classmethod
    def deduplicate_sources(cls, v: list[str]) -> list[str]:
        """Ensure supporting_sources has no duplicates."""
        return _deduplicate_list(v) if v else []


class InsightRecord(TrackedRecord):
    """Structured insight with provenance and lifecycle metadata."""

    summary: str = Field(description="Concise description of the insight")
    status: InsightStatus = Field(
        default=InsightStatus.OPEN,
        description="Lifecycle status for the insight",
    )
    originating_questions: list[str] = Field(
        default_factory=list,
        description="Research sub-questions that produced this insight",
    )

    @field_validator("originating_questions", mode="before")
    @classmethod
    def deduplicate_questions(cls, v: list[str]) -> list[str]:
        """Ensure originating_questions has no duplicates."""
        return _deduplicate_list(v) if v else []


class GapRecord(TrackedRecord):
    """Structured representation of an identified research gap."""

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

    @field_validator("resolved_by", mode="before")
    @classmethod
    def deduplicate_resolved_by(cls, v: list[str]) -> list[str]:
        """Ensure resolved_by has no duplicates."""
        return _deduplicate_list(v) if v else []


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

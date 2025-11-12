from collections.abc import Iterable

from pydantic import BaseModel, Field, PrivateAttr

from haiku.rag.client import HaikuRAG
from haiku.rag.graph.common.models import SearchAnswer
from haiku.rag.graph.research.models import (
    GapRecord,
    InsightAnalysis,
    InsightRecord,
)


class ResearchContext(BaseModel):
    """Context shared across research agents."""

    original_question: str = Field(description="The original research question")
    sub_questions: list[str] = Field(
        default_factory=list, description="Decomposed sub-questions"
    )
    qa_responses: list[SearchAnswer] = Field(
        default_factory=list, description="Structured QA pairs used during research"
    )
    insights: list[InsightRecord] = Field(
        default_factory=list, description="Key insights discovered"
    )
    gaps: list[GapRecord] = Field(
        default_factory=list, description="Identified information gaps"
    )

    # Private dict indexes for O(1) lookups
    _insights_by_id: dict[str, InsightRecord] = PrivateAttr(default_factory=dict)
    _gaps_by_id: dict[str, GapRecord] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: object) -> None:
        """Build indexes after initialization."""
        self._insights_by_id = {ins.id: ins for ins in self.insights}
        self._gaps_by_id = {gap.id: gap for gap in self.gaps}

    def add_qa_response(self, qa: SearchAnswer) -> None:
        """Add a structured QA response (minimal context already included)."""
        self.qa_responses.append(qa)

    def upsert_insights(self, records: Iterable[InsightRecord]) -> list[InsightRecord]:
        """Merge one or more insights into the shared context with deduplication."""
        merged: list[InsightRecord] = []

        for record in records:
            candidate = InsightRecord.model_validate(record)
            existing = self._insights_by_id.get(candidate.id)

            if existing:
                # Update existing insight
                existing.summary = candidate.summary
                existing.status = candidate.status
                if candidate.notes:
                    existing.notes = candidate.notes
                existing.supporting_sources = _merge_unique(
                    existing.supporting_sources, candidate.supporting_sources
                )
                existing.originating_questions = _merge_unique(
                    existing.originating_questions, candidate.originating_questions
                )
                merged.append(existing)
            else:
                # Add new insight
                new_insight = candidate.model_copy(deep=True)
                self.insights.append(new_insight)
                self._insights_by_id[new_insight.id] = new_insight
                merged.append(new_insight)

        return merged

    def upsert_gaps(self, records: Iterable[GapRecord]) -> list[GapRecord]:
        """Merge one or more gap records into the shared context with deduplication."""
        merged: list[GapRecord] = []

        for record in records:
            candidate = GapRecord.model_validate(record)
            existing = self._gaps_by_id.get(candidate.id)

            if existing:
                # Update existing gap
                existing.description = candidate.description
                existing.severity = candidate.severity
                existing.blocking = candidate.blocking
                existing.resolved = candidate.resolved
                if candidate.notes:
                    existing.notes = candidate.notes
                existing.supporting_sources = _merge_unique(
                    existing.supporting_sources, candidate.supporting_sources
                )
                existing.resolved_by = _merge_unique(
                    existing.resolved_by, candidate.resolved_by
                )
                merged.append(existing)
            else:
                # Add new gap
                new_gap = candidate.model_copy(deep=True)
                self.gaps.append(new_gap)
                self._gaps_by_id[new_gap.id] = new_gap
                merged.append(new_gap)

        return merged

    def mark_gap_resolved(
        self, identifier: str, resolved_by: Iterable[str] | None = None
    ) -> GapRecord | None:
        """Mark a gap as resolved by identifier."""
        gap = self._gaps_by_id.get(identifier)
        if gap is None:
            return None

        gap.resolved = True
        gap.blocking = False
        if resolved_by:
            gap.resolved_by = _merge_unique(gap.resolved_by, list(resolved_by))
        return gap

    def integrate_analysis(self, analysis: InsightAnalysis) -> None:
        """Apply an analysis result to the shared context."""
        merged_insights: list[InsightRecord] = []
        if analysis.highlights:
            merged_insights = self.upsert_insights(analysis.highlights)
            analysis.highlights = merged_insights
        if analysis.gap_assessments:
            merged_gaps = self.upsert_gaps(analysis.gap_assessments)
            analysis.gap_assessments = merged_gaps
        if analysis.resolved_gaps:
            resolved_by_list = (
                [ins.id for ins in merged_insights] if merged_insights else None
            )
            for resolved in analysis.resolved_gaps:
                self.mark_gap_resolved(resolved, resolved_by=resolved_by_list)
        for question in analysis.new_questions:
            if question not in self.sub_questions:
                self.sub_questions.append(question)


class ResearchDependencies(BaseModel):
    """Dependencies for research agents with multi-agent context."""

    model_config = {"arbitrary_types_allowed": True}

    client: HaikuRAG = Field(description="RAG client for document operations")
    context: ResearchContext = Field(description="Shared research context")


def _merge_unique(existing: list[str], incoming: Iterable[str]) -> list[str]:
    """Merge two iterables preserving order while removing duplicates."""
    return [k for k in dict.fromkeys([*existing, *incoming]) if k]

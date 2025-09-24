from collections.abc import Iterable

from pydantic import BaseModel, Field
from rich.console import Console

from haiku.rag.client import HaikuRAG
from haiku.rag.research.models import (
    GapRecord,
    InsightAnalysis,
    InsightRecord,
    SearchAnswer,
)
from haiku.rag.research.stream import ResearchStream


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

    def add_qa_response(self, qa: SearchAnswer) -> None:
        """Add a structured QA response (minimal context already included)."""
        self.qa_responses.append(qa)

    def upsert_insights(self, records: Iterable[InsightRecord]) -> list[InsightRecord]:
        """Merge one or more insights into the shared context with deduplication."""

        merged: list[InsightRecord] = []
        for record in records:
            candidate = InsightRecord.model_validate(record)
            existing = next(
                (ins for ins in self.insights if ins.id == candidate.id), None
            )
            if not existing:
                existing = next(
                    (ins for ins in self.insights if ins.summary == candidate.summary),
                    None,
                )

            if existing:
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
                candidate = candidate.model_copy(deep=True)
                if candidate.id is None:  # pragma: no cover - defensive
                    raise ValueError(
                        "InsightRecord.id must be populated after validation"
                    )
                candidate_id: str = candidate.id
                candidate.id = self._allocate_insight_id(candidate_id)
                self.insights.append(candidate)
                merged.append(candidate)

        return merged

    def upsert_gaps(self, records: Iterable[GapRecord]) -> list[GapRecord]:
        """Merge one or more gap records into the shared context with deduplication."""

        merged: list[GapRecord] = []
        for record in records:
            candidate = GapRecord.model_validate(record)
            existing = next((gap for gap in self.gaps if gap.id == candidate.id), None)
            if not existing:
                existing = next(
                    (
                        gap
                        for gap in self.gaps
                        if gap.description == candidate.description
                    ),
                    None,
                )

            if existing:
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
                candidate = candidate.model_copy(deep=True)
                if candidate.id is None:  # pragma: no cover - defensive
                    raise ValueError("GapRecord.id must be populated after validation")
                candidate_id: str = candidate.id
                candidate.id = self._allocate_gap_id(candidate_id)
                self.gaps.append(candidate)
                merged.append(candidate)

        return merged

    def mark_gap_resolved(
        self, identifier: str, resolved_by: Iterable[str] | None = None
    ) -> GapRecord | None:
        """Mark a gap as resolved by identifier (id or description)."""

        gap = self._find_gap(identifier)
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
                [ins.id for ins in merged_insights if ins.id is not None]
                if merged_insights
                else None
            )
            for resolved in analysis.resolved_gaps:
                self.mark_gap_resolved(resolved, resolved_by=resolved_by_list)
        for question in analysis.new_questions:
            if question not in self.sub_questions:
                self.sub_questions.append(question)

    def _allocate_insight_id(self, candidate_id: str) -> str:
        taken: set[str] = set()
        for ins in self.insights:
            if ins.id is not None:
                taken.add(ins.id)
        return _allocate_sequential_id(candidate_id, taken)

    def _allocate_gap_id(self, candidate_id: str) -> str:
        taken: set[str] = set()
        for gap in self.gaps:
            if gap.id is not None:
                taken.add(gap.id)
        return _allocate_sequential_id(candidate_id, taken)

    def _find_gap(self, identifier: str) -> GapRecord | None:
        normalized = identifier.lower().strip()
        for gap in self.gaps:
            if gap.id is not None and gap.id == normalized:
                return gap
            if gap.description.lower().strip() == normalized:
                return gap
        return None


class ResearchDependencies(BaseModel):
    """Dependencies for research agents with multi-agent context."""

    model_config = {"arbitrary_types_allowed": True}

    client: HaikuRAG = Field(description="RAG client for document operations")
    context: ResearchContext = Field(description="Shared research context")
    console: Console | None = None
    stream: ResearchStream | None = Field(
        default=None, description="Optional research event stream"
    )


def _merge_unique(existing: list[str], incoming: Iterable[str]) -> list[str]:
    """Merge two iterables preserving order while removing duplicates."""

    merged = list(existing)
    seen = {item for item in existing if item}
    for item in incoming:
        if item and item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def _allocate_sequential_id(candidate: str, taken: set[str]) -> str:
    slug = candidate
    if slug not in taken:
        return slug
    base = slug
    counter = 2
    while True:
        slug = f"{base}-{counter}"
        if slug not in taken:
            return slug
        counter += 1

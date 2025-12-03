"""Common models used across different graph implementations."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from haiku.rag.store.models import SearchResult


class ResearchPlan(BaseModel):
    """A structured research plan with sub-questions to explore."""

    sub_questions: list[str] = Field(
        ...,
        description="Specific questions to research, phrased as complete questions",
    )

    @field_validator("sub_questions")
    @classmethod
    def validate_sub_questions(cls, v: list[str]) -> list[str]:
        if len(v) < 1:
            raise ValueError("Must have at least 1 sub-question")
        if len(v) > 12:
            raise ValueError("Cannot have more than 12 sub-questions")
        return v


class Citation(BaseModel):
    """Resolved citation with full metadata for display/visual grounding."""

    document_id: str
    chunk_id: str
    document_uri: str
    document_title: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    headings: list[str] | None = None
    content: str


class RawSearchAnswer(BaseModel):
    """Answer to a search query with chunk references."""

    query: str = Field(..., description="The question that was answered")
    answer: str = Field(..., description="The answer to the question")
    cited_chunks: list[str] = Field(
        default_factory=list,
        description="IDs of chunks used to form the answer",
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score for this answer (0-1)",
        ge=0.0,
        le=1.0,
    )


class SearchAnswer(RawSearchAnswer):
    """Answer to a search query with resolved citations."""

    citations: list[Citation] = Field(
        default_factory=list,
        description="Resolved citations with full metadata",
    )

    @classmethod
    def from_raw(
        cls,
        raw: RawSearchAnswer,
        search_results: "list[SearchResult]",
    ) -> "SearchAnswer":
        """Create SearchAnswer from RawSearchAnswer with resolved citations."""
        citations = resolve_citations(raw.cited_chunks, search_results)
        return cls(
            query=raw.query,
            answer=raw.answer,
            cited_chunks=raw.cited_chunks,
            confidence=raw.confidence,
            citations=citations,
        )


def resolve_citations(
    cited_chunk_ids: list[str],
    search_results: "list[SearchResult]",
) -> list[Citation]:
    """Resolve chunk IDs to full Citation objects with metadata."""
    # Build lookup by chunk_id
    by_id = {r.chunk_id: r for r in search_results if r.chunk_id}

    citations = []
    for chunk_id in cited_chunk_ids:
        r = by_id.get(chunk_id)
        if not r:
            continue
        citations.append(
            Citation(
                document_id=r.document_id or "",
                chunk_id=chunk_id,
                document_uri=r.document_uri or "",
                document_title=r.document_title,
                page_numbers=r.page_numbers,
                headings=r.headings,
                content=r.content,
            )
        )
    return citations

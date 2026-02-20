from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from haiku.rag.store.models import SearchResult


class IterativePlanResult(BaseModel):
    """Output from iterative planning step."""

    is_complete: bool = Field(
        description="Whether research is complete and can be synthesized"
    )
    next_question: str | None = Field(
        default=None, description="Next question to investigate, if not complete"
    )
    reasoning: str = Field(description="Brief explanation of the decision")


class Citation(BaseModel):
    """Resolved citation with full metadata for display/visual grounding.

    Used by research graph and chat applications. The optional index field
    supports UI display ordering in chat contexts.
    """

    index: int | None = None
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

    @property
    def primary_source(self) -> str | None:
        """Get primary source title from citations."""
        if not self.citations:
            return None
        first = self.citations[0]
        return first.document_title or first.document_uri

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

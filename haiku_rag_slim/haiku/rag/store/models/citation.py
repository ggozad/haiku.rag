from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from haiku.rag.store.models.document_item import PICTURE_REF_PREFIX

if TYPE_CHECKING:
    from haiku.rag.store.models import SearchResult


class Citation(BaseModel):
    """Resolved citation with full metadata for display/visual grounding.

    Used by the rag and analysis skills and rendered by the CLI / chat
    application. The optional index field supports UI display ordering.

    ``picture_refs`` lists the ``self_ref`` values of picture items in the
    cited chunk. Empty for text-only citations. UIs can fetch the picture
    bytes via ``DocumentItemRepository.get_picture_bytes(document_id, ref)``
    and render them alongside the text content.
    """

    index: int | None = None
    document_id: str
    chunk_id: str
    document_uri: str
    document_title: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    headings: list[str] | None = None
    content: str
    picture_refs: list[str] = Field(default_factory=list)


def resolve_citations(
    cited_chunk_ids: list[str],
    search_results: "list[SearchResult]",
) -> list[Citation]:
    """Resolve chunk IDs to full Citation objects with metadata."""
    by_id = {r.chunk_id: r for r in search_results if r.chunk_id}

    citations = []
    for raw_id in cited_chunk_ids:
        chunk_id = raw_id.strip("[]")
        r = by_id.get(chunk_id)
        if not r:
            continue
        picture_refs = [
            ref for ref in r.doc_item_refs if ref.startswith(PICTURE_REF_PREFIX)
        ]
        citations.append(
            Citation(
                document_id=r.document_id or "",
                chunk_id=chunk_id,
                document_uri=r.document_uri or "",
                document_title=r.document_title,
                page_numbers=r.page_numbers,
                headings=r.headings,
                content=r.content,
                picture_refs=picture_refs,
            )
        )
    return citations

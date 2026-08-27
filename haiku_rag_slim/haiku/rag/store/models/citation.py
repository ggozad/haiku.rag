from collections.abc import Iterable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from haiku.rag.store.exceptions import AmbiguousCitationError
from haiku.rag.store.models.document_item import PICTURE_REF_PREFIX

if TYPE_CHECKING:
    from haiku.rag.store.models import SearchResult


class Citation(BaseModel):
    """Resolved citation with full metadata for display/visual grounding.

    Used by the RAG and analysis capabilities and rendered by the CLI / chat
    application. The optional index field supports UI display ordering.

    ``picture_refs`` lists the ``self_ref`` values of picture items in the
    cited chunk. Empty for text-only citations. UIs can fetch the picture
    bytes via ``HaikuRAG.get_picture_bytes(document_id, ref, source)`` and
    render them alongside the text content.

    ``chunk_ids`` lists the ids of all chunks whose expansion ranges merged
    into the cited result (always includes ``chunk_id``).

    ``source`` names the configured database the cited chunk came from: the name
    from ``lancedb.databases``, never a path or URI. It is None only where no
    database is named, as with the single ``lancedb.uri``.

    ``doc_item_refs`` are the ``self_ref`` values of every item in the cited
    content — the exact items the model saw. Visual grounding resolves bounding
    boxes from them so the rendered pages match the citation precisely.
    ``picture_refs`` is the picture-labeled subset.

    ``document_meta`` carries the cited document's metadata for UIs.

    ``chunk_meta`` is the cited chunk's raw, unparsed ``Chunk.metadata``
    dict — lossless and independent of the typed fields above, so a
    third-party chunker's own fields survive here even as this schema
    evolves.
    """

    index: int | None = None
    document_id: str
    source: str | None = None
    chunk_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_meta: dict = Field(default_factory=dict)
    document_uri: str
    document_title: str | None = None
    document_meta: dict = Field(default_factory=dict)
    page_numbers: list[int] = Field(default_factory=list)
    headings: list[str] | None = None
    content: str
    doc_item_refs: list[str] = Field(default_factory=list)
    picture_refs: list[str] = Field(default_factory=list)


def ambiguous_citation(
    chunk_id: str, sources: Iterable[str | None]
) -> AmbiguousCitationError:
    """The refusal for an id that names a chunk in more than one database."""
    named = ", ".join(sorted(s or "unnamed" for s in sources))
    return AmbiguousCitationError(
        f"chunk id {chunk_id} names a chunk in more than one database "
        f"({named}); a citation records the id alone and cannot say which"
    )


def resolve_citations(
    cited_chunk_ids: list[str],
    search_results: "list[SearchResult]",
) -> list[Citation]:
    """Resolve chunk IDs to full Citation objects with metadata.

    A chunk returned by more than one search resolves to its last occurrence.
    Raises ``AmbiguousCitationError`` instead when a cited id names a chunk in
    more than one of the databases searched, as after copying a database: a
    citation records the id alone, so resolving one would attribute the answer
    to a database it may not have come from.
    """
    by_id: dict[str, SearchResult] = {}
    ambiguous: dict[str, set[str | None]] = {}
    for r in search_results:
        # A result built by hand carries no id and nothing can cite it.
        if cid := r.chunk_id:
            if (held := by_id.get(cid)) is not None and held.source != r.source:
                ambiguous.setdefault(cid, {held.source}).add(r.source)
            # A chunk found by several searches is expanded once per search, so
            # the copies differ in content, window and figures. The later entry
            # wins.
            by_id[cid] = r

    citations = []
    for raw_id in cited_chunk_ids:
        chunk_id = raw_id.strip("[]")
        if chunk_id in ambiguous:
            raise ambiguous_citation(chunk_id, ambiguous[chunk_id])
        r = by_id.get(chunk_id)
        if not r:
            continue
        picture_refs = [
            ref for ref in r.doc_item_refs if ref.startswith(PICTURE_REF_PREFIX)
        ]
        citations.append(
            Citation(
                document_id=r.document_id or "",
                source=r.source,
                chunk_id=chunk_id,
                chunk_ids=r.chunk_ids or [chunk_id],
                chunk_meta=r.chunk_meta,
                document_uri=r.document_uri or "",
                document_title=r.document_title,
                document_meta=r.document_meta,
                page_numbers=r.page_numbers,
                headings=r.headings,
                content=r.content,
                doc_item_refs=list(r.doc_item_refs),
                picture_refs=picture_refs,
            )
        )
    return citations

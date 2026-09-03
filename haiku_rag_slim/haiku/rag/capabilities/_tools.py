from collections.abc import Iterable
from collections.abc import Set as AbstractSet

from pydantic import BaseModel

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import SearchResult, qualified_id
from haiku.rag.tools.search import picture_keys


class CodeExecutionEntry(BaseModel):
    code: str
    stdout: str
    stderr: str = ""
    success: bool = True


EvidenceKey = tuple[tuple[str | None, str | None], tuple[str, frozenset]]
"""What tells one rendered result from another: qualified id, then signature.

The qualified id comes first because the rendered string alone would conflate
identical renderings of the same chunk id held by two databases.
"""


def evidence_signature(result: SearchResult, include_collection: bool) -> tuple:
    """The rendered evidence a result shows the model, as an equivalence key.

    Rank and total are held at neutral values: they vary with a result's
    position, and position (like score) must not tell two renderings apart.
    """
    return (
        result.format_for_agent(rank=0, total=0, include_collection=include_collection),
        picture_keys(result),
    )


def evidence_key(result: SearchResult, include_collection: bool) -> EvidenceKey:
    return (
        qualified_id(result.source, result.chunk_id),
        evidence_signature(result, include_collection),
    )


async def search_corpus(
    rag: HaikuRAG,
    query: str,
    limit: int | None = None,
    document_filter: str | None = None,
    sources: list[str] | None = None,
    shown: AbstractSet[EvidenceKey] = frozenset(),
) -> tuple[str, list[SearchResult], set[EvidenceKey], bool]:
    """Search and context-expand results, eliding evidence already shown.

    Returns the formatted results, the full result list, the evidence keys the
    formatting rendered in full, and whether results name their collection. A
    result whose key is in ``shown`` keeps its slot but collapses to one line;
    the result list is never filtered.
    """
    results = await rag.search(
        query, limit=limit, filter=document_filter, sources=sources
    )
    results = await rag.expand_context(results)
    # Named from the selection, not the hits: a search that could have drawn on
    # two collections names them even when everything came back from one.
    selected = rag.source_names if sources is None else sources
    include_collection = len(set(selected)) > 1
    rendered: set[EvidenceKey] = set()
    parts: list[str] = []
    total = len(results)
    for index, result in enumerate(results):
        key = evidence_key(result, include_collection)
        if key in shown or key in rendered:
            parts.append(
                f"Also matched, shown above: [{result.chunk_id}] "
                f"[rank {index + 1} of {total}]"
            )
        else:
            parts.append(
                result.format_for_agent(
                    rank=index + 1, total=total, include_collection=include_collection
                )
            )
            rendered.add(key)
    formatted = "\n\n---\n\n".join(parts)
    return formatted or "No results found.", list(results), rendered, include_collection


def merge_results(
    existing: list[SearchResult], incoming: Iterable[SearchResult]
) -> None:
    """Add the results not already held.

    Identity is the database and the chunk id: results built by hand carry
    neither and cannot be told apart, so they collapse to the first.
    """
    seen = {qualified_id(result.source, result.chunk_id) for result in existing}
    for result in incoming:
        key = qualified_id(result.source, result.chunk_id)
        if key not in seen:
            existing.append(result)
            seen.add(key)


__all__ = [
    "CodeExecutionEntry",
    "EvidenceKey",
    "evidence_key",
    "evidence_signature",
    "merge_results",
    "search_corpus",
]

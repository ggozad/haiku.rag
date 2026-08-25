from collections.abc import Iterable

from pydantic import BaseModel

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import SearchResult, qualified_id


class CodeExecutionEntry(BaseModel):
    code: str
    stdout: str
    stderr: str = ""
    success: bool = True


async def search_corpus(
    rag: HaikuRAG,
    query: str,
    limit: int | None = None,
    document_filter: str | None = None,
    sources: list[str] | None = None,
) -> tuple[str, list[SearchResult]]:
    """Search and context-expand results for a capability tool."""
    results = await rag.search(
        query, limit=limit, filter=document_filter, sources=sources
    )
    results = await rag.expand_context(results)
    formatted = "\n\n---\n\n".join(
        result.format_for_agent(rank=index + 1, total=len(results))
        for index, result in enumerate(results)
    )
    return formatted or "No results found.", list(results)


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
    "merge_results",
    "search_corpus",
]

from pydantic import BaseModel

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import SearchResult


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
) -> tuple[str, list[SearchResult]]:
    """Search and context-expand results for a capability tool."""
    results = await rag.search(query, limit=limit, filter=document_filter)
    results = await rag.expand_context(results)
    formatted = "\n\n---\n\n".join(
        result.format_for_agent(rank=index + 1, total=len(results))
        for index, result in enumerate(results)
    )
    return formatted, list(results)


__all__ = [
    "CodeExecutionEntry",
    "search_corpus",
]

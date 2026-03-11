from collections.abc import Callable

from pydantic_ai import FunctionToolset, RunContext

from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult
from haiku.rag.tools.context import RAGDeps


def create_search_toolset(
    config: AppConfig,
    expand_context: bool = True,
    base_filter: str | None = None,
    tool_name: str = "search",
    on_results: Callable[[list[SearchResult]], None] | None = None,
    max_searches: int | None = None,
) -> FunctionToolset[RAGDeps]:
    """Create a toolset with search capabilities.

    Args:
        config: Application configuration.
        expand_context: Whether to expand search results with surrounding context.
            Defaults to True.
        base_filter: Optional base SQL WHERE clause applied to all searches.
            Combined with any filter passed to the search tool.
        tool_name: Name for the search tool. Defaults to "search".
        on_results: Optional callback invoked with search results after each search.
            Useful for accumulating results externally (e.g., for citation resolution).
        max_searches: Maximum number of searches allowed. When exceeded, returns
            a message directing the agent to answer with existing results.

    Returns:
        FunctionToolset with a search tool.
    """
    # Per-run search counter. Resets when run_id changes so the toolset
    # can safely be reused across multiple agent.run() calls.
    search_count = 0
    current_run_id: str | None = None

    async def search(
        ctx: RunContext[RAGDeps],
        query: str,
        limit: int | None = None,
    ) -> str:
        """Search the knowledge base for relevant documents.

        Args:
            query: The search query (what to search for).
            limit: Number of results to return (default: from config).

        Returns:
            Formatted search results with content and metadata.
        """
        nonlocal search_count, current_run_id
        if ctx.run_id != current_run_id:
            current_run_id = ctx.run_id
            search_count = 0
        search_count += 1
        if max_searches is not None and search_count > max_searches:
            return (
                "Search limit reached. "
                "Answer the question using the results you already have."
            )

        client = ctx.deps.client

        effective_filter = base_filter
        effective_limit = limit or config.search.limit
        results = await client.search(
            query, limit=effective_limit, filter=effective_filter
        )

        if expand_context:
            results = await client.expand_context(results)

        results_list = list(results)

        if on_results:
            on_results(results_list)

        if not results_list:
            return "No results found."

        total = len(results_list)
        formatted = [
            r.format_for_agent(rank=i + 1, total=total)
            for i, r in enumerate(results_list)
        ]
        return "\n\n".join(formatted)

    toolset: FunctionToolset[RAGDeps] = FunctionToolset()
    toolset.add_function(search, name=tool_name, retries=3)
    return toolset

from pydantic import BaseModel
from pydantic_ai import FunctionToolset

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.filters import combine_filters

SEARCH_NAMESPACE = "haiku.rag.search"


class SearchState(BaseModel):
    """State for search toolset.

    Accumulates search results across tool invocations.
    """

    results: list[SearchResult] = []


def create_search_toolset(
    client: HaikuRAG,
    config: AppConfig,
    context: ToolContext | None = None,
    expand_context: bool = True,
    base_filter: str | None = None,
    tool_name: str = "search",
) -> FunctionToolset:
    """Create a toolset with search capabilities.

    Args:
        client: HaikuRAG client for search operations.
        config: Application configuration.
        context: Optional ToolContext for state accumulation.
            If provided, search results are accumulated in SearchState.
        expand_context: Whether to expand search results with surrounding context.
            Defaults to True.
        base_filter: Optional base SQL WHERE clause applied to all searches.
            Combined with any filter passed to the search tool.
        tool_name: Name for the search tool. Defaults to "search".

    Returns:
        FunctionToolset with a search tool.
    """
    # Get or create search state if context provided
    state: SearchState | None = None
    if context is not None:
        state = context.get_or_create(SEARCH_NAMESPACE, SearchState)

    async def search(
        query: str,
        limit: int | None = None,
        filter: str | None = None,
    ) -> str:
        """Search the knowledge base for relevant documents.

        Args:
            query: The search query (what to search for).
            limit: Number of results to return (default: from config).
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            Formatted search results with content and metadata.
        """
        effective_limit = limit or config.search.limit
        effective_filter = combine_filters(base_filter, filter)
        results = await client.search(
            query, limit=effective_limit, filter=effective_filter
        )

        if expand_context:
            results = await client.expand_context(results)

        # Accumulate results in state if context provided
        if state is not None:
            state.results.extend(results)

        if not results:
            return "No results found."

        # Format results for agent context
        total = len(results)
        formatted = [
            r.format_for_agent(rank=i + 1, total=total) for i, r in enumerate(results)
        ]
        return "\n\n".join(formatted)

    toolset = FunctionToolset()
    toolset.add_function(search, name=tool_name)
    return toolset

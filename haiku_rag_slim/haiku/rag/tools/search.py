from pydantic import BaseModel
from pydantic_ai import FunctionToolset, ToolReturn

from haiku.rag.agents.research.models import Citation
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult
from haiku.rag.tools.context import ToolContext
from haiku.rag.tools.filters import combine_filters, get_session_filter
from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState, compute_state_delta

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
            If SessionState is registered, it will be used for dynamic
            document filtering and citation indexing.
        expand_context: Whether to expand search results with surrounding context.
            Defaults to True.
        base_filter: Optional base SQL WHERE clause applied to all searches.
            Combined with any filter passed to the search tool.
        tool_name: Name for the search tool. Defaults to "search".

    Returns:
        FunctionToolset with a search tool.
    """
    # Get or create search state if context provided
    search_state: SearchState | None = None
    if context is not None:
        search_state = context.get_or_create(SEARCH_NAMESPACE, SearchState)

    async def search(
        query: str,
        limit: int | None = None,
        filter: str | None = None,
    ) -> ToolReturn | str:
        """Search the knowledge base for relevant documents.

        Args:
            query: The search query (what to search for).
            limit: Number of results to return (default: from config).
            filter: Optional SQL WHERE clause to filter documents.

        Returns:
            Formatted search results with content and metadata.
        """
        # Get session state for dynamic filters and citation indexing
        session_state: SessionState | None = None
        old_session_state: SessionState | None = None
        if context is not None:
            session_state = context.get_typed(SESSION_NAMESPACE, SessionState)
            if session_state is not None:
                old_session_state = session_state.model_copy(deep=True)

        # Combine all filters: base_filter AND session_filter AND tool filter
        effective_filter = combine_filters(
            get_session_filter(context, base_filter), filter
        )

        effective_limit = limit or config.search.limit
        results = await client.search(
            query, limit=effective_limit, filter=effective_filter
        )

        if expand_context:
            results = await client.expand_context(results)

        # Accumulate results in search state if context provided
        if search_state is not None:
            search_state.results.extend(results)

        if not results:
            return "No results found."

        # Build citations if session state is available
        if session_state is not None:
            citations = []
            for r in results:
                chunk_id = r.chunk_id or ""
                if chunk_id:
                    index = session_state.get_or_assign_index(chunk_id)
                else:
                    index = len(session_state.citation_registry) + 1
                citations.append(
                    Citation(
                        index=index,
                        document_id=r.document_id or "",
                        chunk_id=chunk_id,
                        document_uri=r.document_uri or "",
                        document_title=r.document_title,
                        page_numbers=r.page_numbers or [],
                        headings=r.headings,
                        content=r.content,
                    )
                )
            session_state.citations = citations

            # Format results with citation indices
            result_lines = []
            for c in citations:
                title = c.document_title or c.document_uri or "Unknown"
                snippet = c.content[:300].replace("\n", " ").strip()
                if len(c.content) > 300:
                    snippet += "..."

                line = f"[{c.index}] **{title}**"
                if c.page_numbers:
                    line += f" (pages {', '.join(map(str, c.page_numbers))})"
                line += f"\n    {snippet}"
                result_lines.append(line)

            formatted = f"Found {len(results)} results:\n\n" + "\n\n".join(result_lines)

            # Compute state delta if session state changed
            if old_session_state is not None:
                state_event = compute_state_delta(old_session_state, session_state)
                if state_event is not None:
                    return ToolReturn(
                        return_value=formatted,
                        metadata=[state_event],
                    )

            return formatted

        # Format results without citation indexing (standalone use)
        total = len(results)
        formatted = [
            r.format_for_agent(rank=i + 1, total=total) for i, r in enumerate(results)
        ]
        return "\n\n".join(formatted)

    toolset = FunctionToolset()
    toolset.add_function(search, name=tool_name)
    return toolset

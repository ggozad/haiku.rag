import logging
import time
from collections.abc import Callable

from pydantic_ai import FunctionToolset, RunContext

from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult
from haiku.rag.tools.context import RAGDeps

logger = logging.getLogger(__name__)


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
    # Per-run search counter keyed by run_id. Safe for concurrent runs
    # and reuse across sequential agent.run() calls.
    search_counts: dict[str, int] = {}
    _last_tool_return: list[float] = []

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
        tool_start = time.perf_counter()

        if _last_tool_return:
            llm_think_time = tool_start - _last_tool_return[0]
            logger.info(
                "tool.llm_thinking took %.3fs", llm_think_time
            )

        rid = ctx.run_id or ""
        search_counts[rid] = search_counts.get(rid, 0) + 1
        if max_searches is not None and search_counts[rid] > max_searches:
            _last_tool_return[:] = [time.perf_counter()]
            return (
                "Search limit reached. "
                "Answer the question using the results you already have."
            )

        client = ctx.deps.client

        effective_filter = base_filter
        effective_limit = limit or config.search.limit

        t0 = time.perf_counter()
        results = await client.search(
            query, limit=effective_limit, filter=effective_filter
        )
        logger.info(
            "tool.search query=%r took %.3fs",
            query[:80],
            time.perf_counter() - t0,
        )

        if expand_context:
            t0 = time.perf_counter()
            results = await client.expand_context(results)
            logger.info(
                "tool.expand_context results=%d took %.3fs",
                len(results),
                time.perf_counter() - t0,
            )

        results_list = list(results)

        if on_results:
            on_results(results_list)

        if not results_list:
            _last_tool_return[:] = [time.perf_counter()]
            return "No results found."

        t0 = time.perf_counter()
        total = len(results_list)
        formatted = [
            r.format_for_agent(rank=i + 1, total=total)
            for i, r in enumerate(results_list)
        ]
        output = "\n\n".join(formatted)
        logger.info(
            "tool.format results=%d chars=%d took %.3fs",
            total,
            len(output),
            time.perf_counter() - t0,
        )

        logger.info(
            "tool.search_total took %.3fs",
            time.perf_counter() - tool_start,
        )
        _last_tool_return[:] = [time.perf_counter()]
        return output

    toolset: FunctionToolset[RAGDeps] = FunctionToolset()
    toolset.add_function(search, name=tool_name, retries=3)
    return toolset

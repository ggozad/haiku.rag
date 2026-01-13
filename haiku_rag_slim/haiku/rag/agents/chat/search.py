from pydantic_ai import Agent, RunContext

from haiku.rag.agents.chat.prompts import SEARCH_SYSTEM_PROMPT
from haiku.rag.agents.chat.state import SearchDeps
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult
from haiku.rag.utils import get_model


class SearchAgent:
    """Agent that generates multiple queries and consolidates results."""

    def __init__(self, client: HaikuRAG, config: AppConfig):
        self._client = client
        self._config = config

        model = get_model(config.qa.model, config)
        self._agent: Agent[SearchDeps, str] = Agent(
            model,
            deps_type=SearchDeps,
            output_type=str,
            instructions=SEARCH_SYSTEM_PROMPT,
        )

        @self._agent.tool
        async def run_search(
            ctx: RunContext[SearchDeps],
            query: str,
            limit: int | None = None,
        ) -> str:
            """Run a single search query against the knowledge base.

            Args:
                query: The search query
                limit: Number of results to fetch (default: 5)
            """
            effective_limit = limit or 5
            results = await ctx.deps.client.search(
                query, limit=effective_limit, filter=ctx.deps.filter
            )
            results = await ctx.deps.client.expand_context(results)
            ctx.deps.search_results.extend(results)

            if not results:
                return f"No results for: {query}"
            return f"Found {len(results)} results for: {query}"

    async def search(
        self,
        query: str,
        context: str | None = None,
        filter: str | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Execute search with query expansion and deduplication.

        Args:
            query: The user's search request
            context: Optional conversation context
            filter: Optional SQL WHERE clause to filter documents
            limit: Maximum number of results to return (default: config limit)

        Returns:
            Deduplicated list of SearchResult sorted by score
        """
        prompt = query
        if context:
            prompt = f"Context: {context}\n\nSearch request: {query}"

        deps = SearchDeps(client=self._client, config=self._config, filter=filter)
        await self._agent.run(prompt, deps=deps)

        # Deduplicate by chunk_id, keeping highest score
        seen: dict[str, SearchResult] = {}
        for result in deps.search_results:
            chunk_id = result.chunk_id or ""
            if chunk_id not in seen or result.score > seen[chunk_id].score:
                seen[chunk_id] = result

        # Sort by score descending and apply limit
        effective_limit = limit or self._config.search.limit
        return sorted(seen.values(), key=lambda r: r.score, reverse=True)[
            :effective_limit
        ]

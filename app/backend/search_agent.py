from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult
from haiku.rag.utils import get_model


@dataclass
class SearchDeps:
    """Dependencies for search agent."""

    client: HaikuRAG
    config: AppConfig
    filter: str | None = None
    search_results: list[SearchResult] = field(default_factory=list)


SEARCH_SYSTEM_PROMPT = """You are a search query optimizer for a document knowledge base.

Given a user's search request:
1. ALWAYS run the original query first as-is
2. Then generate 1-2 alternative queries using different keywords or phrasings
3. Keep queries SHORT (2-5 words) - use keywords, not full sentences
4. After all searches, respond with "Search complete"

Example: User asks "latrines" → queries: "latrines", "latrine sanitation", "field toilet"
Example: User asks "waste disposal" → queries: "waste disposal", "garbage management", "refuse handling"

Do NOT generate long verbose queries like "environmental impact of waste disposal methods" - keep it simple."""


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
        ) -> str:
            """Run a single search query against the knowledge base.

            Args:
                query: The search query
            """
            limit = ctx.deps.config.search.limit
            results = await ctx.deps.client.search(
                query, limit=limit, filter=ctx.deps.filter
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
    ) -> list[SearchResult]:
        """Execute search with query expansion and deduplication.

        Args:
            query: The user's search request
            context: Optional conversation context
            filter: Optional SQL WHERE clause to filter documents

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

        # Sort by score descending and limit to config
        limit = self._config.search.limit
        return sorted(seen.values(), key=lambda r: r.score, reverse=True)[:limit]

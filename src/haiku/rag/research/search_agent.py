"""Search specialist agent for advanced document retrieval."""

from pydantic_ai import RunContext

from haiku.rag.research.base import BaseResearchAgent, SearchResult
from haiku.rag.research.dependencies import ResearchDependencies


class SearchSpecialistAgent(BaseResearchAgent):
    """Agent specialized in advanced document search and retrieval."""

    def __init__(self, provider: str, model: str):
        super().__init__(provider, model, output_type=list[SearchResult])

    def get_system_prompt(self) -> str:
        return """You are a search specialist agent focused on document retrieval.
        Your role is to:
        1. Generate multiple search queries from different perspectives
        2. Identify key terms and synonyms for comprehensive search
        3. Execute searches and rank results by relevance
        4. Return the most relevant documents for the research question

        Use the search tools to explore the knowledge base thoroughly."""

    def register_tools(self) -> None:
        """Register search-specific tools."""

        @self.agent.tool
        async def search(
            ctx: RunContext[ResearchDependencies],
            queries: str | list[str],
            limit: int = 5,
        ) -> list[SearchResult]:
            """Execute search with single or multiple query variants."""
            # Normalize to list
            query_list = [queries] if isinstance(queries, str) else queries

            all_results = []
            seen_chunk_ids = set()

            for query in query_list:
                # Use the default hybrid search
                search_results = await ctx.deps.client.search(query, limit=limit)

                # Expand context for better relevance
                expanded = await ctx.deps.client.expand_context(search_results)

                for chunk, score in expanded:
                    # Avoid duplicates based on chunk ID
                    if chunk.id and chunk.id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk.id)
                        all_results.append(
                            SearchResult(
                                content=chunk.content,
                                score=score,
                                document_uri=chunk.document_uri or "",
                                metadata={"query": query, "chunk_id": chunk.id},
                            )
                        )

            # Sort by score and limit results
            all_results.sort(key=lambda x: x.score, reverse=True)
            final_results = all_results[: limit * len(query_list)]

            # Store in context
            query_summary = (
                query_list[0]
                if len(query_list) == 1
                else f"Multi-query: {len(query_list)} variants"
            )
            ctx.deps.context.add_search_result(query_summary, final_results)

            return final_results

"""Search specialist agent for document retrieval."""

from pydantic_ai import RunContext

from haiku.rag.research.base import BaseResearchAgent
from haiku.rag.research.dependencies import ResearchDependencies
from haiku.rag.store.models.chunk import Chunk


class SearchSpecialistAgent(BaseResearchAgent):
    """Agent specialized in document search and retrieval."""

    def __init__(self, provider: str, model: str):
        # No specific output type needed - the tool handles everything
        super().__init__(provider, model)

    def get_system_prompt(self) -> str:
        return """You are a search specialist agent focused on document retrieval from a knowledge base that uses hybrid (semantic and full-text search) search.
        Your role is to:
        1. Understand the search query and context
        2. Execute targeted searches to find relevant documents

        Use the search tool to perform the searches on the knowledge base."""

    def register_tools(self) -> None:
        """Register search-specific tools."""

        @self.agent.tool
        async def search(
            ctx: RunContext[ResearchDependencies],
            query: str,
            limit: int = 5,
        ) -> list[tuple[Chunk, float]]:
            """Execute search and return raw results from client."""
            # Use the default hybrid search
            search_results = await ctx.deps.client.search(query, limit=limit)

            # Expand context for better relevance
            expanded = await ctx.deps.client.expand_context(search_results)

            # Store in context (convert to SearchResult for context storage)
            from haiku.rag.research.base import SearchResult

            results_for_context = []
            for chunk, score in expanded:
                results_for_context.append(
                    SearchResult(
                        content=chunk.content,
                        score=score,
                        document_uri=chunk.document_uri or "",
                        metadata={"chunk_id": chunk.id} if chunk.id else {},
                    )
                )
            ctx.deps.context.add_search_result(query, results_for_context)

            # Return raw chunk, score tuples
            return expanded

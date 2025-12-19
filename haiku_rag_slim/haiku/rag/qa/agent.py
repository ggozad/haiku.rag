from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.output import ToolOutput

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.graph.research.models import Citation, RawSearchAnswer, resolve_citations
from haiku.rag.qa.prompts import QA_SYSTEM_PROMPT
from haiku.rag.store.models import SearchResult
from haiku.rag.utils import get_model


class Dependencies(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    client: HaikuRAG
    search_results: list[SearchResult] = []
    search_filter: str | None = None


class QuestionAnswerAgent:
    def __init__(
        self,
        client: HaikuRAG,
        model_config: ModelConfig,
        config: AppConfig | None = None,
        system_prompt: str | None = None,
    ):
        self._client = client
        model_obj = get_model(model_config, config)

        self._agent = Agent(
            model=model_obj,
            deps_type=Dependencies,
            output_type=ToolOutput(RawSearchAnswer, max_retries=3),
            instructions=system_prompt or QA_SYSTEM_PROMPT,
            retries=3,
        )

        @self._agent.tool
        async def search_documents(
            ctx: RunContext[Dependencies],
            query: str,
            limit: int | None = None,
        ) -> str:
            """Search the knowledge base for relevant documents.

            Returns results with chunk IDs and relevance scores.
            Reference results by their chunk_id in cited_chunks.
            """
            results = await ctx.deps.client.search(
                query, limit=limit, filter=ctx.deps.search_filter
            )
            results = await ctx.deps.client.expand_context(results)
            # Store results for citation resolution
            ctx.deps.search_results = results
            # Format with metadata for agent context
            parts = [r.format_for_agent() for r in results]
            return "\n\n".join(parts) if parts else "No results found."

    async def answer(
        self, question: str, filter: str | None = None
    ) -> tuple[str, list[Citation]]:
        """Answer a question using the RAG system.

        Args:
            question: The question to answer
            filter: SQL WHERE clause to filter documents

        Returns:
            Tuple of (answer text, list of resolved citations)
        """
        deps = Dependencies(client=self._client, search_filter=filter)
        result = await self._agent.run(question, deps=deps)
        citations = resolve_citations(result.output.cited_chunks, deps.search_results)
        return result.output.answer, citations

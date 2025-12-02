from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.output import ToolOutput

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.config.models import ModelConfig
from haiku.rag.graph.common import get_model
from haiku.rag.graph.common.models import (
    Citation,
    SearchAnswer,
    resolve_citations,
)
from haiku.rag.qa.prompts import QA_SYSTEM_PROMPT
from haiku.rag.store.models import SearchResult


class Dependencies(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    client: HaikuRAG
    search_results: list[SearchResult] = []


class QuestionAnswerAgent:
    def __init__(
        self,
        client: HaikuRAG,
        model_config: ModelConfig,
        system_prompt: str | None = None,
    ):
        self._client = client
        model_obj = get_model(model_config, Config)

        self._agent = Agent(
            model=model_obj,
            deps_type=Dependencies,
            output_type=ToolOutput(SearchAnswer, max_retries=3),
            instructions=system_prompt or QA_SYSTEM_PROMPT,
            retries=3,
        )

        @self._agent.tool
        async def search_documents(
            ctx: RunContext[Dependencies],
            query: str,
            limit: int = 5,
        ) -> str:
            """Search the knowledge base for relevant documents.

            Returns results with chunk IDs and relevance scores.
            Reference results by their chunk_id in cited_chunks.
            """
            results = await ctx.deps.client.search(query, limit=limit)
            results = await ctx.deps.client.expand_context(results)
            # Store results for citation resolution
            ctx.deps.search_results = results
            # Format with chunk IDs
            parts = []
            for r in results:
                parts.append(f"[{r.chunk_id}] (score: {r.score:.2f}) {r.content}")
            return "\n\n".join(parts) if parts else "No results found."

    async def answer(self, question: str) -> tuple[str, list[Citation]]:
        """Answer a question using the RAG system.

        Returns:
            Tuple of (answer text, list of resolved citations)
        """
        deps = Dependencies(client=self._client)
        result = await self._agent.run(question, deps=deps)
        citations = resolve_citations(result.output.cited_chunks, deps.search_results)
        return result.output.answer, citations

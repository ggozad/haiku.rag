from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider

from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


class RerankResult(BaseModel):
    """Individual rerank result with index and relevance score."""

    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    """Response from the reranking model containing ranked results."""

    results: list[RerankResult]


class OllamaReranker(RerankerBase):
    def __init__(self, model: str = Config.RERANK_MODEL):
        self._model = model

        # Create the reranking prompt
        system_prompt = """You are a document reranking assistant. Given a query and a list of document chunks, you must rank them by relevance to the query.

Return your response as a JSON object with a "results" array. Each result should have:
- "index": the original index of the document (integer)
- "relevance_score": a score between 0.0 and 1.0 indicating relevance (float, where 1.0 is most relevant)

Only return the top documents up to the requested limit, ordered by decreasing relevance score.
/no_think
"""

        model_obj = OpenAIModel(
            model_name=model,
            provider=OllamaProvider(base_url=f"{Config.OLLAMA_BASE_URL}/v1"),
        )

        self._agent = Agent(
            model=model_obj,
            output_type=RerankResponse,
            system_prompt=system_prompt,
        )

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({"index": i, "content": chunk.content})

        documents_text = ""
        for doc in documents:
            documents_text += f"Index {doc['index']}: {doc['content']}\n\n"

        user_prompt = f"""Query: {query}

Documents to rerank:
{documents_text.strip()}

Rank these documents by relevance to the query and return the top {top_n} results as JSON."""

        try:
            result = await self._agent.run(user_prompt)

            return [
                (chunks[result_item.index], result_item.relevance_score)
                for result_item in result.output.results[:top_n]
            ]

        except Exception:
            # Fallback: return chunks in original order with same score
            return [(chunks[i], 1.0) for i in range(min(top_n, len(chunks)))]

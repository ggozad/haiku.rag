import httpx

from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


class VLLMReranker(RerankerBase):
    def __init__(self, model: str):
        self._model = model
        self._base_url = Config.providers.vllm.rerank_base_url

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        # Prepare documents for reranking
        documents = [chunk.content for chunk in chunks]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/v1/rerank",
                json={"model": self._model, "query": query, "documents": documents},
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            result = response.json()

            # Extract scores and pair with chunks
            scored_chunks = []
            for item in result.get("results", []):
                index = item["index"]
                score = item["relevance_score"]
                scored_chunks.append((chunks[index], score))

            # Sort by score (descending) and return top_n
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            return scored_chunks[:top_n]

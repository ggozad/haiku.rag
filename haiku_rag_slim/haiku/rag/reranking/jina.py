import os

import httpx

from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


class JinaReranker(RerankerBase):
    """Jina AI reranker using the Jina Reranker API."""

    def __init__(self, model: str = "jina-reranker-v3"):
        self._model = model
        self._api_key = os.environ.get("JINA_API_KEY")
        if not self._api_key:
            raise ValueError("JINA_API_KEY environment variable required")

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        documents = [chunk.content for chunk in chunks]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.jina.ai/v1/rerank",
                json={
                    "model": self._model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            result = response.json()

            scored_chunks = []
            for item in result.get("results", []):
                index = item["index"]
                score = item["relevance_score"]
                scored_chunks.append((chunks[index], score))

            return scored_chunks

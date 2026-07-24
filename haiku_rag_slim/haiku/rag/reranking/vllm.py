import base64

import httpx

from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


def _document(chunk: Chunk) -> str | dict:
    """Rerank document for a chunk: plain text, or content parts carrying the
    picture bytes as a data URI when the chunk has them (multimodal rerank)."""
    data = chunk._picture_data
    if data is None:
        return chunk.content

    mime = "image/jpeg" if data.startswith(b"\xff\xd8") else "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    parts: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    ]
    if chunk.content:
        parts.append({"type": "text", "text": chunk.content})
    return {"content": parts}


class VLLMReranker(RerankerBase):
    def __init__(self, model: str, base_url: str):
        self._model = model
        self._base_url = base_url
        # One client reused across rerank calls (connection kept alive).
        self._client = httpx.AsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        documents = [_document(chunk) for chunk in chunks]

        response = await self._client.post(
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

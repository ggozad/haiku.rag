import base64

import httpx

from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.utils import image_media_type, vllm_base_url


def data_uri(data: bytes) -> str:
    """Picture bytes as a ``data:`` URI carrying their sniffed media type."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{image_media_type(data)};base64,{encoded}"


class VLLMReranker(RerankerBase):
    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        self._model = model
        self._base_url = vllm_base_url(base_url)
        self._headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        # One client reused across rerank calls (connection kept alive).
        # Multimodal document batches can take far longer than httpx's 5s
        # default timeout to score.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    def _scoreable(self, chunk: Chunk) -> bool:
        return bool(chunk.content or chunk._picture_data)

    def _document(self, chunk: Chunk) -> str | dict:
        """Rerank document for a chunk: plain text, or content parts carrying
        the picture bytes as a data URI when the chunk has them."""
        data = chunk._picture_data
        if data is None:
            return chunk.content

        parts: list[dict] = [
            {"type": "image_url", "image_url": {"url": data_uri(data)}}
        ]
        if chunk.content:
            parts.append({"type": "text", "text": chunk.content})
        return {"content": parts}

    async def _rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        documents = [self._document(chunk) for chunk in chunks]

        response = await self._client.post(
            f"{self._base_url}/rerank",
            json={"model": self._model, "query": query, "documents": documents},
            headers=self._headers,
        )
        response.raise_for_status()

        result = response.json()

        scored_chunks = []
        for item in result.get("results", []):
            index = item["index"]
            score = item["relevance_score"]
            scored_chunks.append((chunks[index], score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_n]

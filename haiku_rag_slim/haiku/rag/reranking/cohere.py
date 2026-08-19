from haiku.rag.utils import raise_missing_extra

try:
    import cohere
except ModuleNotFoundError as e:  # pragma: no cover
    raise_missing_extra("cohere", "cohere", e)

from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


class CohereReranker(RerankerBase):  # pragma: no cover
    def __init__(self, model: str | None = None):
        self._model = model
        # Cohere SDK reads CO_API_KEY from environment by default
        self._client = cohere.AsyncClientV2()

    async def _rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        documents = [chunk.content for chunk in chunks]

        model_name = self._model or "rerank-v3.5"
        response = await self._client.rerank(
            model=model_name, query=query, documents=documents, top_n=top_n
        )

        reranked_chunks = []
        for result in response.results:
            original_chunk = chunks[result.index]
            reranked_chunks.append((original_chunk, result.relevance_score))

        return reranked_chunks

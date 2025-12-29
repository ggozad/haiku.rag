from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk

try:
    import cohere
except ImportError as e:
    raise ImportError(
        "cohere is not installed. Please install it with `pip install cohere` or use the cohere optional dependency."
    ) from e


class CohereReranker(RerankerBase):  # pragma: no cover
    def __init__(self):
        # Cohere SDK reads CO_API_KEY from environment by default
        self._client = cohere.AsyncClientV2()

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

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

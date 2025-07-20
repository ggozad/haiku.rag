from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk

try:
    import cohere
except ImportError as e:
    raise ImportError(
        "cohere is not installed. Please install it with `pip install cohere` or use the cohere optional dependency."
    ) from e


class CohereReranker(RerankerBase):
    def __init__(self):
        self._client = cohere.ClientV2(api_key=Config.COHERE_API_KEY)

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        documents = [chunk.content for chunk in chunks]

        response = self._client.rerank(
            model=self._model, query=query, documents=documents, top_n=top_n
        )

        reranked_chunks = []
        for result in response.results:
            original_chunk = chunks[result.index]
            reranked_chunks.append((original_chunk, result.relevance_score))

        return reranked_chunks

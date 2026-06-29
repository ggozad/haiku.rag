from zeroentropy import AsyncZeroEntropy

from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


class ZeroEntropyReranker(RerankerBase):  # pragma: no cover
    """Zero Entropy reranker implementation using the zerank-1 model."""

    def __init__(self, model: str = "zerank-1"):
        """Initialize the Zero Entropy reranker.

        Args:
            model: The Zero Entropy model to use (default: "zerank-1")
        """
        self._model = model
        # Zero Entropy SDK reads ZEROENTROPY_API_KEY from environment by default
        self._client = AsyncZeroEntropy()

    async def _rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        # Prepare documents for Zero Entropy API
        documents = [chunk.content for chunk in chunks]

        # Call Zero Entropy reranking API
        model_name = self._model or "zerank-1"
        response = await self._client.models.rerank(
            model=model_name,
            query=query,
            documents=documents,
        )

        # Zero Entropy returns results sorted by relevance with scores
        reranked_results = []
        for result in response.results[:top_n]:
            chunk_index = result.index
            score = result.relevance_score

            if chunk_index < len(chunks):
                reranked_results.append((chunks[chunk_index], score))

        return reranked_results

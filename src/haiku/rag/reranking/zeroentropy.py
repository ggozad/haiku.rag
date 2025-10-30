from zeroentropy import ZeroEntropy

from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


class ZeroEntropyReranker(RerankerBase):
    """Zero Entropy reranker implementation using the zerank-1 model."""

    def __init__(self, model: str = "zerank-1"):
        """Initialize the Zero Entropy reranker.

        Args:
            model: The Zero Entropy model to use (default: "zerank-1")
        """
        self._model = model
        # Zero Entropy SDK reads ZEROENTROPY_API_KEY from environment by default
        self._client = ZeroEntropy()

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        """Rerank the given chunks based on relevance to the query.

        Args:
            query: The query to rank against
            chunks: The chunks to rerank
            top_n: The number of top results to return

        Returns:
            A list of (chunk, score) tuples, sorted by relevance
        """
        if not chunks:
            return []

        # Prepare documents for Zero Entropy API
        documents = [chunk.content for chunk in chunks]

        # Call Zero Entropy reranking API
        response = self._client.models.rerank(
            model=self._model,
            query=query,
            documents=documents,
        )

        # Extract results and map back to chunks
        # Zero Entropy returns results sorted by relevance with scores
        reranked_results = []

        # Get top_n results
        for i, result in enumerate(response.results[:top_n]):
            # Zero Entropy returns index and score for each document
            chunk_index = result.index
            score = result.relevance_score

            if chunk_index < len(chunks):
                reranked_results.append((chunks[chunk_index], score))

        return reranked_results

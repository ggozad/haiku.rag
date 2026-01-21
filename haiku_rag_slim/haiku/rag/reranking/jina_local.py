try:
    from transformers import (
        AutoModelForSequenceClassification,  # pyright: ignore[reportMissingImports]
    )
except ImportError as e:
    raise ImportError(
        "transformers is not installed. Please install it with `pip install transformers torch` "
        "or use the jina optional dependency."
    ) from e

from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


class JinaLocalReranker(RerankerBase):  # pragma: no cover
    """Jina reranker using local model inference via transformers.

    Note: The Jina Reranker v3 model is licensed under CC BY-NC 4.0,
    which restricts commercial use.
    """

    def __init__(self, model: str = "jinaai/jina-reranker-v3"):
        self._model = model
        self._reranker = AutoModelForSequenceClassification.from_pretrained(
            model, trust_remote_code=True
        )

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        documents = [chunk.content for chunk in chunks]
        sentence_pairs = [[query, doc] for doc in documents]

        scores = self._reranker.compute_score(sentence_pairs)

        # Handle both single score and list of scores
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored_chunks = list(zip(chunks, scores, strict=False))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [(chunk, float(score)) for chunk, score in scored_chunks[:top_n]]

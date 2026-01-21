try:
    from transformers import (
        AutoModel,  # pyright: ignore[reportMissingImports]
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
        self._reranker = AutoModel.from_pretrained(model, trust_remote_code=True)
        self._reranker.eval()

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        documents = [chunk.content for chunk in chunks]

        results = self._reranker.rerank(query, documents, top_n=top_n)

        return [(chunks[r["index"]], float(r["relevance_score"])) for r in results]

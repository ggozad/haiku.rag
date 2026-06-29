import asyncio

try:
    from sentence_transformers import (
        CrossEncoder,  # pyright: ignore[reportMissingImports]
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is not installed. Install it with "
        "`pip install sentence-transformers` or use the cross-encoder optional dependency."
    ) from e

from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


class CrossEncoderReranker(RerankerBase):
    """Reranker for any sentence-transformers CrossEncoder model.

    Loads the model in-process. Pass any HuggingFace cross-encoder reranker
    as ``model`` (e.g. ``BAAI/bge-reranker-v2-m3``, ``Qwen/Qwen3-Reranker-0.6B``,
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``).
    """

    def __init__(self, model: str):
        self._model = model
        self._reranker = CrossEncoder(model)

    async def _rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        documents = [chunk.content for chunk in chunks]
        rankings = await asyncio.to_thread(
            lambda: self._reranker.rank(query, documents, top_k=top_n)
        )
        return [(chunks[r["corpus_id"]], float(r["score"])) for r in rankings]

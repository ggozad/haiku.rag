import asyncio

from haiku.rag.utils import raise_missing_extra

try:
    from transformers import AutoModel
except ModuleNotFoundError as e:  # pragma: no cover
    if e.name not in ("torch", "transformers"):
        raise
    raise_missing_extra(e.name, "jina", e)

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

    async def _rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        documents = [chunk.content for chunk in chunks]

        results = await asyncio.to_thread(
            lambda: self._reranker.rerank(query, documents, top_n=top_n)
        )

        return [(chunks[r["index"]], float(r["relevance_score"])) for r in results]

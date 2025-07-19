from mxbai_rerank import MxbaiRerankV2

from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk


class MxBAIReranker(RerankerBase):
    def __init__(self):
        self._client = MxbaiRerankV2(
            Config.RERANK_MODEL, disable_transformers_warnings=True
        )

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []

        documents = [chunk.content for chunk in chunks]

        results = self._client.rank(query=query, documents=documents, top_k=top_n)
        reranked_chunks = []
        for result in results:
            original_chunk = chunks[result.index]
            reranked_chunks.append((original_chunk, result.score))

        return reranked_chunks

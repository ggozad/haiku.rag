import asyncio
import threading

import tqdm
from mxbai_rerank import MxbaiRerankV2  # pyright: ignore[reportMissingImports]

from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase
from haiku.rag.store.models.chunk import Chunk

# tqdm's default class lock is a multiprocessing.RLock; constructing it spawns
# resource_tracker, which inherits sys.stderr's fileno. Inside Textual's chat
# TUI, sys.stderr.fileno() returns -1, landing in fds_to_keep and failing the
# fork_exec validation. A threading lock is sufficient since we never share
# tqdm progress bars across processes.
tqdm.tqdm.set_lock(threading.RLock())


class MxBAIReranker(RerankerBase):
    def __init__(self):
        model_name = (
            Config.reranking.model.name
            if Config.reranking.model
            else "mixedbread-ai/mxbai-rerank-base-v2"
        )
        self._client = MxbaiRerankV2(model_name, disable_transformers_warnings=True)

    async def _rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        documents = [chunk.content for chunk in chunks]

        results = await asyncio.to_thread(
            lambda: self._client.rank(query=query, documents=documents, top_k=top_n)
        )
        reranked_chunks = []
        for result in results:
            original_chunk = chunks[result.index]
            reranked_chunks.append((original_chunk, result.score))

        return reranked_chunks

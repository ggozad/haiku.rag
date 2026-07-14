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


def _prepare_for_model(
    ids: list[int],
    pair_ids: list[int] | None = None,
    max_length: int | None = None,
    **_,
) -> dict[str, list[int]]:
    # transformers 5.x removed tokenizer.prepare_for_model, which mxbai-rerank
    # calls with add_special_tokens=False and truncation="only_second"; for that
    # call pattern it reduces to truncating the pair and concatenating.
    pair_ids = pair_ids or []
    if max_length is not None:
        pair_ids = pair_ids[: max(0, max_length - len(ids))]
    return {"input_ids": ids + pair_ids}


class MxBAIReranker(RerankerBase):
    def __init__(self):
        model_name = (
            Config.reranking.model.name
            if Config.reranking.model
            else "mixedbread-ai/mxbai-rerank-base-v2"
        )
        self._client = MxbaiRerankV2(model_name, disable_transformers_warnings=True)
        if not hasattr(self._client.tokenizer, "prepare_for_model"):
            self._client.tokenizer.prepare_for_model = _prepare_for_model

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

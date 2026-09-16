from haiku.rag.store.models.chunk import Chunk


class RerankerBase:
    _model: str | None = None

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        chunks = [chunk for chunk in chunks if self._scoreable(chunk)]
        if not chunks:
            return []
        return await self._rerank(query, chunks, top_n)

    def _scoreable(self, chunk: Chunk) -> bool:
        """Whether this reranker has anything to send for `chunk`.

        A picture chunk has no text, and its bytes are attached only under
        reranking.multimodal, so it can arrive with nothing to score.
        """
        return bool(chunk.content)

    async def _rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        """Score and order `chunks`, returning the top `top_n`.

        Return objects taken from `chunks`, not copies: searching several
        databases maps a scored chunk back to the one holding it by identity,
        because chunk ids repeat between copies of a database.
        """
        raise NotImplementedError(
            "Reranker is an abstract class. Please implement the _rerank method in a subclass."
        )

    async def aclose(self) -> None:
        """Release resources held by the reranker. No-op by default;
        rerankers that own an HTTP client override this."""

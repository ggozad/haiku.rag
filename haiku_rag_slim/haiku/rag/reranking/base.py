from haiku.rag.store.models.chunk import Chunk


class RerankerBase:
    _model: str | None = None

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        return await self._rerank(query, chunks, top_n)

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

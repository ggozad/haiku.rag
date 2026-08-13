from haiku.rag.config import Config
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.telemetry import logfire


class RerankerBase:
    _model: str | None = Config.reranking.model.name if Config.reranking.model else None

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        # Instrumented here rather than in each subclass: this is the one
        # choke point every provider passes through, so the span covers
        # local weights and remote APIs alike. `candidates` is the search's
        # limit*10 fan-out, the main lever on how long scoring takes.
        with logfire.span(
            "search.rerank",
            provider=type(self).__name__,
            model=self._model,
            candidates=len(chunks),
            top_n=top_n,
        ) as span:
            results = await self._rerank(query, chunks, top_n)
            scores = [score for _, score in results]
            span.set_attribute("results", len(results))
            # Spread separates a reranker that discriminated from one whose
            # scores saturated onto the same value, leaving order to the sort.
            span.set_attribute("top_score", max(scores) if scores else None)
            span.set_attribute(
                "score_spread", max(scores) - min(scores) if scores else None
            )
            return results

    async def _rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError(
            "Reranker is an abstract class. Please implement the _rerank method in a subclass."
        )

    async def aclose(self) -> None:
        """Release resources held by the reranker. No-op by default;
        rerankers that own an HTTP client override this."""

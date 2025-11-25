from haiku.rag.config import Config
from haiku.rag.store.models.chunk import Chunk


class RerankerBase:
    _model: str | None = Config.reranking.model.name if Config.reranking.model else None

    async def rerank(
        self, query: str, chunks: list[Chunk], top_n: int = 10
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError(
            "Reranker is an abstract class. Please implement the rerank method in a subclass."
        )

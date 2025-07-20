from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase

try:
    from haiku.rag.reranking.cohere import CohereReranker
except ImportError:
    pass

_reranker: RerankerBase | None = None


def get_reranker() -> RerankerBase:
    """
    Factory function to get the appropriate reranker based on the configuration.
    """
    global _reranker
    if _reranker is not None:
        return _reranker
    if Config.RERANK_PROVIDER == "mxbai":
        from haiku.rag.reranking.mxbai import MxBAIReranker

        _reranker = MxBAIReranker()
        return _reranker

    if Config.RERANK_PROVIDER == "cohere":
        try:
            from haiku.rag.reranking.cohere import CohereReranker
        except ImportError:
            raise ImportError(
                "Cohere reranker requires the 'cohere' package. "
                "Please install haiku.rag with the 'cohere' extra:"
                "uv pip install haiku.rag[cohere]"
            )
        _reranker = CohereReranker()
        return _reranker

    raise ValueError(f"Unsupported reranker provider: {Config.RERANK_PROVIDER}")

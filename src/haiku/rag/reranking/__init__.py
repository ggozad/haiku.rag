from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase

_reranker: RerankerBase | None = None


def get_reranker() -> RerankerBase | None:
    """
    Factory function to get the appropriate reranker based on the configuration.
    Returns None if if reranking is disabled.
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    if Config.RERANK_PROVIDER == "mxbai":
        try:
            from haiku.rag.reranking.mxbai import MxBAIReranker

            _reranker = MxBAIReranker()
            return _reranker
        except ImportError:
            return None

    if Config.RERANK_PROVIDER == "cohere":
        try:
            from haiku.rag.reranking.cohere import CohereReranker

            _reranker = CohereReranker()
            return _reranker
        except ImportError:
            return None

    if Config.RERANK_PROVIDER == "ollama":
        from haiku.rag.reranking.ollama import OllamaReranker

        _reranker = OllamaReranker()
        return _reranker

    return None

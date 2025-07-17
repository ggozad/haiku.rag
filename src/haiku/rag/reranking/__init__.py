from haiku.rag.config import Config
from haiku.rag.reranking.base import RerankerBase

try:
    from haiku.rag.reranking.cohere import CohereReranker
except ImportError:
    pass


def get_reranker() -> RerankerBase:
    """
    Factory function to get the appropriate reranker based on the configuration.
    """

    if Config.RERANK_PROVIDER == "cohere":
        try:
            from haiku.rag.reranking.cohere import CohereReranker
        except ImportError:
            raise ImportError(
                "Cohere reranker requires the 'cohere' package. "
                "Please install haiku.rag with the 'cohere' extra:"
                "uv pip install haiku.rag --extra cohere"
            )
        return CohereReranker()

    raise ValueError(f"Unsupported reranker provider: {Config.RERANK_PROVIDER}")

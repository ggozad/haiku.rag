import os

from haiku.rag.config import AppConfig, Config
from haiku.rag.reranking.base import RerankerBase


def get_reranker(config: AppConfig = Config) -> RerankerBase | None:
    """
    Factory function to get the appropriate reranker based on the configuration.
    Returns None if reranking is disabled.

    Args:
        config: Configuration to use. Defaults to global Config.

    Returns:
        A reranker instance if configured, None otherwise.
    """
    if config.reranking.model and config.reranking.model.provider == "mxbai":
        try:
            from haiku.rag.reranking.mxbai import MxBAIReranker

            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            return MxBAIReranker()
        except ImportError:  # pragma: no cover
            return None

    if config.reranking.model and config.reranking.model.provider == "cohere":
        try:
            from haiku.rag.reranking.cohere import CohereReranker

            return CohereReranker()
        except ImportError:  # pragma: no cover
            return None

    if config.reranking.model and config.reranking.model.provider == "vllm":
        try:
            from haiku.rag.reranking.vllm import VLLMReranker

            base_url = config.reranking.model.base_url
            if not base_url:
                raise ValueError("vLLM reranker requires base_url in reranking.model")
            return VLLMReranker(config.reranking.model.name, base_url)
        except ImportError:  # pragma: no cover
            return None

    if config.reranking.model and config.reranking.model.provider == "zeroentropy":
        try:
            from haiku.rag.reranking.zeroentropy import ZeroEntropyReranker

            model = config.reranking.model.name or "zerank-1"
            return ZeroEntropyReranker(model)
        except ImportError:  # pragma: no cover
            return None

    return None

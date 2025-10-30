import os

from haiku.rag.config import AppConfig, Config
from haiku.rag.reranking.base import RerankerBase

_reranker_cache: dict[int, RerankerBase | None] = {}


def get_reranker(config: AppConfig = Config) -> RerankerBase | None:
    """
    Factory function to get the appropriate reranker based on the configuration.
    Returns None if reranking is disabled.

    Args:
        config: Configuration to use. Defaults to global Config.

    Returns:
        A reranker instance if configured, None otherwise.
    """
    # Use config id as cache key to support multiple configs
    config_id = id(config)
    if config_id in _reranker_cache:
        return _reranker_cache[config_id]

    reranker: RerankerBase | None = None

    if config.reranking.provider == "mxbai":
        try:
            from haiku.rag.reranking.mxbai import MxBAIReranker

            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            reranker = MxBAIReranker()
        except ImportError:
            reranker = None

    elif config.reranking.provider == "cohere":
        try:
            from haiku.rag.reranking.cohere import CohereReranker

            reranker = CohereReranker()
        except ImportError:
            reranker = None

    elif config.reranking.provider == "vllm":
        try:
            from haiku.rag.reranking.vllm import VLLMReranker

            reranker = VLLMReranker(config.reranking.model)
        except ImportError:
            reranker = None

    elif config.reranking.provider == "zeroentropy":
        try:
            from haiku.rag.reranking.zeroentropy import ZeroEntropyReranker

            # Use configured model or default to zerank-1
            model = config.reranking.model or "zerank-1"
            reranker = ZeroEntropyReranker(model)
        except ImportError:
            reranker = None

    _reranker_cache[config_id] = reranker
    return reranker

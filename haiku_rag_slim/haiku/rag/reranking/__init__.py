from haiku.rag.config import AppConfig, Config
from haiku.rag.reranking.base import RerankerBase


def get_reranker(config: AppConfig = Config) -> RerankerBase | None:
    """Build the configured reranker, or None if reranking is disabled or its
    optional dependency is not installed."""
    model = config.reranking.model
    if model is None:
        return None

    if config.reranking.multimodal and model.provider != "vllm":
        raise ValueError("reranking.multimodal is only supported on the vllm provider")

    try:
        if model.provider == "cohere":
            from haiku.rag.reranking.cohere import CohereReranker

            return CohereReranker()

        if model.provider == "vllm":
            if not model.base_url:
                raise ValueError("vLLM reranker requires base_url in reranking.model")
            from haiku.rag.reranking.vllm import VLLMReranker

            return VLLMReranker(model.name, model.base_url)

        if model.provider == "zeroentropy":
            from haiku.rag.reranking.zeroentropy import ZeroEntropyReranker

            return ZeroEntropyReranker(model.name or "zerank-1")

        if model.provider == "jina":
            from haiku.rag.reranking.jina import JinaReranker

            return JinaReranker(model.name or "jina-reranker-v3")

        if model.provider == "jina-local":
            from haiku.rag.reranking.jina_local import JinaLocalReranker

            return JinaLocalReranker(model.name or "jinaai/jina-reranker-v3")

        if model.provider == "cross-encoder":
            if not model.name:
                raise ValueError(
                    "cross-encoder reranker requires name in reranking.model"
                )
            from haiku.rag.reranking.cross_encoder import CrossEncoderReranker

            return CrossEncoderReranker(model.name)
    except ImportError:  # pragma: no cover
        return None

    raise ValueError(f"Unknown reranking provider: {model.provider}")

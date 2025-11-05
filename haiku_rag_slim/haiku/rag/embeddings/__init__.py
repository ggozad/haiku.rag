from haiku.rag.config import AppConfig, Config
from haiku.rag.embeddings.base import EmbedderBase
from haiku.rag.embeddings.ollama import Embedder as OllamaEmbedder


def get_embedder(config: AppConfig = Config) -> EmbedderBase:
    """
    Factory function to get the appropriate embedder based on the configuration.

    Args:
        config: Configuration to use. Defaults to global Config.

    Returns:
        An embedder instance configured according to the config.
    """

    if config.embeddings.provider == "ollama":
        return OllamaEmbedder(
            config.embeddings.model, config.embeddings.vector_dim, config
        )

    if config.embeddings.provider == "voyageai":
        try:
            from haiku.rag.embeddings.voyageai import Embedder as VoyageAIEmbedder
        except ImportError:
            raise ImportError(
                "VoyageAI embedder requires the 'voyageai' package. "
                "Please install haiku.rag with the 'voyageai' extra: "
                "uv pip install haiku.rag[voyageai]"
            )
        return VoyageAIEmbedder(
            config.embeddings.model, config.embeddings.vector_dim, config
        )

    if config.embeddings.provider == "openai":
        from haiku.rag.embeddings.openai import Embedder as OpenAIEmbedder

        return OpenAIEmbedder(
            config.embeddings.model, config.embeddings.vector_dim, config
        )

    if config.embeddings.provider == "vllm":
        from haiku.rag.embeddings.vllm import Embedder as VllmEmbedder

        return VllmEmbedder(
            config.embeddings.model, config.embeddings.vector_dim, config
        )

    raise ValueError(f"Unsupported embedding provider: {config.embeddings.provider}")

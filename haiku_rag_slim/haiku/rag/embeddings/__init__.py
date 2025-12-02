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
    embedding_model = config.embeddings.model

    if embedding_model.provider == "ollama":
        return OllamaEmbedder(embedding_model.name, embedding_model.vector_dim, config)

    if embedding_model.provider == "voyageai":
        try:
            from haiku.rag.embeddings.voyageai import Embedder as VoyageAIEmbedder
        except ImportError:
            raise ImportError(
                "VoyageAI embedder requires the 'voyageai' package. "
                "Please install haiku.rag with the 'voyageai' extra: "
                "uv pip install haiku.rag[voyageai]"
            )
        return VoyageAIEmbedder(
            embedding_model.name, embedding_model.vector_dim, config
        )

    if embedding_model.provider == "openai":
        from haiku.rag.embeddings.openai import Embedder as OpenAIEmbedder

        return OpenAIEmbedder(embedding_model.name, embedding_model.vector_dim, config)

    if embedding_model.provider == "vllm":
        from haiku.rag.embeddings.vllm import Embedder as VllmEmbedder

        return VllmEmbedder(embedding_model.name, embedding_model.vector_dim, config)

    if embedding_model.provider == "lm_studio":
        from haiku.rag.embeddings.lm_studio import Embedder as LMStudioEmbedder

        return LMStudioEmbedder(
            embedding_model.name, embedding_model.vector_dim, config
        )

    raise ValueError(f"Unsupported embedding provider: {embedding_model.provider}")

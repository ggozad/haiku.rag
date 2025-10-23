from haiku.rag.config import Config
from haiku.rag.embeddings.base import EmbedderBase
from haiku.rag.embeddings.ollama import Embedder as OllamaEmbedder


def get_embedder() -> EmbedderBase:
    """
    Factory function to get the appropriate embedder based on the configuration.
    """

    if Config.embeddings.provider == "ollama":
        return OllamaEmbedder(Config.embeddings.model, Config.embeddings.vector_dim)

    if Config.embeddings.provider == "voyageai":
        try:
            from haiku.rag.embeddings.voyageai import Embedder as VoyageAIEmbedder
        except ImportError:
            raise ImportError(
                "VoyageAI embedder requires the 'voyageai' package. "
                "Please install haiku.rag with the 'voyageai' extra: "
                "uv pip install haiku.rag[voyageai]"
            )
        return VoyageAIEmbedder(Config.embeddings.model, Config.embeddings.vector_dim)

    if Config.embeddings.provider == "openai":
        from haiku.rag.embeddings.openai import Embedder as OpenAIEmbedder

        return OpenAIEmbedder(Config.embeddings.model, Config.embeddings.vector_dim)

    if Config.embeddings.provider == "vllm":
        from haiku.rag.embeddings.vllm import Embedder as VllmEmbedder

        return VllmEmbedder(Config.embeddings.model, Config.embeddings.vector_dim)

    raise ValueError(f"Unsupported embedding provider: {Config.embeddings.provider}")

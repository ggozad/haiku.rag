from haiku.rag.config import Config
from haiku.rag.embeddings.base import EmbedderBase
from haiku.rag.embeddings.ollama import Embedder as OllamaEmbedder


def get_embedder() -> EmbedderBase:
    """
    Factory function to get the appropriate embedder based on the configuration.
    """

    if Config.EMBEDDINGS_PROVIDER == "ollama":
        return OllamaEmbedder(Config.EMBEDDINGS_MODEL, Config.EMBEDDINGS_VECTOR_DIM)

    if Config.EMBEDDINGS_PROVIDER == "voyageai":
        try:
            from haiku.rag.embeddings.voyageai import Embedder as VoyageAIEmbedder
        except ImportError:
            raise ImportError(
                "VoyageAI embedder requires the 'voyageai' package. "
                "Please install haiku.rag with the 'voyageai' extra: "
                "uv pip install haiku.rag[voyageai]"
            )
        return VoyageAIEmbedder(Config.EMBEDDINGS_MODEL, Config.EMBEDDINGS_VECTOR_DIM)

    if Config.EMBEDDINGS_PROVIDER == "openai":
        from haiku.rag.embeddings.openai import Embedder as OpenAIEmbedder

        return OpenAIEmbedder(Config.EMBEDDINGS_MODEL, Config.EMBEDDINGS_VECTOR_DIM)

    if Config.EMBEDDINGS_PROVIDER == "vllm":
        from haiku.rag.embeddings.vllm import Embedder as VllmEmbedder

        return VllmEmbedder(Config.EMBEDDINGS_MODEL, Config.EMBEDDINGS_VECTOR_DIM)

    raise ValueError(f"Unsupported embedding provider: {Config.EMBEDDINGS_PROVIDER}")

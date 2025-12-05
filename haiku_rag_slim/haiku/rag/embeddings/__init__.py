from typing import TYPE_CHECKING

from haiku.rag.config import AppConfig, Config
from haiku.rag.embeddings.base import EmbedderBase
from haiku.rag.embeddings.ollama import Embedder as OllamaEmbedder

if TYPE_CHECKING:
    from haiku.rag.store.models.chunk import Chunk


def contextualize(chunks: list["Chunk"]) -> list[str]:
    """Prepare chunk content for embedding by adding context.

    Prepends section headings to chunk content for better semantic search.
    The embeddings will capture section context while stored content stays raw.

    Args:
        chunks: List of chunks to contextualize.

    Returns:
        List of contextualized text strings for embedding.
    """
    texts = []
    for chunk in chunks:
        meta = chunk.get_chunk_metadata()
        if meta.headings:
            text = "\n".join(meta.headings) + "\n" + chunk.content
        else:
            text = chunk.content
        texts.append(text)
    return texts


async def embed_chunks(
    chunks: list["Chunk"], config: AppConfig = Config
) -> list["Chunk"]:
    """Generate embeddings for chunks.

    Contextualizes chunks (prepends headings) before embedding for better
    semantic search. Returns new Chunk objects with embeddings set.

    Args:
        chunks: List of chunks to embed.
        config: Configuration for embedder selection.

    Returns:
        New list of Chunk objects with embedding field populated.
    """
    if not chunks:
        return []

    from haiku.rag.store.models.chunk import Chunk

    embedder = get_embedder(config)
    texts = contextualize(chunks)
    embeddings = await embedder.embed(texts)

    return [
        Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            metadata=chunk.metadata,
            order=chunk.order,
            document_uri=chunk.document_uri,
            document_title=chunk.document_title,
            document_meta=chunk.document_meta,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]


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

from typing import TYPE_CHECKING, Any

from pydantic_ai.embeddings import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import AppConfig, Config

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from haiku.rag.store.models.chunk import Chunk


ImageInput = "bytes | PILImage.Image"


class EmbedderWrapper:
    """Wrapper around pydantic-ai Embedder with explicit query/document methods.

    Subclasses set ``supports_images = True`` and override the image methods
    when the underlying model can encode pictures into the same vector space.
    """

    supports_images: bool = False

    def __init__(self, embedder: Embedder | None, vector_dim: int):
        self._embedder = embedder
        self._vector_dim = vector_dim

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        assert self._embedder is not None
        result = await self._embedder.embed_query(text)
        return list(result.embeddings[0])

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents/chunks for indexing."""
        if not texts:
            return []
        assert self._embedder is not None
        result = await self._embedder.embed_documents(texts)
        return [list(e) for e in result.embeddings]

    async def embed_image_query(self, image: "Any") -> list[float]:
        """Embed a single image as a search query."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support image embedding. "
            "Configure a multimodal provider (e.g. provider='mlx' or "
            "provider='vllm')."
        )

    async def embed_images(self, images: list["Any"]) -> list[list[float]]:
        """Batch-embed images for indexing into the same vector space as text."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support image embedding. "
            "Configure a multimodal provider (e.g. provider='mlx' or "
            "provider='vllm')."
        )


def contextualize(chunks: list["Chunk"]) -> list[str]:
    """Prepare chunk content for embedding/FTS by adding context.

    Prepends section headings to chunk content for better semantic search.

    Args:
        chunks: List of chunks to contextualize.

    Returns:
        List of contextualized text strings.
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


EMBEDDING_BATCH_SIZE = 512


async def embed_chunks(
    chunks: list["Chunk"], config: AppConfig = Config
) -> list["Chunk"]:
    """Generate embeddings for chunks.

    Contextualizes chunks (prepends headings) before embedding for better
    semantic search. Returns new Chunk objects with embeddings set.

    Embeddings are generated in batches to avoid request size limits
    and timeouts with large document sets.

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

    # Batch embedding calls to avoid request size limits
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = await embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

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
        for chunk, embedding in zip(chunks, all_embeddings)
    ]


def get_embedder(config: AppConfig = Config) -> EmbedderWrapper:
    """Factory function to get the appropriate embedder based on the configuration.

    Args:
        config: Configuration to use. Defaults to global Config.

    Returns:
        An embedder instance configured according to the config.
    """
    embedding_model = config.embeddings.model
    provider = embedding_model.provider
    model_name = embedding_model.name
    vector_dim = embedding_model.vector_dim

    if provider == "ollama":
        # Use model-level base_url if set, otherwise fall back to providers config
        base_url = embedding_model.base_url or f"{config.providers.ollama.base_url}/v1"
        model = OpenAIEmbeddingModel(
            model_name,
            provider=OllamaProvider(base_url=base_url),
        )
        return EmbedderWrapper(Embedder(model), vector_dim)

    if provider == "openai":
        if embedding_model.base_url:
            model = OpenAIEmbeddingModel(
                model_name,
                provider=OpenAIProvider(base_url=embedding_model.base_url),
            )
            return EmbedderWrapper(Embedder(model), vector_dim)
        return EmbedderWrapper(Embedder(f"openai:{model_name}"), vector_dim)

    if provider == "voyageai":
        return EmbedderWrapper(Embedder(f"voyageai:{model_name}"), vector_dim)

    if provider == "cohere":
        return EmbedderWrapper(Embedder(f"cohere:{model_name}"), vector_dim)

    if provider == "sentence-transformers":
        return EmbedderWrapper(
            Embedder(f"sentence-transformers:{model_name}"), vector_dim
        )

    if provider == "mlx":
        from haiku.rag.embeddings.mlx import MLXEmbedder

        return MLXEmbedder(model_name, vector_dim)

    if provider == "vllm":
        from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

        base_url = embedding_model.base_url or "http://localhost:8000/v1"
        return VLLMMultimodalEmbedder(model_name, vector_dim, base_url=base_url)

    raise ValueError(f"Unsupported embedding provider: {provider}")

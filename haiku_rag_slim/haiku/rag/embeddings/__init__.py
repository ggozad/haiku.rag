from typing import TYPE_CHECKING, Any

from pydantic_ai.embeddings import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import AppConfig, Config

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from haiku.rag.config.models import EmbeddingModelConfig
    from haiku.rag.store.models.chunk import Chunk


ImageInput = "bytes | PILImage.Image"


class EmbedderWrapper:
    """Wrapper around pydantic-ai Embedder with explicit query/document methods.

    Subclasses that can encode pictures into the same vector space as text either
    set the ``supports_images`` class attribute or pass ``supports_images=True``,
    and override the image methods.
    """

    supports_images: bool = False

    def __init__(
        self,
        embedder: Embedder | None,
        vector_dim: int,
        supports_images: bool | None = None,
    ):
        self._embedder = embedder
        self._vector_dim = vector_dim
        if supports_images is not None:
            self.supports_images = supports_images

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

    async def embed_image(self, image: "Any") -> list[float]:
        """Embed a single image into the same vector space as text.

        Multimodal providers override this. Picture embedding is single-image:
        vLLM's ``/v1/embeddings`` accepts one image per request via the
        ``messages`` superset. Callers loop when they need many.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support image embedding. Set "
            "embeddings.model.multimodal: true on a vllm, voyageai, or cohere model."
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


async def embed_chunks(
    chunks: list["Chunk"], embedder: "EmbedderWrapper", config: AppConfig = Config
) -> list["Chunk"]:
    """Generate embeddings for chunks, dispatching text vs picture variants.

    Text chunks are contextualized (headings prepended) and routed through
    ``embed_documents``. Picture chunks (those carrying ``_picture_data``)
    are routed through ``embed_images`` and require a multimodal embedder.
    Vectors land in the original chunk order.
    """
    if not chunks:
        return []

    from haiku.rag.store.models.chunk import Chunk

    text_chunks: list[Chunk] = []
    picture_chunks: list[Chunk] = []
    for chunk in chunks:
        if chunk._picture_data is not None:
            picture_chunks.append(chunk)
        else:
            text_chunks.append(chunk)

    text_embeddings: list[list[float]] = []
    if text_chunks:
        texts = contextualize(text_chunks)
        batch_size = config.embeddings.batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            text_embeddings.extend(await embedder.embed_documents(batch))

    picture_embeddings: list[list[float]] = []
    if picture_chunks:
        if not embedder.supports_images:
            raise ValueError(
                "Picture chunks require a multimodal embedder. Set "
                "embeddings.model.multimodal: true on a vllm, voyageai, or cohere "
                "model, or omit picture chunks."
            )
        for chunk in picture_chunks:
            picture_embeddings.append(await embedder.embed_image(chunk._picture_data))

    text_iter = iter(text_embeddings)
    picture_iter = iter(picture_embeddings)
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
            embedding=(
                next(picture_iter)
                if chunk._picture_data is not None
                else next(text_iter)
            ),
        )
        for chunk in chunks
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

    if embedding_model.multimodal:
        return _get_multimodal_embedder(embedding_model)

    if provider == "ollama":
        # Use model-level base_url if set, otherwise fall back to providers config
        base_url = embedding_model.base_url or config.providers.ollama.base_url
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
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

    if provider == "vllm":
        from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

        base_url = _vllm_base_url(embedding_model.base_url)
        return VLLMMultimodalEmbedder(
            model_name, vector_dim, base_url=base_url, supports_images=False
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


def _vllm_base_url(base_url: str | None) -> str:
    base_url = base_url or "http://localhost:8000/v1"
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return base_url


def _get_multimodal_embedder(
    embedding_model: "EmbeddingModelConfig",
) -> EmbedderWrapper:
    """Build an image-capable embedder for providers that support multimodal.

    Each provider passes images in its own wire format, so the capability lives
    in a per-provider embedder rather than a generic flag.
    """
    provider = embedding_model.provider
    model_name = embedding_model.name
    vector_dim = embedding_model.vector_dim

    if provider == "vllm":
        from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

        base_url = _vllm_base_url(embedding_model.base_url)
        return VLLMMultimodalEmbedder(
            model_name, vector_dim, base_url=base_url, supports_images=True
        )

    if provider == "voyageai":
        from haiku.rag.embeddings.voyageai import VoyageMultimodalEmbedder

        return VoyageMultimodalEmbedder(model_name, vector_dim)

    raise ValueError(
        f"Provider '{provider}' does not support multimodal embedding. Set "
        "embeddings.model.multimodal: true on a vllm, voyageai, or cohere model."
    )

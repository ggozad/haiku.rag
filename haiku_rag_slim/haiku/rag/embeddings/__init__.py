import base64
import hashlib
import io
from typing import TYPE_CHECKING, Any

import httpx
from pydantic_ai.embeddings import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import AppConfig, Config

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from haiku.rag.config.models import EmbeddingHTTPConfig, EmbeddingModelConfig
    from haiku.rag.store.models.chunk import Chunk


def build_http_client(http: "EmbeddingHTTPConfig") -> httpx.AsyncClient:
    """A pooled ``httpx.AsyncClient`` configured from ``EmbeddingHTTPConfig``.

    Shared across an embedder's requests so a connection (and its name
    resolution) is established once and kept warm rather than rebuilt per call.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(http.timeout_s),
        limits=httpx.Limits(
            max_connections=http.max_connections,
            max_keepalive_connections=http.max_keepalive_connections,
            keepalive_expiry=http.keepalive_expiry_s,
        ),
    )


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
        *,
        owned_http_client: "Any | None" = None,
    ):
        self._embedder = embedder
        self._vector_dim = vector_dim
        if supports_images is not None:
            self.supports_images = supports_images
        # An httpx.AsyncClient this wrapper built and must close on teardown —
        # e.g. the pooled client passed to an openai/ollama provider. None when
        # the underlying SDK owns its own transport.
        self._owned_http_client = owned_http_client

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
        return await self._embed_documents(texts)

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
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

    async def aclose(self) -> None:
        """Release any resources held by the embedder. Closes a pooled HTTP
        client this wrapper owns (openai/ollama); a no-op otherwise. Lets
        callers tear down uniformly regardless of embedder type. Subclasses
        that own their own client (e.g. vLLM) override this."""
        if self._owned_http_client is not None:
            await self._owned_http_client.aclose()
            self._owned_http_client = None


def _to_data_uri(image: "bytes | PILImage.Image") -> str:
    """Render an image as a ``data:image/png;base64,...`` URI."""
    if isinstance(image, bytes):
        return f"data:image/png;base64,{base64.b64encode(image).decode('ascii')}"

    from PIL import Image as PILImageModule

    if isinstance(image, PILImageModule.Image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return (
            f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        )

    raise TypeError(f"Unsupported image type: {type(image)!r}")


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
        # Identical image bytes embed to identical vectors, so embed each
        # distinct image once and reuse the result for every chunk that shares
        # it. A document that repeats one figure across many pages (header,
        # watermark, logo) collapses from one request per occurrence to one
        # per unique image. Keyed by a FIPS-safe content hash; order is
        # preserved because we append one vector per chunk in chunk order.
        embedding_cache: dict[bytes, list[float]] = {}
        for chunk in picture_chunks:
            data = chunk._picture_data
            key = (
                hashlib.sha256(data, usedforsecurity=False).digest()
                if isinstance(data, bytes | bytearray)
                else None
            )
            if key is not None and (cached := embedding_cache.get(key)) is not None:
                picture_embeddings.append(cached)
                continue
            embedding = await embedder.embed_image(data)
            if key is not None:
                embedding_cache[key] = embedding
            picture_embeddings.append(embedding)

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
    http = config.embeddings.http
    provider = embedding_model.provider
    model_name = embedding_model.name
    vector_dim = embedding_model.vector_dim

    if embedding_model.multimodal:
        return _get_multimodal_embedder(embedding_model, http)

    if provider == "ollama":
        # Use model-level base_url if set, otherwise fall back to providers config
        base_url = embedding_model.base_url or config.providers.ollama.base_url
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        client = build_http_client(http)
        model = OpenAIEmbeddingModel(
            model_name,
            provider=OllamaProvider(base_url=base_url, http_client=client),
        )
        return EmbedderWrapper(Embedder(model), vector_dim, owned_http_client=client)

    if provider == "openai":
        client = build_http_client(http)
        provider_kwargs: dict[str, Any] = {"http_client": client}
        if embedding_model.base_url:
            provider_kwargs["base_url"] = embedding_model.base_url
        model = OpenAIEmbeddingModel(
            model_name,
            provider=OpenAIProvider(**provider_kwargs),
        )
        return EmbedderWrapper(Embedder(model), vector_dim, owned_http_client=client)

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
            model_name, vector_dim, base_url=base_url, http=http, supports_images=False
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


def _vllm_base_url(base_url: str | None) -> str:
    base_url = base_url or "http://localhost:8000/v1"
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return base_url


def _get_multimodal_embedder(
    embedding_model: "EmbeddingModelConfig",
    http: "EmbeddingHTTPConfig",
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
            model_name, vector_dim, base_url=base_url, http=http, supports_images=True
        )

    if provider == "voyageai":
        from haiku.rag.embeddings.voyageai import VoyageMultimodalEmbedder

        return VoyageMultimodalEmbedder(model_name, vector_dim)

    if provider == "cohere":
        from haiku.rag.embeddings.cohere import CohereMultimodalEmbedder

        return CohereMultimodalEmbedder(model_name, vector_dim)

    raise ValueError(
        f"Provider '{provider}' does not support multimodal embedding. Set "
        "embeddings.model.multimodal: true on a vllm, voyageai, or cohere model."
    )

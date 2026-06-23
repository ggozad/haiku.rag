from pathlib import Path

import numpy as np
import pytest

from haiku.rag.config import (
    AppConfig,
    Config,
    EmbeddingModelConfig,
    EmbeddingsConfig,
)
from haiku.rag.embeddings import (
    EmbedderWrapper,
    contextualize,
    embed_chunks,
    get_embedder,
)
from haiku.rag.store.models.chunk import Chunk


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent / "cassettes" / "test_embedder")


def similarities(embeddings, test_embedding):
    """Calculate cosine similarity between embeddings and a test embedding."""
    return [
        np.dot(embedding, test_embedding)
        / (np.linalg.norm(embedding) * np.linalg.norm(test_embedding))
        for embedding in embeddings
    ]


@pytest.mark.vcr()
async def test_ollama_embedder(allow_model_requests):
    """Test Ollama embedder via pydantic-ai."""
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="ollama", name="mxbai-embed-large", vector_dim=1024
            )
        )
    )
    embedder = get_embedder(config)
    phrases = [
        "I enjoy eating great food.",
        "Python is my favorite programming language.",
        "I love to travel and see new places.",
    ]

    # Test batch embedding (documents)
    embeddings = await embedder.embed_documents(phrases)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert all(isinstance(emb, list) for emb in embeddings)
    embeddings = [np.array(emb) for emb in embeddings]

    # Test query embedding
    test_phrase = "I am going for a camping trip."
    test_embedding = await embedder.embed_query(test_phrase)
    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[2]

    test_phrase = "When is dinner ready?"
    test_embedding = await embedder.embed_query(test_phrase)
    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[0]

    test_phrase = "I work as a software developer."
    test_embedding = await embedder.embed_query(test_phrase)
    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[1]


def test_contextualize_with_headings():
    """Test that contextualize prepends headings to chunk content."""
    chunks = [
        Chunk(
            content="This is the content.",
            metadata={"headings": ["Chapter 1", "Section 1.1"]},
        ),
        Chunk(
            content="More content here.",
            metadata={"headings": ["Chapter 2"]},
        ),
    ]

    texts = contextualize(chunks)

    assert len(texts) == 2
    assert texts[0] == "Chapter 1\nSection 1.1\nThis is the content."
    assert texts[1] == "Chapter 2\nMore content here."


def test_contextualize_without_headings():
    """Test that contextualize returns raw content when no headings."""
    chunks = [
        Chunk(content="Plain content."),
        Chunk(content="Another chunk.", metadata={}),
        Chunk(content="With empty headings.", metadata={"headings": None}),
    ]

    texts = contextualize(chunks)

    assert len(texts) == 3
    assert texts[0] == "Plain content."
    assert texts[1] == "Another chunk."
    assert texts[2] == "With empty headings."


def test_contextualize_empty_list():
    """Test that contextualize handles empty list."""
    texts = contextualize([])
    assert texts == []


class _StubEmbedder(EmbedderWrapper):
    def __init__(self):
        super().__init__(embedder=None, vector_dim=8)
        self.doc_batches = 0

    async def embed_documents(self, texts):
        self.doc_batches += 1
        return [[0.1] * 8 for _ in texts]


async def test_embed_chunks_uses_provided_embedder(monkeypatch):
    """embed_chunks embeds via the embedder it is given and never builds one."""
    import haiku.rag.embeddings as embeddings_mod

    def fail(*args, **kwargs):
        raise AssertionError("embed_chunks must not build its own embedder")

    monkeypatch.setattr(embeddings_mod, "get_embedder", fail)

    embedder = _StubEmbedder()
    chunks = [
        Chunk(id="a", content="alpha", order=0),
        Chunk(id="b", content="beta", order=1),
    ]

    embedded = await embed_chunks(chunks, embedder, AppConfig())

    assert [c.embedding for c in embedded] == [[0.1] * 8, [0.1] * 8]
    assert embedder.doc_batches == 1


async def test_client_embedder_is_store_embedder(temp_db_path):
    """The client exposes the Store's cached embedder rather than its own."""
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(temp_db_path, create=True) as client:
        assert client.embedder is client.store.embedder


@pytest.mark.vcr()
async def test_embed_chunks_basic(allow_model_requests):
    """Test that embed_chunks generates embeddings for chunks."""
    chunks = [
        Chunk(
            id="chunk1",
            document_id="doc1",
            content="I enjoy eating great food.",
            metadata={"headings": ["Food"]},
            order=0,
        ),
        Chunk(
            id="chunk2",
            document_id="doc1",
            content="Python is my favorite programming language.",
            metadata={"headings": ["Programming"]},
            order=1,
        ),
    ]

    embedded_chunks = await embed_chunks(chunks, get_embedder(Config))

    assert len(embedded_chunks) == 2
    # Check that all original fields are preserved
    assert embedded_chunks[0].id == "chunk1"
    assert embedded_chunks[0].document_id == "doc1"
    assert embedded_chunks[0].content == "I enjoy eating great food."
    assert embedded_chunks[0].metadata == {"headings": ["Food"]}
    assert embedded_chunks[0].order == 0
    # Check that embeddings are generated
    assert embedded_chunks[0].embedding is not None
    assert len(embedded_chunks[0].embedding) > 0
    assert embedded_chunks[1].embedding is not None


@pytest.mark.vcr()
async def test_embed_chunks_returns_new_objects(allow_model_requests):
    """Test that embed_chunks returns new Chunk objects (immutable pattern)."""
    original = Chunk(id="orig", content="Test content.")
    embedded = await embed_chunks([original], get_embedder(Config))

    # Original should be unchanged
    assert original.embedding is None
    # New chunk should have embedding
    assert embedded[0].embedding is not None
    # They should be different objects
    assert embedded[0] is not original


async def test_embed_chunks_empty_list():
    """Test that embed_chunks handles empty list."""
    result = await embed_chunks([], _StubEmbedder())
    assert result == []


async def test_embed_chunks_picture_with_text_only_embedder_raises():
    """A picture chunk fed through a text-only embedder must surface a
    clear error, not silently drop the chunk or call ``embed_image``
    on something that doesn't support it."""
    chunk = Chunk(id="pic", content="x")
    chunk._picture_data = b"\x89PNG\r\n\x1a\nfake"

    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="ollama", name="qwen3-embedding:4b", vector_dim=2560
            )
        )
    )

    with pytest.raises(ValueError, match="multimodal embedder"):
        await embed_chunks([chunk], get_embedder(config), config)


async def test_embed_chunks_respects_configured_batch_size(monkeypatch):
    """`embeddings.batch_size` controls how `embed_chunks` slices its input.

    Voyage models cap total tokens per /embeddings call (120K for
    voyage-3-large and friends). Exposing the slice size lets users tune
    it down without dropping `chunk_size` and harming retrieval quality.
    """
    from haiku.rag.config import AppConfig
    from haiku.rag.embeddings import EmbedderWrapper

    call_sizes: list[int] = []

    async def tracking_embed(self, texts):
        call_sizes.append(len(texts))
        return [[0.1] * 10 for _ in texts]

    monkeypatch.setattr(EmbedderWrapper, "embed_documents", tracking_embed)

    config = AppConfig()
    config.embeddings.batch_size = 7

    num_chunks = 20
    chunks = [
        Chunk(id=f"chunk-{i}", content=f"Content {i}", order=i)
        for i in range(num_chunks)
    ]

    result = await embed_chunks(chunks, get_embedder(config), config)

    assert len(result) == num_chunks
    # 20 chunks / 7 per batch -> 7, 7, 6
    assert call_sizes == [7, 7, 6]
    assert result[0].id == "chunk-0"
    assert result[-1].id == f"chunk-{num_chunks - 1}"
    assert all(r.embedding == [0.1] * 10 for r in result)


@pytest.mark.vcr()
async def test_embed_chunks_preserves_all_fields(allow_model_requests):
    """Test that embed_chunks preserves all chunk fields."""
    chunk = Chunk(
        id="test-id",
        document_id="doc-id",
        content="Test content",
        metadata={"key": "value", "headings": ["Heading"]},
        order=5,
        document_uri="https://example.com/doc",
        document_title="Test Document",
        document_meta={"author": "Test"},
    )

    embedded = await embed_chunks([chunk], get_embedder(Config))

    assert embedded[0].id == "test-id"
    assert embedded[0].document_id == "doc-id"
    assert embedded[0].content == "Test content"
    assert embedded[0].metadata == {"key": "value", "headings": ["Heading"]}
    assert embedded[0].order == 5
    assert embedded[0].document_uri == "https://example.com/doc"
    assert embedded[0].document_title == "Test Document"
    assert embedded[0].document_meta == {"author": "Test"}
    assert embedded[0].embedding is not None


# Multimodal embedder support


def _ollama_text_only_config():
    return AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="ollama", name="mxbai-embed-large", vector_dim=1024
            )
        )
    )


async def test_text_only_embedder_does_not_support_images():
    embedder = get_embedder(_ollama_text_only_config())
    assert embedder.supports_images is False
    with pytest.raises(NotImplementedError, match="multimodal"):
        await embedder.embed_image(b"\x89PNG\r\n\x1a\n")


async def test_vllm_embed_text_request_shape(monkeypatch):
    """vLLM text embedding posts a standard OpenAI ``input`` array (real
    server-side batching), not the ``messages`` superset (which is reserved
    for image inputs)."""
    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json, headers):
            captured["url"] = url
            captured["body"] = json
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    embedder = VLLMMultimodalEmbedder(
        model_name="Qwen/Qwen3-VL-Embedding-2B",
        vector_dim=2048,
        base_url="http://localhost:8000/v1",
    )
    vecs = await embedder.embed_documents(["a photo of a cat", "a sleeping dog"])
    assert vecs == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert captured["url"] == "http://localhost:8000/v1/embeddings"
    body = captured["body"]
    assert body["model"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert body["input"] == ["a photo of a cat", "a sleeping dog"]
    assert "messages" not in body
    assert body["encoding_format"] == "float"


async def test_vllm_embed_image_request_shape(monkeypatch):
    """vLLM image embedding posts an `image_url` content part with a data: URI."""
    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [0.4] * 4}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json, headers):
            captured["body"] = json
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    embedder = VLLMMultimodalEmbedder(
        model_name="some-model",
        vector_dim=4,
        base_url="http://localhost:8000/v1",
    )
    raw = b"\x89PNG\r\n\x1a\nfake"
    vec = await embedder.embed_image(raw)
    assert vec == [0.4, 0.4, 0.4, 0.4]
    content = captured["body"]["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "image_url"
    url = content[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


async def test_vllm_supports_images_flag():
    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    embedder = VLLMMultimodalEmbedder(
        model_name="x", vector_dim=2, base_url="http://localhost:8000/v1"
    )
    assert embedder.supports_images is True


async def test_vllm_text_only_embed_image_raises():
    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    embedder = VLLMMultimodalEmbedder(
        model_name="x",
        vector_dim=2,
        base_url="http://localhost:8000/v1",
        supports_images=False,
    )
    assert embedder.supports_images is False
    with pytest.raises(NotImplementedError, match="text-only"):
        await embedder.embed_image(b"\x89PNG\r\n\x1a\n")


async def test_vllm_connect_error_surfaces_helpful_message(monkeypatch):
    import httpx

    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("All connection attempts failed")

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    embedder = VLLMMultimodalEmbedder(
        model_name="x", vector_dim=2, base_url="http://nope:8000/v1"
    )
    with pytest.raises(ValueError, match="Could not connect to vLLM"):
        await embedder.embed_query("hi")


async def test_vllm_timeout_surfaces_helpful_message(monkeypatch):
    import httpx

    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    embedder = VLLMMultimodalEmbedder(
        model_name="x", vector_dim=2, base_url="http://localhost:8000/v1"
    )
    with pytest.raises(ValueError, match="timed out"):
        await embedder.embed_query("hi")


async def test_vllm_401_surfaces_auth_error(monkeypatch):
    import httpx

    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "401",
                request=httpx.Request("POST", "http://x"),
                response=self,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            )

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    embedder = VLLMMultimodalEmbedder(
        model_name="x", vector_dim=2, base_url="http://localhost:8000/v1", api_key="bad"
    )
    with pytest.raises(ValueError, match="Authentication failed"):
        await embedder.embed_query("hi")


async def test_vllm_other_http_error_surfaces(monkeypatch):
    import httpx

    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    class FakeResponse:
        status_code = 500

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "500",
                request=httpx.Request("POST", "http://x"),
                response=self,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            )

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    embedder = VLLMMultimodalEmbedder(
        model_name="x", vector_dim=2, base_url="http://localhost:8000/v1"
    )
    with pytest.raises(ValueError, match="HTTP error from vLLM"):
        await embedder.embed_query("hi")


async def test_vllm_empty_data_response_raises(monkeypatch):
    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    embedder = VLLMMultimodalEmbedder(
        model_name="x", vector_dim=2, base_url="http://localhost:8000/v1"
    )
    with pytest.raises(ValueError, match="returned no embeddings"):
        await embedder.embed_query("hi")


async def test_vllm_embed_documents_empty_list_skips_request(monkeypatch):
    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    called = False

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr("httpx.AsyncClient", FakeAsyncClient)

    embedder = VLLMMultimodalEmbedder(
        model_name="x", vector_dim=2, base_url="http://localhost:8000/v1"
    )
    assert await embedder.embed_documents([]) == []
    assert called is False


async def test_vllm_pil_image_roundtrips_to_data_uri():
    from PIL import Image

    from haiku.rag.embeddings.vllm import _to_data_uri

    img = Image.new("RGB", (4, 4), color="red")
    uri = _to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")


async def test_vllm_to_data_uri_rejects_unsupported():
    from haiku.rag.embeddings.vllm import _to_data_uri

    with pytest.raises(TypeError):
        _to_data_uri("not bytes")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


async def test_vllm_get_embedder_routes_to_multimodal():
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="vllm",
                name="Qwen/Qwen3-VL-Embedding-2B",
                vector_dim=2048,
                base_url="http://my-vllm:8000/v1",
                multimodal=True,
            )
        )
    )
    embedder = get_embedder(config)
    assert embedder.supports_images is True
    assert embedder._base_url == "http://my-vllm:8000/v1"  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]


def test_multimodal_defaults_to_false():
    model = EmbeddingModelConfig(provider="vllm", name="x", vector_dim=2)
    assert model.multimodal is False
    assert "multimodal" in model.model_dump()


async def test_vllm_text_only_when_multimodal_unset():
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="vllm",
                name="qwen3-embedding:4b",
                vector_dim=2560,
                base_url="http://my-vllm:8000/v1",
            )
        )
    )
    embedder = get_embedder(config)
    assert embedder.supports_images is False


@pytest.mark.parametrize("provider", ["ollama", "openai", "sentence-transformers"])
async def test_multimodal_unsupported_provider_raises(provider):
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider=provider, name="x", vector_dim=2, multimodal=True
            )
        )
    )
    with pytest.raises(ValueError, match="does not support multimodal"):
        get_embedder(config)


@pytest.mark.vcr()
async def test_vllm_embed_text_and_image_end_to_end():
    """End-to-end against a real vLLM ``/v1/embeddings`` server: confirm
    both the text (``input`` array) and image (``messages`` with
    ``image_url``) shapes return embeddings of the configured dimension
    in the same vector space.

    Recorded against ``Qwen/Qwen3-VL-Embedding-8B`` (4096-dim) served by
    a real vLLM build (the multimodal ``messages``-with-``image_url``
    superset on ``/v1/embeddings`` is a real-vLLM feature, not currently
    available in vllm-mlx). To re-record, point port 8000 at such a vLLM
    and run with ``--record-mode=rewrite``."""
    from PIL import Image

    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    embedder = VLLMMultimodalEmbedder(
        model_name="qwen3-embedding-v-8b",
        vector_dim=4096,
        base_url="http://localhost:8000/v1",
    )

    text_vec = await embedder.embed_query("a photo of a red square")
    assert len(text_vec) == 4096
    assert any(abs(x) > 1e-6 for x in text_vec), "text embedding is all zeros"

    text_batch = await embedder.embed_documents(["hello world", "another doc"])
    assert len(text_batch) == 2
    assert all(len(v) == 4096 for v in text_batch)

    image = Image.new("RGB", (64, 64), color=(255, 0, 0))
    image_vec = await embedder.embed_image(image)
    assert len(image_vec) == 4096
    assert any(abs(x) > 1e-6 for x in image_vec), "image embedding is all zeros"


class _FakeVoyageResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


def _fake_voyage_client(captured, embeddings):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["init"] = kwargs

        async def multimodal_embed(self, **kwargs):
            captured.update(kwargs)
            return _FakeVoyageResult(embeddings)

    return FakeAsyncClient


async def test_voyage_embed_documents_request_shape(monkeypatch):
    from haiku.rag.embeddings.voyageai import VoyageMultimodalEmbedder

    captured: dict = {}
    monkeypatch.setattr(
        "voyageai.AsyncClient",
        _fake_voyage_client(captured, [[0.1, 0.2], [0.3, 0.4]]),
    )

    embedder = VoyageMultimodalEmbedder("voyage-multimodal-3", vector_dim=2)
    vecs = await embedder.embed_documents(["a cat", "a dog"])

    assert vecs == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["model"] == "voyage-multimodal-3"
    assert captured["input_type"] == "document"
    assert captured["inputs"] == [["a cat"], ["a dog"]]
    assert captured["output_dimension"] == 2


async def test_voyage_embed_query_request_shape(monkeypatch):
    from haiku.rag.embeddings.voyageai import VoyageMultimodalEmbedder

    captured: dict = {}
    monkeypatch.setattr(
        "voyageai.AsyncClient", _fake_voyage_client(captured, [[0.5, 0.6]])
    )

    embedder = VoyageMultimodalEmbedder("voyage-multimodal-3", vector_dim=2)
    vec = await embedder.embed_query("find the cat")

    assert vec == [0.5, 0.6]
    assert captured["model"] == "voyage-multimodal-3"
    assert captured["input_type"] == "query"
    assert captured["inputs"] == [["find the cat"]]


async def test_voyage_embed_image_passes_pil(monkeypatch):
    from PIL import Image

    from haiku.rag.embeddings.voyageai import VoyageMultimodalEmbedder

    captured: dict = {}
    monkeypatch.setattr(
        "voyageai.AsyncClient", _fake_voyage_client(captured, [[0.7, 0.8]])
    )

    embedder = VoyageMultimodalEmbedder("voyage-multimodal-3", vector_dim=2)
    import io

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format="PNG")
    vec = await embedder.embed_image(buf.getvalue())

    assert vec == [0.7, 0.8]
    assert captured["model"] == "voyage-multimodal-3"
    inputs = captured["inputs"]
    assert len(inputs) == 1 and len(inputs[0]) == 1
    assert isinstance(inputs[0][0], Image.Image)
    assert captured["input_type"] == "document"


async def test_voyage_embed_documents_empty_list_skips_request(monkeypatch):
    from haiku.rag.embeddings.voyageai import VoyageMultimodalEmbedder

    captured: dict = {}
    monkeypatch.setattr("voyageai.AsyncClient", _fake_voyage_client(captured, []))

    embedder = VoyageMultimodalEmbedder("voyage-multimodal-3", vector_dim=2)
    assert await embedder.embed_documents([]) == []
    assert "inputs" not in captured


async def test_voyage_get_embedder_routes_to_multimodal(monkeypatch):
    monkeypatch.setattr("voyageai.AsyncClient", _fake_voyage_client({}, []))
    from haiku.rag.embeddings.voyageai import VoyageMultimodalEmbedder

    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="voyageai",
                name="voyage-multimodal-3",
                vector_dim=1024,
                multimodal=True,
            )
        )
    )
    embedder = get_embedder(config)
    assert isinstance(embedder, VoyageMultimodalEmbedder)
    assert embedder.supports_images is True


@pytest.mark.vcr()
async def test_voyage_embed_text_and_image_end_to_end():
    """End-to-end against the real VoyageAI ``multimodal_embed`` API: text and
    image inputs return embeddings of the configured dimension in a shared
    vector space. To re-record, set ``VOYAGE_API_KEY`` and run with
    ``--record-mode=rewrite``."""
    from PIL import Image

    from haiku.rag.embeddings.voyageai import VoyageMultimodalEmbedder

    embedder = VoyageMultimodalEmbedder("voyage-multimodal-3", vector_dim=1024)

    text_vec = await embedder.embed_query("a photo of a red square")
    assert len(text_vec) == 1024
    assert any(abs(x) > 1e-6 for x in text_vec), "text embedding is all zeros"

    text_batch = await embedder.embed_documents(["hello world", "another doc"])
    assert len(text_batch) == 2
    assert all(len(v) == 1024 for v in text_batch)

    image_vec = await embedder.embed_image(Image.new("RGB", (64, 64), (255, 0, 0)))
    assert len(image_vec) == 1024
    assert any(abs(x) > 1e-6 for x in image_vec), "image embedding is all zeros"


class _FakeCohereEmbeddings:
    def __init__(self, float_):
        self.float_ = float_


class _FakeCohereResult:
    def __init__(self, float_):
        self.embeddings = _FakeCohereEmbeddings(float_)


def _fake_cohere_client(captured, float_):
    class FakeAsyncClientV2:
        def __init__(self, *args, **kwargs):
            captured["init"] = kwargs

        async def embed(self, **kwargs):
            captured.update(kwargs)
            return _FakeCohereResult(float_)

    return FakeAsyncClientV2


async def test_cohere_embed_documents_request_shape(monkeypatch):
    from haiku.rag.embeddings.cohere import CohereMultimodalEmbedder

    captured: dict = {}
    monkeypatch.setattr(
        "cohere.AsyncClientV2", _fake_cohere_client(captured, [[0.1, 0.2], [0.3, 0.4]])
    )

    embedder = CohereMultimodalEmbedder("embed-v4.0", vector_dim=2)
    vecs = await embedder.embed_documents(["a cat", "a dog"])

    assert vecs == [[0.1, 0.2], [0.3, 0.4]]
    assert captured["model"] == "embed-v4.0"
    assert captured["input_type"] == "search_document"
    assert captured["texts"] == ["a cat", "a dog"]
    assert captured["output_dimension"] == 2
    assert captured["embedding_types"] == ["float"]


async def test_cohere_embed_query_request_shape(monkeypatch):
    from haiku.rag.embeddings.cohere import CohereMultimodalEmbedder

    captured: dict = {}
    monkeypatch.setattr(
        "cohere.AsyncClientV2", _fake_cohere_client(captured, [[0.5, 0.6]])
    )

    embedder = CohereMultimodalEmbedder("embed-v4.0", vector_dim=2)
    vec = await embedder.embed_query("find the cat")

    assert vec == [0.5, 0.6]
    assert captured["model"] == "embed-v4.0"
    assert captured["input_type"] == "search_query"
    assert captured["texts"] == ["find the cat"]


async def test_cohere_embed_image_uses_image_input_type(monkeypatch):
    from haiku.rag.embeddings.cohere import CohereMultimodalEmbedder

    captured: dict = {}
    monkeypatch.setattr(
        "cohere.AsyncClientV2", _fake_cohere_client(captured, [[0.7, 0.8]])
    )

    embedder = CohereMultimodalEmbedder("embed-v4.0", vector_dim=2)
    vec = await embedder.embed_image(b"\x89PNG\r\n\x1a\nfake")

    assert vec == [0.7, 0.8]
    assert captured["model"] == "embed-v4.0"
    assert captured["input_type"] == "image"
    images = captured["images"]
    assert len(images) == 1
    assert images[0].startswith("data:image/png;base64,")


async def test_cohere_embed_documents_empty_list_skips_request(monkeypatch):
    from haiku.rag.embeddings.cohere import CohereMultimodalEmbedder

    captured: dict = {}
    monkeypatch.setattr("cohere.AsyncClientV2", _fake_cohere_client(captured, []))

    embedder = CohereMultimodalEmbedder("embed-v4.0", vector_dim=2)
    assert await embedder.embed_documents([]) == []
    assert "texts" not in captured


async def test_cohere_get_embedder_routes_to_multimodal(monkeypatch):
    monkeypatch.setattr("cohere.AsyncClientV2", _fake_cohere_client({}, []))
    from haiku.rag.embeddings.cohere import CohereMultimodalEmbedder

    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="cohere",
                name="embed-v4.0",
                vector_dim=1536,
                multimodal=True,
            )
        )
    )
    embedder = get_embedder(config)
    assert isinstance(embedder, CohereMultimodalEmbedder)
    assert embedder.supports_images is True


@pytest.mark.vcr()
async def test_cohere_embed_text_and_image_end_to_end():
    """End-to-end against the real Cohere ``embed`` API (``embed-v4.0``): text
    and image inputs return embeddings of the configured dimension in a shared
    vector space. To re-record, set ``CO_API_KEY`` and run with
    ``--record-mode=rewrite``."""
    from PIL import Image

    from haiku.rag.embeddings.cohere import CohereMultimodalEmbedder

    embedder = CohereMultimodalEmbedder("embed-v4.0", vector_dim=1536)

    text_vec = await embedder.embed_query("a photo of a red square")
    assert len(text_vec) == 1536
    assert any(abs(x) > 1e-6 for x in text_vec), "text embedding is all zeros"

    text_batch = await embedder.embed_documents(["hello world", "another doc"])
    assert len(text_batch) == 2
    assert all(len(v) == 1536 for v in text_batch)

    image_vec = await embedder.embed_image(Image.new("RGB", (64, 64), (255, 0, 0)))
    assert len(image_vec) == 1536
    assert any(abs(x) > 1e-6 for x in image_vec), "image embedding is all zeros"

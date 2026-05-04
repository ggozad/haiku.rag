from pathlib import Path

import numpy as np
import pytest

from haiku.rag.config import AppConfig, EmbeddingModelConfig, EmbeddingsConfig
from haiku.rag.embeddings import contextualize, embed_chunks, get_embedder
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

    embedded_chunks = await embed_chunks(chunks)

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
    embedded = await embed_chunks([original])

    # Original should be unchanged
    assert original.embedding is None
    # New chunk should have embedding
    assert embedded[0].embedding is not None
    # They should be different objects
    assert embedded[0] is not original


async def test_embed_chunks_empty_list():
    """Test that embed_chunks handles empty list."""
    result = await embed_chunks([])
    assert result == []


async def test_embed_chunks_batches_large_inputs(monkeypatch):
    """Test that embed_chunks batches calls when chunk count exceeds batch size."""
    from haiku.rag.embeddings import EMBEDDING_BATCH_SIZE, EmbedderWrapper

    call_sizes: list[int] = []

    async def tracking_embed(self, texts):
        call_sizes.append(len(texts))
        return [[0.1] * 10 for _ in texts]

    monkeypatch.setattr(EmbedderWrapper, "embed_documents", tracking_embed)

    # Create more chunks than one batch
    num_chunks = EMBEDDING_BATCH_SIZE + 100
    chunks = [
        Chunk(id=f"chunk-{i}", content=f"Content {i}", order=i)
        for i in range(num_chunks)
    ]

    result = await embed_chunks(chunks)

    assert len(result) == num_chunks
    assert len(call_sizes) == 2
    assert call_sizes[0] == EMBEDDING_BATCH_SIZE
    assert call_sizes[1] == 100
    # Verify order is preserved
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

    embedded = await embed_chunks([chunk])

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
    with pytest.raises(NotImplementedError, match="multimodal provider"):
        await embedder.embed_image_query(b"\x89PNG\r\n\x1a\n")


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
    vec = await embedder.embed_image_query(raw)
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


async def test_vllm_get_embedder_routes_to_multimodal():
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="vllm",
                name="Qwen/Qwen3-VL-Embedding-2B",
                vector_dim=2048,
                base_url="http://my-vllm:8000/v1",
            )
        )
    )
    embedder = get_embedder(config)
    assert embedder.supports_images is True
    assert embedder._base_url == "http://my-vllm:8000/v1"  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]


@pytest.mark.integration
async def test_vllm_embed_text_and_image_end_to_end():
    """Hit a real vLLM ``/v1/embeddings`` server and confirm both the text
    (``input`` array) and image (``messages`` with ``image_url``) shapes
    return embeddings of the configured dimension in the same vector space.

    Configure via env vars:
      HAIKU_RAG_VLLM_BASE_URL  (default http://localhost:8000/v1)
      HAIKU_RAG_VLLM_MODEL     (default Qwen/Qwen3-VL-Embedding-8B)
      HAIKU_RAG_VLLM_VECTOR_DIM (default 4096)

    Run a server first, e.g.:
        vllm serve Qwen/Qwen3-VL-Embedding-8B \\
            --runner pooling --dtype bfloat16 --trust-remote-code
    """
    import os

    from PIL import Image

    from haiku.rag.embeddings.vllm import VLLMMultimodalEmbedder

    base_url = os.environ.get("HAIKU_RAG_VLLM_BASE_URL", "http://localhost:8000/v1")
    model_name = os.environ.get("HAIKU_RAG_VLLM_MODEL", "Qwen/Qwen3-VL-Embedding-8B")
    vector_dim = int(os.environ.get("HAIKU_RAG_VLLM_VECTOR_DIM", "4096"))

    embedder = VLLMMultimodalEmbedder(
        model_name=model_name,
        vector_dim=vector_dim,
        base_url=base_url,
    )

    text_vec = await embedder.embed_query("a photo of a red square")
    assert len(text_vec) == vector_dim
    assert any(abs(x) > 1e-6 for x in text_vec), "text embedding is all zeros"

    text_batch = await embedder.embed_documents(["hello world", "another doc"])
    assert len(text_batch) == 2
    assert all(len(v) == vector_dim for v in text_batch)

    image = Image.new("RGB", (64, 64), color=(255, 0, 0))
    image_vec = await embedder.embed_image_query(image)
    assert len(image_vec) == vector_dim
    assert any(abs(x) > 1e-6 for x in image_vec), "image embedding is all zeros"

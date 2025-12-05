import os

import numpy as np
import pytest

from haiku.rag.config import Config
from haiku.rag.embeddings import contextualize, embed_chunks
from haiku.rag.embeddings.ollama import Embedder as OllamaEmbedder
from haiku.rag.embeddings.openai import Embedder as OpenAIEmbedder
from haiku.rag.embeddings.vllm import Embedder as VLLMEmbedder
from haiku.rag.store.models.chunk import Chunk

OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
VOYAGEAI_AVAILABLE = bool(os.getenv("VOYAGE_API_KEY"))
VLLM_EMBEDDINGS_AVAILABLE = bool(Config.providers.vllm.embeddings_base_url)


# Calculate cosine similarity
def similarities(embeddings, test_embedding):
    return [
        np.dot(embedding, test_embedding)
        / (np.linalg.norm(embedding) * np.linalg.norm(test_embedding))
        for embedding in embeddings
    ]


@pytest.mark.asyncio
async def test_ollama_embedder():
    embedder = OllamaEmbedder("mxbai-embed-large", 1024)
    phrases = [
        "I enjoy eating great food.",
        "Python is my favorite programming language.",
        "I love to travel and see new places.",
    ]

    # Test batch embedding
    embeddings = await embedder.embed(phrases)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert all(isinstance(emb, list) for emb in embeddings)
    embeddings = [np.array(emb) for emb in embeddings]

    test_phrase = "I am going for a camping trip."
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[2]

    test_phrase = "When is dinner ready?"
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[0]

    test_phrase = "I work as a software developer."
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[1]


@pytest.mark.asyncio
@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI API key not available")
async def test_openai_embedder():
    embedder = OpenAIEmbedder("text-embedding-3-small", 1536)
    phrases = [
        "I enjoy eating great food.",
        "Python is my favorite programming language.",
        "I love to travel and see new places.",
    ]

    # Test batch embedding
    embeddings = await embedder.embed(phrases)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert all(isinstance(emb, list) for emb in embeddings)
    embeddings = [np.array(emb) for emb in embeddings]

    test_phrase = "I am going for a camping trip."
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[2]

    test_phrase = "When is dinner ready?"
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[0]

    test_phrase = "I work as a software developer."
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[1]


@pytest.mark.asyncio
@pytest.mark.skipif(not VOYAGEAI_AVAILABLE, reason="VoyageAI API key not available")
async def test_voyageai_embedder():
    try:
        from haiku.rag.embeddings.voyageai import Embedder as VoyageAIEmbedder

        embedder = VoyageAIEmbedder("voyage-3.5", 1024)
        phrases = [
            "I enjoy eating great food.",
            "Python is my favorite programming language.",
            "I love to travel and see new places.",
        ]

        # Test batch embedding
        embeddings = await embedder.embed(phrases)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        embeddings = [np.array(emb) for emb in embeddings]

        test_phrase = "I am going for a camping trip."
        test_embedding = await embedder.embed(test_phrase)

        sims = similarities(embeddings, test_embedding)
        assert max(sims) == sims[2]

        test_phrase = "When is dinner ready?"
        test_embedding = await embedder.embed(test_phrase)

        sims = similarities(embeddings, test_embedding)
        assert max(sims) == sims[0]

        test_phrase = "I work as a software developer."
        test_embedding = await embedder.embed(test_phrase)

        sims = similarities(embeddings, test_embedding)
        assert max(sims) == sims[1]

    except ImportError:
        pytest.skip("VoyageAI package not installed")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not VLLM_EMBEDDINGS_AVAILABLE, reason="vLLM embeddings server not configured"
)
async def test_vllm_embedder():
    embedder = VLLMEmbedder("mixedbread-ai/mxbai-embed-large-v1", 512)
    phrases = [
        "I enjoy eating great food.",
        "Python is my favorite programming language.",
        "I love to travel and see new places.",
    ]

    # Test batch embedding
    embeddings = await embedder.embed(phrases)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert all(isinstance(emb, list) for emb in embeddings)
    embeddings = [np.array(emb) for emb in embeddings]

    test_phrase = "I am going for a camping trip."
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[2]

    test_phrase = "When is dinner ready?"
    test_embedding = await embedder.embed(test_phrase)

    sims = similarities(embeddings, test_embedding)
    assert max(sims) == sims[0]

    test_phrase = "I work as a software developer."
    test_embedding = await embedder.embed(test_phrase)

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


@pytest.mark.asyncio
async def test_embed_chunks_basic():
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


@pytest.mark.asyncio
async def test_embed_chunks_returns_new_objects():
    """Test that embed_chunks returns new Chunk objects (immutable pattern)."""
    original = Chunk(id="orig", content="Test content.")
    embedded = await embed_chunks([original])

    # Original should be unchanged
    assert original.embedding is None
    # New chunk should have embedding
    assert embedded[0].embedding is not None
    # They should be different objects
    assert embedded[0] is not original


@pytest.mark.asyncio
async def test_embed_chunks_empty_list():
    """Test that embed_chunks handles empty list."""
    result = await embed_chunks([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_chunks_preserves_all_fields():
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

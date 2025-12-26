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


@pytest.mark.vcr()
async def test_openai_embedder(allow_model_requests):
    """Test OpenAI embedder via pydantic-ai."""
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="openai", name="text-embedding-3-small", vector_dim=1536
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


@pytest.mark.vcr()
async def test_voyageai_embedder(allow_model_requests):
    """Test VoyageAI embedder."""
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="voyageai", name="voyage-3.5", vector_dim=1024
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


@pytest.mark.vcr()
async def test_embed_chunks_empty_list(allow_model_requests):
    """Test that embed_chunks handles empty list."""
    result = await embed_chunks([])
    assert result == []


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

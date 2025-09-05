import numpy as np
import pytest

from haiku.rag.config import Config
from haiku.rag.embeddings.ollama import Embedder as OllamaEmbedder
from haiku.rag.embeddings.openai import Embedder as OpenAIEmbedder
from haiku.rag.embeddings.vllm import Embedder as VLLMEmbedder

OPENAI_AVAILABLE = bool(Config.OPENAI_API_KEY)
VOYAGEAI_AVAILABLE = bool(Config.VOYAGE_API_KEY)
VLLM_EMBEDDINGS_AVAILABLE = bool(Config.VLLM_EMBEDDINGS_BASE_URL)


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

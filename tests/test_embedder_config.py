import pytest

from haiku.rag.config import (
    AppConfig,
    EmbeddingModelConfig,
    EmbeddingsConfig,
    OllamaConfig,
    ProvidersConfig,
)
from haiku.rag.embeddings import get_embedder


def test_ollama_embedder_uses_config():
    """Test that Ollama embedder uses the config passed to get_embedder."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="ollama", name="custom-model", vector_dim=512
            ),
        ),
        providers=ProvidersConfig(
            ollama=OllamaConfig(base_url="http://custom-ollama:8080"),
        ),
    )

    embedder = get_embedder(custom_config)

    assert embedder._vector_dim == 512


def test_openai_embedder_uses_config():
    """Test that OpenAI embedder uses the config passed to get_embedder."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="openai", name="text-embedding-3-large", vector_dim=3072
            ),
        ),
    )

    embedder = get_embedder(custom_config)

    assert embedder._vector_dim == 3072


def test_openai_embedder_with_base_url():
    """Test that OpenAI embedder uses custom base_url for vLLM/LM Studio."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="openai",
                name="some-local-model",
                vector_dim=768,
                base_url="http://localhost:8000/v1",
            ),
        ),
    )

    embedder = get_embedder(custom_config)

    assert embedder._vector_dim == 768


def test_cohere_embedder_uses_config():
    """Test that Cohere embedder uses the config passed to get_embedder."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="cohere", name="embed-v4.0", vector_dim=1024
            ),
        ),
    )

    embedder = get_embedder(custom_config)

    assert embedder._vector_dim == 1024


def test_sentence_transformers_embedder_uses_config():
    """Test that SentenceTransformers embedder uses the config."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="sentence-transformers",
                name="all-MiniLM-L6-v2",
                vector_dim=384,
            ),
        ),
    )

    embedder = get_embedder(custom_config)

    assert embedder._vector_dim == 384


def test_unsupported_provider_raises():
    """Test that unsupported provider raises ValueError."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="unsupported-provider", name="model", vector_dim=512
            ),
        ),
    )

    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embedder(custom_config)

import pytest

from haiku.rag.config import (
    AppConfig,
    EmbeddingsConfig,
    OllamaConfig,
    ProvidersConfig,
    VLLMConfig,
)
from haiku.rag.embeddings import get_embedder


def test_embedder_uses_config_from_get_embedder():
    """Test that embedders use the config passed to get_embedder."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            provider="ollama",
            model="custom-model",
            vector_dim=512,
        ),
        providers=ProvidersConfig(
            ollama=OllamaConfig(base_url="http://custom-ollama:8080"),
            vllm=VLLMConfig(embeddings_base_url="http://custom-vllm:9000"),
        ),
    )

    embedder = get_embedder(custom_config)

    assert embedder._model == "custom-model"
    assert embedder._vector_dim == 512
    assert embedder._config.providers.ollama.base_url == "http://custom-ollama:8080"


def test_vllm_embedder_uses_config():
    """Test that vllm embedder uses the config passed to get_embedder."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            provider="vllm",
            model="custom-vllm-model",
            vector_dim=768,
        ),
        providers=ProvidersConfig(
            vllm=VLLMConfig(embeddings_base_url="http://custom-vllm:9001"),
        ),
    )

    embedder = get_embedder(custom_config)

    assert embedder._model == "custom-vllm-model"
    assert embedder._vector_dim == 768
    assert (
        embedder._config.providers.vllm.embeddings_base_url == "http://custom-vllm:9001"
    )


def test_openai_embedder_uses_config():
    """Test that openai embedder uses the config passed to get_embedder."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            provider="openai",
            model="text-embedding-3-large",
            vector_dim=3072,
        ),
    )

    embedder = get_embedder(custom_config)

    assert embedder._model == "text-embedding-3-large"
    assert embedder._vector_dim == 3072
    assert embedder._config == custom_config


@pytest.mark.skipif(
    True, reason="VoyageAI is an optional dependency, may not be installed"
)
def test_voyageai_embedder_uses_config():
    """Test that voyageai embedder uses the config passed to get_embedder."""
    custom_config = AppConfig(
        embeddings=EmbeddingsConfig(
            provider="voyageai",
            model="voyage-large-2",
            vector_dim=1536,
        ),
    )

    embedder = get_embedder(custom_config)

    assert embedder._model == "voyage-large-2"
    assert embedder._vector_dim == 1536
    assert embedder._config == custom_config

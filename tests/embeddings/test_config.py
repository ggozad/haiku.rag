import pytest

from haiku.rag.config import (
    AppConfig,
    EmbeddingModelConfig,
    EmbeddingsConfig,
    OllamaConfig,
    ProvidersConfig,
)
from haiku.rag.embeddings import get_embedder


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


@pytest.mark.parametrize("provider", ["ollama", "vllm"])
@pytest.mark.parametrize("suffix", ["", "/v1"])
def test_embedder_base_url_ends_in_one_v1(provider, suffix):
    """A per-model base_url gets /v1 appended once, and one already ending in
    /v1 is left alone."""
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider=provider,
                name="some-model",
                vector_dim=2560,
                base_url=f"http://my-host:8000{suffix}",
            ),
        ),
    )
    embedder = get_embedder(config)
    if provider == "ollama":
        base_url = str(embedder._embedder._model.base_url)  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
    else:
        base_url = embedder._base_url  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    assert base_url.rstrip("/") == "http://my-host:8000/v1"


def test_vector_dim_property_reports_configured_dimension():
    from haiku.rag.embeddings import EmbedderWrapper

    assert EmbedderWrapper(embedder=None, vector_dim=512).vector_dim == 512


@pytest.mark.parametrize(
    "provider,env_var,name",
    [
        ("voyageai", "VOYAGE_API_KEY", "some-model"),
        ("cohere", "CO_API_KEY", "some-model"),
        ("sentence-transformers", None, "all-MiniLM-L6-v2"),
    ],
)
def test_sdk_providers_are_wired_without_a_request(
    monkeypatch, provider, env_var, name
):
    """Construction wires the SDK and reports the configured dimension."""
    if env_var is not None:
        monkeypatch.setenv(env_var, "test-key")
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(provider=provider, name=name, vector_dim=1024),
        ),
    )

    embedder = get_embedder(config)

    assert embedder.vector_dim == 1024
    assert embedder.supports_images is False
    # The provider and model reach the underlying pydantic-ai embedder.
    assert embedder._embedder._model == f"{provider}:{name}"  # ty: ignore[unresolved-attribute]


def test_cohere_floats_rejects_missing_embeddings():
    from types import SimpleNamespace

    from haiku.rag.embeddings.cohere import _floats

    result = SimpleNamespace(embeddings=SimpleNamespace(float_=None))

    with pytest.raises(ValueError, match="no float embeddings"):
        _floats(result)


def test_voyageai_to_pil_rejects_unsupported_type():
    from haiku.rag.embeddings.voyageai import _to_pil

    with pytest.raises(TypeError, match="Unsupported image type"):
        _to_pil("not an image")  # ty: ignore[invalid-argument-type]


def test_openai_embedder_api_key_from_config():
    """A config-supplied api_key reaches the embedding client."""
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="openai",
                name="some-local-model",
                vector_dim=768,
                base_url="http://localhost:8000/v1",
                api_key="sk-vendor-a",
            ),
        ),
    )

    embedder = get_embedder(config)

    assert embedder._vector_dim == 768
    assert embedder._embedder.model._client.api_key == "sk-vendor-a"  # ty: ignore[unresolved-attribute]


def test_ollama_embedder_api_key_from_config():
    """The api_key and the providers.ollama base_url reach the embedding client."""
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="ollama",
                name="qwen3-embedding:4b",
                vector_dim=512,
                api_key="sk-proxy",
            ),
        ),
        providers=ProvidersConfig(
            ollama=OllamaConfig(base_url="http://custom-ollama:8080"),
        ),
    )

    embedder = get_embedder(config)

    assert embedder._vector_dim == 512
    assert str(embedder._embedder.model._client.base_url).startswith(  # ty: ignore[unresolved-attribute]
        "http://custom-ollama:8080/v1"
    )
    assert embedder._embedder.model._client.api_key == "sk-proxy"  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize("multimodal", [False, True])
def test_vllm_embedder_api_key_from_config(multimodal):
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="vllm",
                name="Qwen/Qwen3-VL-Embedding-8B",
                vector_dim=1024,
                base_url="http://vllm:8000/v1",
                api_key="sk-vllm",
                multimodal=multimodal,
            ),
        ),
    )

    embedder = get_embedder(config)

    assert embedder._headers()["Authorization"] == "Bearer sk-vllm"  # ty: ignore[unresolved-attribute]


def test_embedder_api_key_rejected_on_unplumbed_provider():
    """Providers whose client we never build read their own vendor variable;
    an api_key there would be silently dropped."""
    config = AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="voyageai", name="voyage-3", vector_dim=1024, api_key="sk-x"
            ),
        ),
    )

    with pytest.raises(ValueError, match="api_key is not supported"):
        get_embedder(config)

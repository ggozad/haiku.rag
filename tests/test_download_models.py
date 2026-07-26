from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from haiku.rag.client.downloads import download_models
from haiku.rag.config import Config
from haiku.rag.config.models import ModelConfig


@pytest.fixture
def mock_to_thread():
    """Patch asyncio.to_thread to skip docling/tokenizer downloads."""
    with patch("haiku.rag.client.downloads.asyncio.to_thread", new_callable=AsyncMock):
        yield


@asynccontextmanager
async def _mock_httpx_client(stream_fn):
    """Create a mock httpx.AsyncClient context manager with a given stream function."""
    mock_client = AsyncMock()
    mock_client.stream = stream_fn
    yield mock_client


async def test_download_models_ollama_connect_error(mock_to_thread):
    """When Ollama is not running, download_models raises ConnectionError."""

    @asynccontextmanager
    async def failing_stream(method, url, **kwargs):
        raise httpx.ConnectError("All connection attempts failed")
        yield  # unreachable, but needed for generator syntax

    with patch(
        "haiku.rag.client.downloads.httpx.AsyncClient",
        return_value=_mock_httpx_client(failing_stream),
    ):
        with pytest.raises(
            ConnectionError, match="Cannot connect to Ollama"
        ) as exc_info:
            async for _ in download_models(Config):
                pass

        assert "ollama serve" in str(exc_info.value)


async def test_download_models_ollama_pulls_models(mock_to_thread):
    """download_models yields correct progress events for Ollama model pulls."""
    stream_lines = [
        '{"status": "pulling manifest"}',
        "",
        '{"status": "downloading", "digest": "sha256:abc", "total": 1000, "completed": 500}',
        '{"status": "downloading", "digest": "sha256:abc", "total": 1000, "completed": 1000}',
        "not valid json",
        '{"status": "verifying sha256 digest"}',
        '{"status": "writing manifest"}',
        '{"status": "success"}',
    ]

    @asynccontextmanager
    async def mock_stream(method, url, **kwargs):
        mock_resp = AsyncMock()

        async def aiter_lines():
            for line in stream_lines:
                yield line

        mock_resp.aiter_lines = aiter_lines
        yield mock_resp

    with patch(
        "haiku.rag.client.downloads.httpx.AsyncClient",
        return_value=_mock_httpx_client(mock_stream),
    ):
        events = []
        async for progress in download_models(Config):
            events.append(progress)

    # Default config has embeddings=qwen3-embedding:4b, qa=gpt-oss
    ollama_models = {"gpt-oss", "qwen3-embedding:4b"}
    ollama_events = [e for e in events if e.model in ollama_models]
    pulling_events = [e for e in ollama_events if e.status == "pulling"]
    done_events = [e for e in ollama_events if e.status == "done"]
    download_events = [e for e in ollama_events if e.status == "downloading"]

    assert len(pulling_events) == 2
    assert len(done_events) == 2
    assert len(download_events) > 0

    for de in download_events:
        assert de.digest == "sha256:abc"
        assert de.total == 1000
        assert de.completed > 0


async def test_download_models_no_ollama_models(mock_to_thread):
    """When no Ollama models are configured, no Ollama pull events are yielded."""
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.embeddings.model.provider = "openai"
    config.qa.model.provider = "openai"

    events = []
    async for progress in download_models(config):
        events.append(progress)

    models = {e.model for e in events}
    assert "qwen3-embedding:4b" not in models
    assert "gpt-oss" not in models


@pytest.mark.parametrize(
    "configure,expected_model",
    [
        (
            lambda c: setattr(
                c.reranking,
                "model",
                ModelConfig(provider="ollama", name="rerank-model"),
            ),
            "rerank-model",
        ),
        (
            lambda c: (
                setattr(c.processing, "pictures", "description"),
                setattr(
                    c.processing.conversion_options.picture_description.model,
                    "provider",
                    "ollama",
                ),
                setattr(
                    c.processing.conversion_options.picture_description.model,
                    "name",
                    "vision-model",
                ),
            ),
            "vision-model",
        ),
        (
            lambda c: (
                setattr(c.processing, "auto_title", True),
                setattr(c.processing.title_model, "provider", "ollama"),
                setattr(c.processing.title_model, "name", "title-model"),
            ),
            "title-model",
        ),
    ],
    ids=["reranker", "picture_description", "auto_title"],
)
async def test_ollama_models_from_every_config_slot_are_pulled(
    mock_to_thread, configure, expected_model
):
    """Each config slot that can name an ollama model contributes to the pull set."""
    from haiku.rag.config import AppConfig

    config = AppConfig()
    configure(config)

    @asynccontextmanager
    async def mock_stream(method, url, **kwargs):
        mock_resp = AsyncMock()

        async def aiter_lines():
            yield '{"status": "success"}'

        mock_resp.aiter_lines = aiter_lines
        yield mock_resp

    with patch(
        "haiku.rag.client.downloads.httpx.AsyncClient",
        return_value=_mock_httpx_client(mock_stream),
    ):
        pulled = {
            progress.model
            async for progress in download_models(config)
            if progress.status == "pulling"
        }

    assert expected_model in pulled

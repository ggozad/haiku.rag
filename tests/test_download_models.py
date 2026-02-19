from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from haiku.rag.client import HaikuRAG


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent / "cassettes" / "test_download_models")


@pytest.fixture
def mock_to_thread():
    """Patch asyncio.to_thread to skip docling/tokenizer downloads."""
    with patch("haiku.rag.client.asyncio.to_thread", new_callable=AsyncMock):
        yield


@asynccontextmanager
async def _mock_httpx_client(stream_fn):
    """Create a mock httpx.AsyncClient context manager with a given stream function."""
    mock_client = AsyncMock()
    mock_client.stream = stream_fn
    yield mock_client


async def test_download_models_ollama_connect_error(temp_db_path, mock_to_thread):
    """When Ollama is not running, download_models raises ConnectionError."""
    async with HaikuRAG(temp_db_path, create=True) as client:

        @asynccontextmanager
        async def failing_stream(method, url, **kwargs):
            raise httpx.ConnectError("All connection attempts failed")
            yield  # unreachable, but needed for generator syntax

        with patch(
            "haiku.rag.client.httpx.AsyncClient",
            return_value=_mock_httpx_client(failing_stream),
        ):
            with pytest.raises(
                ConnectionError, match="Cannot connect to Ollama"
            ) as exc_info:
                async for _ in client.download_models():
                    pass

            assert "ollama serve" in str(exc_info.value)


async def test_download_models_ollama_pulls_models(temp_db_path, mock_to_thread):
    """download_models yields correct progress events for Ollama model pulls."""
    async with HaikuRAG(temp_db_path, create=True) as client:
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
            "haiku.rag.client.httpx.AsyncClient",
            return_value=_mock_httpx_client(mock_stream),
        ):
            events = []
            async for progress in client.download_models():
                events.append(progress)

        # Default config has embeddings=qwen3-embedding:4b, qa/research=gpt-oss
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


async def test_download_models_no_ollama_models(temp_db_path, mock_to_thread):
    """When no Ollama models are configured, no Ollama pull events are yielded."""
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.embeddings.model.provider = "openai"
    config.qa.model.provider = "openai"
    config.research.model.provider = "openai"

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        events = []
        async for progress in client.download_models():
            events.append(progress)

    models = {e.model for e in events}
    assert "qwen3-embedding:4b" not in models
    assert "gpt-oss" not in models

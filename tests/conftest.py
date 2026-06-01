import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Prevent tests from loading user's local haiku.rag.yaml by setting env var
# to a test config file BEFORE any haiku.rag imports.
# Uses Ollama for embeddings - HTTP calls are recorded/replayed via VCR.
_test_config_dir = tempfile.mkdtemp()
_test_config_path = Path(_test_config_dir) / "test-defaults.yaml"
_test_config_path.write_text("""
embeddings:
  model:
    provider: ollama
    name: qwen3-embedding:4b
    vector_dim: 2560
""")
os.environ["HAIKU_RAG_CONFIG_PATH"] = str(_test_config_path)

import pydantic_ai.models  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

if TYPE_CHECKING:
    from vcr import VCR

setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)
logging.getLogger("vcr.cassette").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def qa_corpus() -> list[dict[str, str]]:
    corpus_path = Path(__file__).parent / "data" / "qa_corpus.json"
    with open(corpus_path) as f:
        return json.load(f)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing.

    Note: Tests that need a database should use HaikuRAG with create=True.
    """
    return tmp_path / "test.lancedb"


@pytest.fixture
def temp_yaml_config(tmp_path, monkeypatch):
    """Create a temporary YAML config file for testing.

    This fixture creates a config file in a temp directory and sets
    the environment variable so config.py will load it.
    """
    config_file = tmp_path / "test-config.yaml"
    config_data = {
        "environment": "production",
        "storage": {
            "data_dir": "",
            "monitor_directories": [],
            "vacuum_retention_seconds": 60,
        },
        "embeddings": {
            "model": {
                "provider": "ollama",
                "name": "qwen3-embedding:4b",
                "vector_dim": 2560,
            }
        },
        "qa": {"model": {"provider": "ollama", "name": "gpt-oss"}},
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Set env var so config loader will find it
    monkeypatch.setenv("HAIKU_RAG_CONFIG_PATH", str(config_file))

    yield config_file


@pytest.fixture
def allow_model_requests():
    with pydantic_ai.models.override_allow_model_requests(True):
        yield


@pytest.fixture(autouse=True)
def set_mock_api_keys(monkeypatch):
    """Set mock API keys for providers that require them during initialization."""
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key-for-vcr-playback")
    if not os.getenv("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-mock-key-for-vcr-playback")
    if not os.getenv("CO_API_KEY"):
        monkeypatch.setenv("CO_API_KEY", "mock-cohere-key-for-vcr-playback")
    if not os.getenv("ZEROENTROPY_API_KEY"):
        monkeypatch.setenv("ZEROENTROPY_API_KEY", "mock-ze-key-for-vcr-playback")
    if not os.getenv("VOYAGE_API_KEY"):
        monkeypatch.setenv("VOYAGE_API_KEY", "mock-voyage-key-for-vcr-playback")
    if not os.getenv("GROQ_API_KEY"):
        monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key-for-vcr-playback")
    if not os.getenv("GOOGLE_API_KEY"):
        monkeypatch.setenv("GOOGLE_API_KEY", "mock-google-key-for-vcr-playback")
    if not os.getenv("AWS_DEFAULT_REGION"):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def pytest_recording_configure(config: Any, vcr: "VCR"):
    from . import json_body_serializer

    vcr.register_serializer("yaml", json_body_serializer)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "ignore_localhost": False,
        "ignore_hosts": ["huggingface.co"],
        "filter_headers": ["authorization", "x-api-key"],
        "decode_compressed_response": True,
    }


@pytest.fixture(scope="session")
def doclaynet_first_page_pdf(tmp_path_factory) -> Path:
    """One-page extract of ``tests/data/doclaynet.pdf`` (the full DocLayNet
    arXiv paper). Most existing tests only need a small PDF with at least
    one picture; this avoids running docling over all nine pages of the
    paper just to assert ``pictures != []``. The full paper is used
    directly by the split-and-merge integration test."""
    import pypdfium2 as pdfium

    src_path = Path(__file__).parent / "data" / "doclaynet.pdf"
    out_dir = tmp_path_factory.mktemp("doclaynet")
    out_path = out_dir / "page0.pdf"

    src = pdfium.PdfDocument(str(src_path))
    try:
        dst = pdfium.PdfDocument.new()
        try:
            dst.import_pages(src, [0])
            with open(out_path, "wb") as f:
                dst.save(f)
        finally:
            dst.close()
    finally:
        src.close()
    return out_path

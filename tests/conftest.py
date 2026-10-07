import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
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

# telemetry.configure() passes send_to_logfire="if-token-present" explicitly,
# which beats LOGFIRE_SEND_TO_LOGFIRE, so no token must resolve: drop the
# environment variable and point credentials discovery at an empty directory.
os.environ.pop("LOGFIRE_TOKEN", None)
os.environ["LOGFIRE_CREDENTIALS_DIR"] = tempfile.mkdtemp()

import pydantic_ai.models  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from .services import reachable  # noqa: E402

if TYPE_CHECKING:
    from vcr import VCR

    from haiku.rag.client import HaikuRAG
    from haiku.rag.client.scope import DatabaseScope
    from haiku.rag.client.session import SingleDatabaseSession
    from haiku.rag.config.models import AppConfig

setattr(pydantic_ai.models, "ALLOW_MODEL_REQUESTS", False)
logging.getLogger("vcr.cassette").setLevel(logging.WARNING)

_CENTRAL_CASSETTE_SUITES = frozenset(
    {
        "client",
        "chunkers",
        "converters",
        "embeddings",
        "interfaces",
        "providers",
        "reranking",
        "store",
    }
)


@contextmanager
def capture_logs(
    logger: logging.Logger, level: int
) -> Iterator[list[logging.LogRecord]]:
    """Collect records emitted by ``logger`` at or above ``level``.

    Attaches directly to the given logger instead of using ``caplog``:
    ``haiku.rag.logging.get_logger()`` sets ``propagate=False`` on the
    ``haiku.rag`` logger, so records never reach caplog's root handler once
    any test in the session has called it.
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=level)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


@pytest.fixture
def exporter():
    """Collect spans in memory, restoring logfire's inert default on teardown."""
    import logfire
    from logfire.testing import SimpleSpanProcessor, TestExporter

    test_exporter = TestExporter()
    logfire.configure(
        send_to_logfire=False,
        console=False,
        additional_span_processors=[SimpleSpanProcessor(test_exporter)],
    )
    yield test_exporter
    logfire.configure(send_to_logfire=False, console=False)


@pytest.fixture(scope="session")
def qa_corpus() -> list[dict[str, str]]:
    corpus_path = Path(__file__).parent / "data" / "qa_corpus.json"
    with open(corpus_path, encoding="utf-8") as f:
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
        "qa": {"model": {"provider": "ollama", "name": "qwen3.8"}},
    }

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)

    # Set env var so config loader will find it
    monkeypatch.setenv("HAIKU_RAG_CONFIG_PATH", str(config_file))

    yield config_file


@pytest.fixture
def allow_model_requests():
    with pydantic_ai.models.override_allow_model_requests(True):
        yield


@pytest.fixture(autouse=True)
def allow_expected_model_requests(request):
    """Let a `vcr` or `integration` test reach a model.

    The request guard sits above VCR's HTTP interception, so it rejects a
    cassette replay, and an integration test calls a live service by design.
    An unrecorded call under `vcr` still fails, on `record_mode=none`.
    """
    if not any(
        request.node.get_closest_marker(marker) for marker in ("vcr", "integration")
    ):
        yield
        return
    with pydantic_ai.models.override_allow_model_requests(True):
        yield


@pytest.fixture(autouse=True)
def skip_docling_serve_delays_during_replay(
    request, record_mode, disable_recording, monkeypatch
):
    """Skip docling-serve polling delays during cassette playback."""
    if (
        request.node.get_closest_marker("vcr") is None
        or request.node.get_closest_marker("integration") is not None
        or record_mode != "none"
        or disable_recording
    ):
        yield
        return

    async def no_delay(_seconds: float) -> None:
        pass

    monkeypatch.setattr("haiku.rag.providers.docling_serve.sleep", no_delay)
    yield


@pytest.fixture(autouse=True)
def set_mock_api_keys(monkeypatch):
    """Set mock API keys for providers that require them during initialization."""
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key-for-vcr-playback")
    if not os.getenv("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-mock-key-for-vcr-playback")
    if not os.getenv("OPENROUTER_API_KEY"):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-mock-key-for-vcr-playback")
    if not os.getenv("CO_API_KEY"):
        monkeypatch.setenv("CO_API_KEY", "mock-cohere-key-for-vcr-playback")
    if not os.getenv("ZEROENTROPY_API_KEY"):
        monkeypatch.setenv("ZEROENTROPY_API_KEY", "mock-ze-key-for-vcr-playback")
    if not os.getenv("VOYAGE_API_KEY"):
        monkeypatch.setenv("VOYAGE_API_KEY", "mock-voyage-key-for-vcr-playback")
    if not os.getenv("GROQ_API_KEY"):
        monkeypatch.setenv("GROQ_API_KEY", "mock-groq-key-for-vcr-playback")
    if not os.getenv("MISTRAL_API_KEY"):
        monkeypatch.setenv("MISTRAL_API_KEY", "mock-mistral-key-for-vcr-playback")
    if not os.getenv("GOOGLE_API_KEY"):
        monkeypatch.setenv("GOOGLE_API_KEY", "mock-google-key-for-vcr-playback")
    if not os.getenv("AWS_DEFAULT_REGION"):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def pytest_recording_configure(config: Any, vcr: "VCR"):
    from . import json_body_serializer

    vcr.register_serializer("yaml", json_body_serializer)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    module_path = Path(str(request.node.path))
    cassette_root = module_path.parent / "cassettes"
    if module_path.parent.name in _CENTRAL_CASSETTE_SUITES:
        cassette_root = Path(__file__).parent / "cassettes"
    return str(cassette_root / module_path.stem)


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


@pytest.fixture(scope="session")
async def docling_local_models(
    tmp_path_factory: pytest.TempPathFactory,
    doclaynet_first_page_pdf: Path,
    worker_id: str,
) -> None:
    """Initialize docling-local model files once across xdist workers."""
    from filelock import FileLock

    from haiku.rag.config import get_config
    from haiku.rag.converters.docling_local import DoclingLocalConverter

    base_temp = tmp_path_factory.getbasetemp()
    shared_temp = base_temp if worker_id == "master" else base_temp.parent
    lock_path = shared_temp / "docling-local-models.lock"
    ready_path = shared_temp / "docling-local-models.ready"
    with FileLock(lock_path, timeout=300):
        if ready_path.exists():
            return
        config = get_config().model_copy(deep=True)
        await DoclingLocalConverter(config).convert_file(doclaynet_first_page_pdf)
        ready_path.touch()


# --- external services for integration tests ---
#
# Integration tests (marked `integration`, excluded in CI via `-m "not
# integration"`) need external services. Bring them up with
#   docker compose -f tests/docker/docker-compose.yml up -d
# These fixtures hand the test a connection URL when the service is reachable
# (or when the matching env var points at an external instance), and skip the
# test otherwise.

_COMPOSE_HINT = (
    "start it with `docker compose -f tests/docker/docker-compose.yml up -d`"
)


@pytest.fixture(scope="session")
def postgres_dburi() -> str:
    """A reachable Postgres queue URL. Uses HAIKU_RAG_TEST_PG_DBURI when set,
    otherwise the docker-compose `postgres` service. Skips when neither is up."""
    override = os.environ.get("HAIKU_RAG_TEST_PG_DBURI")
    if override:
        return override
    if not reachable("localhost", 55432):
        pytest.skip(f"Postgres not reachable on localhost:55432 — {_COMPOSE_HINT}")
    return "postgresql+asyncpg://haiku:secret@localhost:55432/haiku_rag_test"


@pytest.fixture(scope="session")
def docling_serve_url() -> str:
    """A reachable docling-serve base URL. Uses HAIKU_RAG_TEST_DOCLING_SERVE_URL
    when set, otherwise the docker-compose `docling-serve` service. Skips when
    neither is up."""
    override = os.environ.get("HAIKU_RAG_TEST_DOCLING_SERVE_URL")
    if override:
        return override
    if not reachable("localhost", 5001):
        pytest.skip(f"docling-serve not reachable on localhost:5001 — {_COMPOSE_HINT}")
    return "http://localhost:5001"


def run_with_lance_memory_pool(
    script: str, pool_bytes: int, *args: str
) -> "subprocess.CompletedProcess[str]":
    """Run `script` in a fresh interpreter whose lance query memory pool is `pool_bytes`.

    lance reads `LANCE_MEM_POOL_SIZE` once per process, at its first query."""
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        env={**os.environ, "LANCE_MEM_POOL_SIZE": str(pool_bytes)},
        capture_output=True,
        text=True,
    )


def writing(client: "HaikuRAG") -> "SingleDatabaseSession":
    """The database a write implementation works on, from a client holding one.

    Write implementations take a session, never a client, so a set can
    never reach them. Tests that call one directly go through here."""
    from haiku.rag.client.session import SingleDatabaseSession

    assert isinstance(client._session, SingleDatabaseSession)
    return client._session


def build_pdf(attachments: list[tuple[str, bytes]]) -> bytes:
    """Build a minimal one-page PDF with the given (name, bytes) attachments."""
    import io

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(200, 200)
    for name, data in attachments:
        att = pdf.new_attachment(name)
        att.set_data(data)
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


async def fake_ingest_fetch_result(
    session,
    result,
    *,
    title,
    user_metadata,
    stored_uri,
    existing_doc,
    source_id=None,
    depth=0,
    filename=None,
    metadata_provider=None,
):
    """A stand-in for ``_ingest_fetch_result`` that skips docling/embedder
    entirely: it writes the document with content_type/md5/parent_uri set
    correctly, then defers to the real ``_reconcile_pdf_attachments`` so
    recursive logic stays under test. Metadata providers are not simulated."""
    from haiku.rag.client.documents import _reconcile_pdf_attachments
    from haiku.rag.store.models.document import Document

    final_metadata = {
        **(user_metadata or {}),
        "content_type": result.content_type,
        "md5": result.content_hash,
        **result.extra_metadata,
    }
    if result.revision is not None:
        final_metadata["source_revision"] = result.revision
    if source_id is not None:
        final_metadata["source_id"] = source_id

    if existing_doc:
        existing_doc.content = ""
        existing_doc.metadata = final_metadata
        if title is not None:
            existing_doc.title = title
        doc = await session.document_repository.update(existing_doc)
    else:
        doc = await session.document_repository.create(
            Document(
                content="",
                uri=stored_uri,
                title=title,
                metadata=final_metadata,
            )
        )
    await _reconcile_pdf_attachments(session, doc, result.body, depth=depth)
    return doc


def for_path(
    db_path: "Path | str | None" = None, config: "AppConfig | None" = None
) -> "DatabaseScope":
    """A scope covering one database at `db_path`.

    The application layer takes the databases it works on, already resolved.
    Tests that hold a path and need a scope go through here.
    """
    from haiku.rag.client.scope import DatabaseScope
    from haiku.rag.config import get_config

    return DatabaseScope.resolve(
        config if config is not None else get_config(), database_path=db_path
    )


TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def current_trace() -> tuple[str, str]:
    """The current span's trace id and trace state header."""
    from opentelemetry import trace

    span_context = trace.get_current_span().get_span_context()
    return format(span_context.trace_id, "032x"), span_context.trace_state.to_header()


@contextmanager
def _covering_returns(stub, client):
    """Make a patched `HaikuRAG` hand back `client` however it is constructed.

    The TUIs build their client through `HaikuRAG._covering`, so patching the
    constructor alone leaves `_covering` answering with a fresh Mock.
    """
    stub.return_value = client
    stub._covering.return_value = client
    yield stub

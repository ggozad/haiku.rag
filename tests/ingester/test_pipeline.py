from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.models import Job, JobOp, JobStatus
from haiku.rag.ingester.workers.pipeline import run_job
from haiku.rag.store.models.document import Document


def _job(
    *,
    op: JobOp = JobOp.UPSERT,
    uri: str = "https://example.com/a.pdf",
    extra: dict | None = None,
    attempts: int = 0,
) -> Job:
    now = datetime.now(UTC)
    return Job(
        id="job-1",
        source_id="src",
        uri=uri,
        op=op,
        status=JobStatus.CLAIMED,
        attempts=attempts,
        max_attempts=5,
        extra=extra,
        enqueued_at=now,
        scheduled_at=now,
    )


def _mock_client() -> AsyncMock:
    return AsyncMock(spec=HaikuRAG)


@pytest.mark.asyncio
async def test_upsert_calls_create_document_from_source_and_returns_metadata():
    client = _mock_client()
    client.create_document_from_source.return_value = Document(
        id="doc-42",
        content="x",
        uri="https://example.com/a.pdf",
        metadata={
            "md5": "abcd",
            "source_revision": "xyz",
            "content_type": "application/pdf",
        },
    )

    result = await run_job(client, _job())

    assert result.document_id == "doc-42"
    assert result.revision == "xyz"
    assert result.content_hash == "abcd"
    assert result.deleted is False
    client.create_document_from_source.assert_awaited_once_with(
        "https://example.com/a.pdf", sources=None, source_id="src"
    )


@pytest.mark.asyncio
async def test_upsert_threads_configured_sources_to_client():
    """The list of configured Source adapters reaches the client so
    resolve_fetcher can pick the right one by source_id."""
    from haiku.rag.ingester.sources.http import HTTPSource

    client = _mock_client()
    client.create_document_from_source.return_value = Document(
        id="d", content="x", uri="u", metadata={}
    )
    configured = HTTPSource(source_id="urls", headers={"Authorization": "Bearer abc"})

    await run_job(client, _job(), sources=[configured])
    client.create_document_from_source.assert_awaited_once_with(
        "https://example.com/a.pdf", sources=[configured], source_id="src"
    )


@pytest.mark.asyncio
async def test_delete_calls_delete_document_when_present():
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(id="doc-9", content="", uri="u")

    result = await run_job(client, _job(op=JobOp.DELETE))

    assert result.deleted is True
    assert result.document_id is None
    client.delete_document.assert_awaited_once_with("doc-9")


@pytest.mark.asyncio
async def test_delete_is_noop_when_document_missing():
    client = _mock_client()
    client.get_document_by_uri.return_value = None

    result = await run_job(client, _job(op=JobOp.DELETE))

    assert result.deleted is True
    client.delete_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_unsupported_extension_classified_permanent():
    from haiku.rag.client.exceptions import UnsupportedSourceError

    client = _mock_client()
    client.create_document_from_source.side_effect = UnsupportedSourceError(
        "Unsupported file extension: .xyz"
    )

    with pytest.raises(PermanentError, match="Unsupported"):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_unsupported_content_type_classified_permanent():
    from haiku.rag.client.exceptions import UnsupportedSourceError

    client = _mock_client()
    client.create_document_from_source.side_effect = UnsupportedSourceError(
        "Unsupported content type/extension: application/octet-stream/.bin"
    )
    with pytest.raises(PermanentError):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_invalid_s3_uri_classified_permanent():
    from haiku.rag.client.exceptions import UnsupportedSourceError

    client = _mock_client()
    client.create_document_from_source.side_effect = UnsupportedSourceError(
        "Invalid S3 URI: s3:///bad"
    )
    with pytest.raises(PermanentError):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_missing_file_classified_permanent():
    from haiku.rag.client.exceptions import UnsupportedSourceError

    client = _mock_client()
    client.create_document_from_source.side_effect = UnsupportedSourceError(
        "File does not exist: /nope"
    )
    with pytest.raises(PermanentError):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_other_value_error_classified_transient():
    client = _mock_client()
    client.create_document_from_source.side_effect = ValueError("DB busy")
    with pytest.raises(TransientError):
        await run_job(client, _job())


@pytest.mark.parametrize("status", [401, 403, 404, 410])
@pytest.mark.asyncio
async def test_http_4xx_classified_permanent(status):
    client = _mock_client()
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://example.com/a")
    )
    client.create_document_from_source.side_effect = httpx.HTTPStatusError(
        "err", request=response.request, response=response
    )
    with pytest.raises(PermanentError):
        await run_job(client, _job())


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
@pytest.mark.asyncio
async def test_http_408_429_5xx_classified_transient(status):
    client = _mock_client()
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://example.com/a")
    )
    client.create_document_from_source.side_effect = httpx.HTTPStatusError(
        "err", request=response.request, response=response
    )
    with pytest.raises(TransientError):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_connect_error_classified_transient():
    client = _mock_client()
    client.create_document_from_source.side_effect = httpx.ConnectError("boom")
    with pytest.raises(TransientError):
        await run_job(client, _job())


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: TimeoutError("slow"),
        lambda: OSError("io"),
    ],
)
@pytest.mark.asyncio
async def test_timeout_and_io_errors_classified_transient(exc_factory):
    """TimeoutError / OSError both classify as transient. Without this
    branch they'd fall through to the generic "unexpected" wrapper."""
    client = _mock_client()
    client.create_document_from_source.side_effect = exc_factory()
    with pytest.raises(TransientError, match="timeout/io"):
        await run_job(client, _job())


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: httpx.ConnectTimeout("slow connect"),
        lambda: httpx.ReadTimeout("slow read"),
        lambda: httpx.WriteTimeout("slow write"),
        lambda: httpx.PoolTimeout("pool"),
        lambda: httpx.ProxyError("proxy"),
    ],
)
@pytest.mark.asyncio
async def test_transport_subclasses_classified_transient(exc_factory):
    """The classifier umbrellas on httpx.TransportError so every transport-
    layer subclass routes to TransientError, not the generic 'unexpected'
    fallback."""
    client = _mock_client()
    client.create_document_from_source.side_effect = exc_factory()
    with pytest.raises(TransientError, match="network"):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_unknown_exception_classified_transient():
    client = _mock_client()
    client.create_document_from_source.side_effect = RuntimeError("???")
    with pytest.raises(TransientError):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_directory_result_classified_permanent():
    client = _mock_client()
    client.create_document_from_source.return_value = [
        Document(id="a", content="", uri="u1"),
        Document(id="b", content="", uri="u2"),
    ]
    with pytest.raises(PermanentError, match="directory"):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_existing_permanent_error_passes_through_unchanged():
    client = _mock_client()
    sentinel = PermanentError("explicit")
    client.create_document_from_source.side_effect = sentinel
    with pytest.raises(PermanentError) as excinfo:
        await run_job(client, _job())
    assert excinfo.value is sentinel


@pytest.mark.asyncio
async def test_file_not_found_classified_as_permanent():
    """A deleted file should go straight to the DLQ, not retry."""
    client = _mock_client()
    client.create_document_from_source.side_effect = FileNotFoundError("gone")
    with pytest.raises(PermanentError, match="file not found"):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_permission_error_classified_as_permanent():
    """An unreadable file should go straight to the DLQ, not retry."""
    client = _mock_client()
    client.create_document_from_source.side_effect = PermissionError("no access")
    with pytest.raises(PermanentError, match="permission denied"):
        await run_job(client, _job())

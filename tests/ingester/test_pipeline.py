from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.models import Job, JobOp, JobStatus
from haiku.rag.ingester.sources.base import FetchResult, FileTooLargeError, Source
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
        "https://example.com/a.pdf", sources=None, source_id="src", metadata=None
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
        "https://example.com/a.pdf",
        sources=[configured],
        source_id="src",
        metadata=None,
    )


class _MetadataProvider:
    """Provider double returning scripted metadata, or raising."""

    def __init__(self, metadata: dict | None = None, *, error: Exception | None = None):
        self._metadata = metadata or {}
        self._error = error

    async def __call__(self, source_id: str, uri: str) -> dict:
        if self._error is not None:
            raise self._error
        return {**self._metadata, "source": source_id}


@pytest.mark.asyncio
async def test_provider_metadata_passed_to_client():
    client = _mock_client()
    client.create_document_from_source.return_value = Document(
        id="d", content="x", uri="u", metadata={}
    )
    providers = {"src": _MetadataProvider({"classification": "secret"})}

    await run_job(client, _job(), metadata_providers=providers)

    client.create_document_from_source.assert_awaited_once_with(
        "https://example.com/a.pdf",
        sources=None,
        source_id="src",
        metadata={"classification": "secret", "source": "src"},
    )


@pytest.mark.asyncio
async def test_provider_cannot_override_system_keys():
    """Reserved source-derived keys are stripped from provider output so the
    metadata-only refresh path can't let a provider overwrite md5 /
    source_revision / content_type (which would corrupt sync_state)."""
    client = _mock_client()
    client.create_document_from_source.return_value = Document(
        id="d", content="x", uri="u", metadata={}
    )
    providers = {
        "src": _MetadataProvider(
            {
                "md5": "spoof",
                "source_revision": "spoof",
                "content_type": "text/spoof",
                "classification": "secret",
            }
        )
    }

    await run_job(client, _job(), metadata_providers=providers)

    client.create_document_from_source.assert_awaited_once_with(
        "https://example.com/a.pdf",
        sources=None,
        source_id="src",
        metadata={"classification": "secret", "source": "src"},
    )


@pytest.mark.asyncio
async def test_no_provider_for_source_passes_no_metadata():
    """A provider registered for a different source must not apply here."""
    client = _mock_client()
    client.create_document_from_source.return_value = Document(
        id="d", content="x", uri="u", metadata={}
    )
    providers = {"other": _MetadataProvider({"classification": "secret"})}

    await run_job(client, _job(), metadata_providers=providers)

    client.create_document_from_source.assert_awaited_once_with(
        "https://example.com/a.pdf", sources=None, source_id="src", metadata=None
    )


@pytest.mark.asyncio
async def test_provider_error_is_classified_and_blocks_ingest():
    client = _mock_client()
    providers = {"src": _MetadataProvider(error=httpx.ConnectError("provider down"))}

    with pytest.raises(TransientError):
        await run_job(client, _job(), metadata_providers=providers)

    client.create_document_from_source.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_not_called_for_delete():
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(id="doc-9", content="", uri="u")
    provider = _MetadataProvider(error=AssertionError("must not run on DELETE"))

    result = await run_job(
        client, _job(op=JobOp.DELETE), metadata_providers={"src": provider}
    )

    assert result.deleted is True


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


class _StubSource:
    """Source double whose head() returns a scripted revision or raises."""

    def __init__(
        self,
        source_id: str,
        revision: str | None,
        *,
        head_error: Exception | None = None,
    ):
        self.source_id = source_id
        self._revision = revision
        self._head_error = head_error

    def supports(self, uri: str) -> bool:
        return True

    async def head(self, uri: str) -> str | None:
        if self._head_error is not None:
            raise self._head_error
        return self._revision

    async def aclose(self) -> None: ...

    async def fetch(self, uri: str) -> FetchResult:  # pragma: no cover - unused
        raise NotImplementedError

    async def discover(self, since=None, *, known_uris=None):  # pragma: no cover
        raise NotImplementedError
        yield


@pytest.mark.asyncio
async def test_delete_skipped_when_resource_restored_on_source():
    """The file is back on disk by the time the DELETE runs, so head()
    returns a revision and the delete is skipped — otherwise a live document
    gets blackholed until the next sweep."""
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(id="doc-9", content="", uri="u")
    sources: list[Source] = [_StubSource("src", "12345")]

    result = await run_job(client, _job(op=JobOp.DELETE), sources=sources)

    assert result.deleted is False
    client.delete_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_proceeds_when_resource_absent_on_source():
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(id="doc-9", content="", uri="u")
    sources: list[Source] = [_StubSource("src", None)]

    result = await run_job(client, _job(op=JobOp.DELETE), sources=sources)

    assert result.deleted is True
    client.delete_document.assert_awaited_once_with("doc-9")


@pytest.mark.asyncio
async def test_delete_proceeds_when_source_unresolvable():
    """No configured source for the job: the probe can't run, so the delete
    proceeds exactly as before."""
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(id="doc-9", content="", uri="u")

    result = await run_job(client, _job(op=JobOp.DELETE), sources=[])

    assert result.deleted is True
    client.delete_document.assert_awaited_once_with("doc-9")


@pytest.mark.asyncio
async def test_delete_proceeds_when_head_probe_raises():
    """A failing head() probe must not block the delete."""
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(id="doc-9", content="", uri="u")
    sources: list[Source] = [_StubSource("src", None, head_error=OSError("boom"))]

    result = await run_job(client, _job(op=JobOp.DELETE), sources=sources)

    assert result.deleted is True
    client.delete_document.assert_awaited_once_with("doc-9")


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


@pytest.mark.asyncio
@pytest.mark.parametrize("exc_class", [IsADirectoryError, NotADirectoryError])
async def test_directory_errors_classified_as_permanent(exc_class):
    """Pointing at a directory instead of a file should DLQ immediately."""
    client = _mock_client()
    client.create_document_from_source.side_effect = exc_class("not a file")
    with pytest.raises(PermanentError, match="path error"):
        await run_job(client, _job())


@pytest.mark.asyncio
async def test_file_too_large_classified_as_permanent():
    client = _mock_client()
    client.create_document_from_source.side_effect = FileTooLargeError("too big")
    with pytest.raises(PermanentError, match="too big"):
        await run_job(client, _job())

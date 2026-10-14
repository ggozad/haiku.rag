import logging
import socket
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest
from obstore.exceptions import (
    GenericError,
    InvalidPathError,
    JoinError,
    PermissionDeniedError,
    UnauthenticatedError,
    UnknownConfigurationKeyError,
)

from haiku.rag.client import HaikuRAG
from haiku.rag.hooks import Hook
from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.models import Job, JobOp, JobStatus
from haiku.rag.ingester.workers.pipeline import run_job
from haiku.rag.sources.base import FetchResult, FileTooLargeError, Source
from haiku.rag.sources.fs import FSSource
from haiku.rag.store.models.document import Document
from tests.conftest import capture_logs

_PIPELINE_LOGGER = logging.getLogger("haiku.rag.ingester.workers.pipeline")


def _job(
    *,
    op: JobOp = JobOp.UPSERT,
    uri: str = "https://example.com/a.pdf",
    revision: str | None = None,
    extra: dict | None = None,
    attempts: int = 0,
) -> Job:
    now = datetime.now(UTC)
    return Job(
        id="job-1",
        source_id="src",
        uri=uri,
        op=op,
        revision=revision,
        status=JobStatus.CLAIMED,
        attempts=attempts,
        max_attempts=5,
        extra=extra,
        enqueued_at=now,
        scheduled_at=now,
    )


def _mock_client() -> AsyncMock:
    client = AsyncMock(spec=HaikuRAG)
    client.delete_document.return_value = True
    return client


async def test_upsert_ingests_and_returns_metadata():
    client = _mock_client()
    client._ingest_observed.return_value = Document(
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
    client._ingest_observed.assert_awaited_once_with(
        "https://example.com/a.pdf",
        sources=None,
        source_id="src",
        metadata_provider=None,
        observed_revision=None,
    )


async def test_upsert_threads_configured_sources_to_client():
    """The list of configured Source adapters reaches the client so
    resolve_fetcher can pick the right one by source_id."""
    from haiku.rag.sources.http import HTTPSource

    client = _mock_client()
    client._ingest_observed.return_value = Document(
        id="d", content="x", uri="u", metadata={}
    )
    configured = HTTPSource(source_id="urls", headers={"Authorization": "Bearer abc"})

    await run_job(client, _job(), sources=[configured])
    client._ingest_observed.assert_awaited_once_with(
        "https://example.com/a.pdf",
        sources=[configured],
        source_id="src",
        metadata_provider=None,
        observed_revision=None,
    )


class _MetadataProvider:
    """Provider double returning scripted metadata, or raising."""

    def __init__(self, metadata: dict | None = None, *, error: Exception | None = None):
        self._metadata = metadata or {}
        self._error = error

    async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
        if self._error is not None:
            raise self._error
        return {**self._metadata, "source": source_id}


async def test_provider_passed_to_client():
    client = _mock_client()
    client._ingest_observed.return_value = Document(
        id="d", content="x", uri="u", metadata={}
    )
    provider = _MetadataProvider({"classification": "secret"})
    providers = {"src": provider}

    await run_job(client, _job(), metadata_providers=providers)

    client._ingest_observed.assert_awaited_once_with(
        "https://example.com/a.pdf",
        sources=None,
        source_id="src",
        metadata_provider=provider,
        observed_revision=None,
    )


async def test_provider_not_called_in_pipeline():
    """Provider execution happens inside create_document_from_source after
    fetch, where FetchResult exists."""
    client = _mock_client()
    client._ingest_observed.return_value = Document(
        id="d", content="x", uri="u", metadata={}
    )
    provider = _MetadataProvider(
        error=AssertionError("pipeline must not call provider")
    )

    await run_job(client, _job(), metadata_providers={"src": provider})

    client._ingest_observed.assert_awaited_once_with(
        "https://example.com/a.pdf",
        sources=None,
        source_id="src",
        metadata_provider=provider,
        observed_revision=None,
    )


async def test_no_provider_for_source_passes_no_metadata():
    """A provider registered for a different source must not apply here."""
    client = _mock_client()
    client._ingest_observed.return_value = Document(
        id="d", content="x", uri="u", metadata={}
    )
    providers = {"other": _MetadataProvider({"classification": "secret"})}

    await run_job(client, _job(), metadata_providers=providers)

    client._ingest_observed.assert_awaited_once_with(
        "https://example.com/a.pdf",
        sources=None,
        source_id="src",
        metadata_provider=None,
        observed_revision=None,
    )


async def test_provider_error_from_client_is_classified_and_blocks_ingest():
    client = _mock_client()
    client._ingest_observed.side_effect = httpx.ConnectError("provider down")
    providers = {"src": _MetadataProvider()}

    with pytest.raises(TransientError):
        await run_job(client, _job(), metadata_providers=providers)

    client._ingest_observed.assert_awaited_once()


async def test_provider_not_called_for_delete():
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri="u", metadata={"source_id": "src"}
    )
    provider = _MetadataProvider(error=AssertionError("must not run on DELETE"))

    result = await run_job(
        client, _job(op=JobOp.DELETE), metadata_providers={"src": provider}
    )

    assert result.deleted is True


async def test_delete_calls_delete_document_when_present():
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri="u", metadata={"source_id": "src"}
    )

    result = await run_job(client, _job(op=JobOp.DELETE))

    assert result.deleted is True
    assert result.document_id is None
    client.delete_document.assert_awaited_once_with("doc-9", source_id="src")


@pytest.mark.parametrize("manifest", [None, {"_manifest": {"version": 1}}])
async def test_delete_skips_a_document_owned_by_another_source(manifest):
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri="u", metadata={"source_id": "other"}
    )

    with capture_logs(_PIPELINE_LOGGER, logging.WARNING) as records:
        result = await run_job(client, _job(op=JobOp.DELETE, extra=manifest))

    assert result.deleted is False
    client.delete_document.assert_not_awaited()
    assert [r.getMessage() for r in records] == [
        "Not deleting doc-9 (u) for source 'src': it belongs to source 'other'"
    ]


async def test_delete_skips_a_document_without_a_source():
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(id="doc-9", content="", uri="u")

    with capture_logs(_PIPELINE_LOGGER, logging.WARNING) as records:
        result = await run_job(client, _job(op=JobOp.DELETE))

    assert result.deleted is False
    client.delete_document.assert_not_awaited()
    assert [r.getMessage() for r in records] == [
        "Not deleting doc-9 (u) for source 'src': it belongs to no source"
    ]


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
        self.head_calls = 0

    def supports(self, uri: str) -> bool:
        return True

    async def head(self, uri: str) -> str | None:
        self.head_calls += 1
        if self._head_error is not None:
            raise self._head_error
        return self._revision

    async def aclose(self) -> None: ...

    async def fetch(self, uri: str) -> FetchResult:  # pragma: no cover - unused
        raise NotImplementedError

    async def discover(self, since=None, *, known_uris=None):  # pragma: no cover
        raise NotImplementedError
        yield


async def test_delete_skipped_when_resource_restored_on_source():
    """The file is back on disk by the time the DELETE runs, so head()
    returns a revision and the delete is skipped — otherwise a live document
    gets blackholed until the next sweep."""
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri="u", metadata={"source_id": "src"}
    )
    sources: list[Source] = [_StubSource("src", "12345")]

    result = await run_job(client, _job(op=JobOp.DELETE), sources=sources)

    assert result.deleted is False
    client.delete_document.assert_not_awaited()


async def test_manifest_delete_proceeds_when_resource_restored_on_source():
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri="u", metadata={"source_id": "src"}
    )
    sources: list[Source] = [_StubSource("src", "12345")]

    result = await run_job(
        client,
        _job(op=JobOp.DELETE, extra={"_manifest": {"version": 1}}),
        sources=sources,
    )

    assert result.deleted is True
    client.delete_document.assert_awaited_once_with("doc-9", source_id="src")


async def test_delete_proceeds_when_resource_absent_on_source():
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri="u", metadata={"source_id": "src"}
    )
    sources: list[Source] = [_StubSource("src", None)]

    result = await run_job(client, _job(op=JobOp.DELETE), sources=sources)

    assert result.deleted is True
    client.delete_document.assert_awaited_once_with("doc-9", source_id="src")


async def test_manifest_upsert_rejects_stale_revision_before_fetch():
    client = _mock_client()
    sources: list[Source] = [_StubSource("src", "r2")]

    with pytest.raises(PermanentError, match="manifest revision is stale"):
        await run_job(
            client,
            _job(revision="r1", extra={"_manifest": {"version": 1}}),
            sources=sources,
        )

    client._ingest_observed.assert_not_awaited()


async def test_manifest_upsert_ingests_when_revision_matches():
    client = _mock_client()
    client._ingest_observed.return_value = Document(
        id="doc-42",
        content="x",
        uri="https://example.com/a.pdf",
        metadata={"md5": "abcd", "source_revision": "r1"},
    )
    sources: list[Source] = [_StubSource("src", "r1")]

    result = await run_job(
        client,
        _job(revision="r1", extra={"_manifest": {"version": 1}}),
        sources=sources,
    )

    assert result.document_id == "doc-42"
    client._ingest_observed.assert_awaited_once()


async def test_delete_proceeds_when_source_unresolvable():
    """No configured source for the job: the probe can't run, so the delete
    proceeds exactly as before."""
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri="u", metadata={"source_id": "src"}
    )

    result = await run_job(client, _job(op=JobOp.DELETE), sources=[])

    assert result.deleted is True
    client.delete_document.assert_awaited_once_with("doc-9", source_id="src")


async def test_delete_proceeds_when_head_probe_raises():
    """A failing head() probe must not block the delete."""
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri="u", metadata={"source_id": "src"}
    )
    sources: list[Source] = [_StubSource("src", None, head_error=OSError("boom"))]

    result = await run_job(client, _job(op=JobOp.DELETE), sources=sources)

    assert result.deleted is True
    client.delete_document.assert_awaited_once_with("doc-9", source_id="src")


async def test_unsupported_extension_classified_permanent():
    from haiku.rag.client.exceptions import UnsupportedSourceError

    client = _mock_client()
    client._ingest_observed.side_effect = UnsupportedSourceError(
        "Unsupported file extension: .xyz"
    )

    with pytest.raises(PermanentError, match="Unsupported"):
        await run_job(client, _job())


async def test_unsupported_content_type_classified_permanent():
    from haiku.rag.client.exceptions import UnsupportedSourceError

    client = _mock_client()
    client._ingest_observed.side_effect = UnsupportedSourceError(
        "Unsupported content type/extension: application/octet-stream/.bin"
    )
    with pytest.raises(PermanentError):
        await run_job(client, _job())


async def test_invalid_s3_uri_classified_permanent():
    from haiku.rag.client.exceptions import UnsupportedSourceError

    client = _mock_client()
    client._ingest_observed.side_effect = UnsupportedSourceError(
        "Invalid S3 URI: s3:///bad"
    )
    with pytest.raises(PermanentError):
        await run_job(client, _job())


async def test_missing_file_classified_permanent():
    from haiku.rag.client.exceptions import UnsupportedSourceError

    client = _mock_client()
    client._ingest_observed.side_effect = UnsupportedSourceError(
        "File does not exist: /nope"
    )
    with pytest.raises(PermanentError):
        await run_job(client, _job())


async def test_other_value_error_classified_transient():
    client = _mock_client()
    client._ingest_observed.side_effect = ValueError("DB busy")
    with pytest.raises(TransientError):
        await run_job(client, _job())


@pytest.mark.parametrize("status", [401, 403, 404, 410])
async def test_http_4xx_classified_permanent(status):
    client = _mock_client()
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://example.com/a")
    )
    client._ingest_observed.side_effect = httpx.HTTPStatusError(
        "err", request=response.request, response=response
    )
    with pytest.raises(PermanentError):
        await run_job(client, _job())


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503])
async def test_http_408_429_5xx_classified_transient(status):
    client = _mock_client()
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://example.com/a")
    )
    client._ingest_observed.side_effect = httpx.HTTPStatusError(
        "err", request=response.request, response=response
    )
    with pytest.raises(TransientError):
        await run_job(client, _job())


async def test_connect_error_classified_transient():
    client = _mock_client()
    client._ingest_observed.side_effect = httpx.ConnectError("boom")
    with pytest.raises(TransientError):
        await run_job(client, _job())


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: TimeoutError("slow"),
        lambda: OSError("io"),
    ],
)
async def test_timeout_and_io_errors_classified_transient(exc_factory):
    """TimeoutError / OSError both classify as transient. Without this
    branch they'd fall through to the generic "unexpected" wrapper."""
    client = _mock_client()
    client._ingest_observed.side_effect = exc_factory()
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
async def test_transport_subclasses_classified_transient(exc_factory):
    """The classifier umbrellas on httpx.TransportError so every transport-
    layer subclass routes to TransientError, not the generic 'unexpected'
    fallback."""
    client = _mock_client()
    client._ingest_observed.side_effect = exc_factory()
    with pytest.raises(TransientError, match="network"):
        await run_job(client, _job())


async def test_unknown_exception_classified_transient():
    client = _mock_client()
    client._ingest_observed.side_effect = RuntimeError("???")
    with pytest.raises(TransientError):
        await run_job(client, _job())


def _hook_error(cause: BaseException):
    from haiku.rag.hooks import HookError

    try:
        raise HookError("pkg.Hook.before_ingest failed for file:///a.md") from cause
    except HookError as e:
        return e


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    response = httpx.Response(
        status, request=httpx.Request("GET", "https://example.com/a")
    )
    return httpx.HTTPStatusError("err", request=response.request, response=response)


class _ClientConnectError(Exception):
    """A client library's connect error."""


def _raised_from(outer: Exception, inner: BaseException) -> Exception:
    try:
        raise outer from inner
    except Exception as e:
        return e


def _client_connect_error(socket_error: BaseException) -> Exception:
    """Chained as httpx and httpx2 chain theirs: the transport's error is raised
    while handling the socket error, and the client's is raised from it."""
    try:
        try:
            try:
                raise socket_error
            except BaseException:
                raise _ClientConnectError("transport")
        except _ClientConnectError as transport:
            raise _ClientConnectError("client") from transport
    except _ClientConnectError as e:
        return e


@pytest.mark.parametrize(
    "cause",
    [
        ValueError("bad header"),
        RuntimeError("???"),
        _http_status_error(404),
        FileNotFoundError("glossary.csv"),
    ],
)
async def test_a_hook_failure_is_permanent(cause):
    client = _mock_client()
    client._ingest_observed.side_effect = _hook_error(cause)
    with pytest.raises(PermanentError) as raised:
        await run_job(client, _job())
    assert str(raised.value) == "pkg.Hook.before_ingest failed for file:///a.md"


@pytest.mark.parametrize(
    "cause",
    [
        TransientError("busy"),
        httpx.ConnectError("boom"),
        TimeoutError("slow"),
        _http_status_error(408),
        _http_status_error(429),
        _http_status_error(503),
        ConnectionResetError("reset"),
        ConnectionRefusedError("refused"),
        socket.gaierror("name resolution"),
        _client_connect_error(ConnectionRefusedError()),
        _client_connect_error(
            _raised_from(
                OSError("all connection attempts failed"),
                ExceptionGroup(
                    "connect", [ConnectionRefusedError(), ConnectionRefusedError()]
                ),
            )
        ),
    ],
)
async def test_a_hook_failure_caused_by_a_transient_error_is_retried(cause):
    client = _mock_client()
    client._ingest_observed.side_effect = _hook_error(cause)
    with pytest.raises(TransientError) as raised:
        await run_job(client, _job())
    assert str(raised.value) == "pkg.Hook.before_ingest failed for file:///a.md"


async def test_directory_result_classified_permanent():
    client = _mock_client()
    client._ingest_observed.return_value = [
        Document(id="a", content="", uri="u1"),
        Document(id="b", content="", uri="u2"),
    ]
    with pytest.raises(PermanentError, match="directory"):
        await run_job(client, _job())


async def test_existing_permanent_error_passes_through_unchanged():
    client = _mock_client()
    sentinel = PermanentError("explicit")
    client._ingest_observed.side_effect = sentinel
    with pytest.raises(PermanentError) as excinfo:
        await run_job(client, _job())
    assert excinfo.value is sentinel


async def test_file_not_found_classified_as_permanent():
    """A deleted file should go straight to the DLQ, not retry."""
    client = _mock_client()
    client._ingest_observed.side_effect = FileNotFoundError("gone")
    with pytest.raises(PermanentError, match="file not found"):
        await run_job(client, _job())


async def test_permission_error_classified_as_permanent():
    """An unreadable file should go straight to the DLQ, not retry."""
    client = _mock_client()
    client._ingest_observed.side_effect = PermissionError("no access")
    with pytest.raises(PermanentError, match="permission denied"):
        await run_job(client, _job())


@pytest.mark.parametrize("exc_class", [IsADirectoryError, NotADirectoryError])
async def test_directory_errors_classified_as_permanent(exc_class):
    """Pointing at a directory instead of a file should DLQ immediately."""
    client = _mock_client()
    client._ingest_observed.side_effect = exc_class("not a file")
    with pytest.raises(PermanentError, match="path error"):
        await run_job(client, _job())


async def test_file_too_large_classified_as_permanent():
    client = _mock_client()
    client._ingest_observed.side_effect = FileTooLargeError("too big")
    with pytest.raises(PermanentError, match="too big"):
        await run_job(client, _job())


@pytest.mark.parametrize(
    "exc_class",
    [
        PermissionDeniedError,
        UnauthenticatedError,
        UnknownConfigurationKeyError,
        InvalidPathError,
    ],
)
async def test_obstore_config_errors_classified_permanent(exc_class):
    client = _mock_client()
    client._ingest_observed.side_effect = exc_class("bad config")
    with pytest.raises(PermanentError, match="object store"):
        await run_job(client, _job())


@pytest.mark.parametrize(
    "exc_class",
    [GenericError, JoinError],
)
async def test_other_obstore_errors_classified_transient(exc_class):
    """Only the credential/configuration errors are permanent; umbrellaing on
    obstore's BaseError would sweep up retryable failures too."""
    client = _mock_client()
    client._ingest_observed.side_effect = exc_class("upstream hiccup")
    with pytest.raises(TransientError):
        await run_job(client, _job())


async def test_conversion_deadline_is_permanent_and_fatal_to_the_process():
    """The document that exceeded the deadline is dead, and the process that
    abandoned the shared converter cannot convert again."""
    from haiku.rag.converters.exceptions import ConversionTimeoutError

    client = _mock_client()
    client._ingest_observed.side_effect = ConversionTimeoutError(
        "took too long", converter_wedged=True
    )

    with pytest.raises(PermanentError) as excinfo:
        await run_job(client, _job())

    assert excinfo.value.fatal_to_process is True
    assert excinfo.value.conversion_stalled is True


async def test_conversion_deadline_on_its_own_converter_spares_the_process():
    """HTML and Markdown hold no shared converter, so the deadline kills the
    document only."""
    from haiku.rag.converters.exceptions import ConversionTimeoutError

    client = _mock_client()
    client._ingest_observed.side_effect = ConversionTimeoutError(
        "took too long", converter_wedged=False
    )

    with pytest.raises(PermanentError) as excinfo:
        await run_job(client, _job())

    assert excinfo.value.fatal_to_process is False
    # Still tombstoned: the same bytes stall again, and each attempt leaves
    # another daemon thread and its memory behind.
    assert excinfo.value.conversion_stalled is True


async def test_wedged_converter_is_transient_and_never_kills_the_document():
    """A document refused by a converter another document wedged is not at
    fault, so it must not be dead-lettered."""
    from haiku.rag.converters.exceptions import ConverterWedgedError

    client = _mock_client()
    client._ingest_observed.side_effect = ConverterWedgedError("restart the process")

    with pytest.raises(TransientError) as excinfo:
        await run_job(client, _job())

    # classified on its type, not swept up by the unknown-error fallthrough
    assert not str(excinfo.value).startswith("unexpected:")


async def test_upsert_passes_the_observed_revision():
    """Discovery already read the revision, so ingestion need not ask again."""
    client = _mock_client()
    client._ingest_observed.return_value = Document(
        id="doc-1", content="x", uri="https://example.com/a.pdf", metadata={}
    )

    await run_job(client, _job(revision="etag-2"))

    kwargs = client._ingest_observed.await_args.kwargs
    assert kwargs["observed_revision"] == "etag-2"


async def test_upsert_without_a_revision_passes_none():
    client = _mock_client()
    client._ingest_observed.return_value = Document(
        id="doc-1", content="x", uri="https://example.com/a.pdf", metadata={}
    )

    await run_job(client, _job(revision=None))

    kwargs = client._ingest_observed.await_args.kwargs
    assert kwargs["observed_revision"] is None


async def test_manifest_replay_still_revalidates_before_ingesting():
    """The manifest's revision is frozen, so it is checked against the source
    even though the hint is passed on."""
    client = _mock_client()
    client._ingest_observed.return_value = Document(
        id="doc-1", content="x", uri="https://example.com/a.pdf", metadata={}
    )
    source = _StubSource("src", "etag-2")
    job = _job(revision="etag-2", extra={"_manifest": {"generated_at": "now"}})

    await run_job(client, job, sources=[source])

    assert source.head_calls == 1
    kwargs = client._ingest_observed.await_args.kwargs
    assert kwargs["observed_revision"] == "etag-2"


@pytest.mark.vcr()
@pytest.mark.usefixtures("docling_local_models")
async def test_upsert_calls_the_provider_for_pdf_attachments(tmp_path, temp_db_path):
    """End to end through a real client: the source's provider is called for
    the fetched PDF and each attachment, with the job's source id, and only
    the fetched document is attributed to the source."""
    from haiku.rag.client.documents import parent_uri_filter
    from haiku.rag.sources.fs import FSSource
    from tests.conftest import build_pdf

    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(build_pdf([("a.txt", b"alpha notes")]))
    source = FSSource(root=tmp_path, source_id="src")
    provider = _MetadataProvider({"team": "docs"})

    async with HaikuRAG(temp_db_path, create=True) as client:
        result = await run_job(
            client,
            _job(uri=pdf_path.as_uri()),
            sources=[source],
            metadata_providers={"src": provider},
        )
        parent = await client.get_document_by_uri(pdf_path.as_uri())
        assert parent is not None
        (child,) = await client.list_documents(filter=parent_uri_filter(parent.uri))

    assert result.document_id == parent.id
    assert result.revision == parent.metadata["source_revision"]
    assert result.content_hash == parent.metadata["md5"]
    assert parent.metadata["team"] == "docs"
    assert parent.metadata["source_id"] == "src"
    assert child.uri == f"{parent.uri}#attachment=a.txt"
    assert child.metadata["team"] == "docs"
    assert child.metadata["source"] == "src"
    assert "source_id" not in child.metadata


async def test_delete_keeps_a_document_transferred_while_the_job_ran(temp_db_path):
    """Ownership moving to another source after the job read the document,
    and before the delete takes the write lock, keeps the document."""
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.config import get_config
    from haiku.rag.store.models.chunk import Chunk

    dim = get_config().embeddings.model.vector_dim
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.import_document(
            DoclingDocument(name="d"),
            [Chunk(content="body", embedding=[0.1] * dim, order=0)],
            uri="mem://contested",
        )
        assert doc.id is not None
        await client.set_document_source([doc.id], "src")
        read_by_uri = client.get_document_by_uri

        async def _read_then_transfer(uri):
            stale = await read_by_uri(uri)
            repository = client._session.document_repository
            current = await repository.get_by_id(doc.id, include_blobs=True)
            assert current is not None
            current.metadata = {**current.metadata, "source_id": "other"}
            await repository.update(current)
            return stale

        client.get_document_by_uri = _read_then_transfer

        with capture_logs(_PIPELINE_LOGGER, logging.WARNING) as records:
            result = await run_job(client, _job(op=JobOp.DELETE, uri="mem://contested"))

        assert result.deleted is False
        assert [r.getMessage() for r in records] == [
            f"Not deleting {doc.id} (mem://contested) for source 'src': it moved "
            "to another source or was removed while the job ran"
        ]
        kept = await client.get_document_by_id(doc.id)
        assert kept is not None
        assert kept.metadata["source_id"] == "other"


@pytest.mark.parametrize("kept", [["https://example.com/kept.md"], []])
async def test_delete_removes_a_url_taken_out_of_the_http_source(kept):
    """The server still answers the removed URL with an ETag, which the probe
    must not read as the resource coming back."""
    from haiku.rag.sources.http import HTTPSource

    removed = "https://example.com/removed.md"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"etag": '"e1"'})
    )
    source = HTTPSource(source_id="src", urls=kept, transport=transport)
    client = _mock_client()
    client.get_document_by_uri.return_value = Document(
        id="doc-9", content="", uri=removed, metadata={"source_id": "src"}
    )

    result = await run_job(client, _job(op=JobOp.DELETE, uri=removed), sources=[source])

    assert result.deleted is True
    client.delete_document.assert_awaited_once_with("doc-9", source_id="src")


class _ThrowingIngestHook(Hook):
    async def after_ingest(self, client, event):
        raise RuntimeError("ingest hook boom")


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_after_ingest_hook_failure_does_not_fail_job(
    temp_db_path, tmp_path, caplog
):
    """A throwing post-commit hook must not turn a committed ingest into a
    failed job: run_job returns a successful JobResult instead of raising
    into _classify."""
    file_path = tmp_path / "doc.md"
    file_path.write_text("hello")

    fs = FSSource(root=tmp_path, source_id="src")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [_ThrowingIngestHook()]

        with caplog.at_level(logging.ERROR, logger="haiku.rag.hooks"):
            result = await run_job(client, _job(uri=str(file_path)), sources=[fs])

        assert result.document_id is not None
        assert result.deleted is False
        assert any("after_ingest" in r.message for r in caplog.records)


class _RejectHook(Hook):
    def __init__(self, suffix: str):
        self.suffix = suffix

    async def before_ingest(self, client, request):
        from haiku.rag.hooks import DocumentRejected

        if request.uri.endswith(self.suffix):
            raise DocumentRejected("draft")
        return request


async def test_a_rejected_document_is_a_permanent_failure(temp_db_path, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("hello")
    fs = FSSource(root=tmp_path, source_id="src")

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [_RejectHook("doc.md")]
        with pytest.raises(PermanentError) as raised:
            await run_job(client, _job(uri=path.as_uri()), sources=[fs])

    assert str(raised.value) == f"{path.as_uri()} rejected: draft"


@pytest.mark.vcr()
@pytest.mark.usefixtures("docling_local_models")
async def test_a_rejected_attachment_fails_its_parent_until_retried(
    tmp_path, temp_db_path
):
    """The parent and the earlier sibling stay stored, the parent without the
    md5 that would short-circuit a retry, and a retry once the hook accepts
    the attachment completes the set."""
    from haiku.rag.client.documents import parent_uri_filter
    from tests.conftest import build_pdf

    pdf_path = tmp_path / "parent.pdf"
    pdf_path.write_bytes(build_pdf([("a.txt", b"alpha notes"), ("b.txt", b"beta")]))
    fs = FSSource(root=tmp_path, source_id="src")
    child_b = f"{pdf_path.as_uri()}#attachment=b.txt"

    async with HaikuRAG(temp_db_path, create=True) as client:
        client._hooks = [_RejectHook("#attachment=b.txt")]
        with pytest.raises(PermanentError) as raised:
            await run_job(client, _job(uri=pdf_path.as_uri()), sources=[fs])
        parent = await client.get_document_by_uri(pdf_path.as_uri())
        assert parent is not None
        children = await client.list_documents(filter=parent_uri_filter(parent.uri))
        assert [c.uri for c in children] == [f"{pdf_path.as_uri()}#attachment=a.txt"]
        assert "md5" not in parent.metadata

        client._hooks = []
        result = await run_job(client, _job(uri=pdf_path.as_uri()), sources=[fs])
        children = await client.list_documents(filter=parent_uri_filter(parent.uri))
        parent = await client.get_document_by_uri(pdf_path.as_uri())

    assert str(raised.value) == f"{child_b} rejected: draft"
    assert sorted(c.uri for c in children) == sorted(
        [f"{pdf_path.as_uri()}#attachment=a.txt", child_b]
    )
    assert parent is not None
    assert result.content_hash == parent.metadata["md5"]

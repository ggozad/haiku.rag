import asyncio
import logging
import socket
from collections.abc import Iterator
from contextlib import nullcontext
from typing import TYPE_CHECKING

import httpx
from obstore.exceptions import (
    InvalidPathError,
    PermissionDeniedError,
    UnauthenticatedError,
    UnknownConfigurationKeyError,
)
from pydantic import BaseModel

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.converters.exceptions import (
    ConversionTimeoutError,
    ConverterWedgedError,
)
from haiku.rag.hooks import DocumentRejected, HookError
from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.models import Job, JobOp
from haiku.rag.sources.base import FileTooLargeError
from haiku.rag.sources.http import HTTPSource, is_failure_marker
from haiku.rag.sources.registry import resolve_configured_source
from haiku.rag.telemetry import attach_context, logfire
from haiku.rag.utils.http import is_transient_status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from haiku.rag.client import HaikuRAG
    from haiku.rag.ingester.metadata import MetadataProvider
    from haiku.rag.sources.base import Source

logger = logging.getLogger(__name__)


class JobResult(BaseModel):
    """What the worker needs after a successful job: enough metadata to
    update sync_state. document_id is None for DELETE ops."""

    document_id: str | None = None
    revision: str | None = None
    content_hash: str | None = None
    deleted: bool = False
    resource_restored: bool = False


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """The exceptions behind `exc`, including every member of a group, such as
    anyio's group of one failed connection per resolved address."""
    seen: set[int] = set()
    pending = [exc.__cause__ or exc.__context__]
    while pending:
        cause = pending.pop()
        if cause is None or id(cause) in seen:
            continue
        seen.add(id(cause))
        yield cause
        pending.append(cause.__cause__ or cause.__context__)
        if isinstance(cause, BaseExceptionGroup):
            pending.extend(cause.exceptions)


def _is_transient_cause(cause: BaseException) -> bool:
    # httpx's and httpx2's connect and timeout errors chain down to these
    # builtins.
    if isinstance(
        cause,
        TransientError
        | httpx.TransportError
        | ConnectionError
        | TimeoutError
        | socket.gaierror,
    ):
        return True
    return isinstance(cause, httpx.HTTPStatusError) and is_transient_status(
        cause.response.status_code
    )


def _classify(exc: BaseException) -> Exception:
    """Wrap an unclassified exception into Permanent or Transient. Already-
    classified errors pass through unchanged."""
    if isinstance(exc, PermanentError | TransientError):
        return exc

    # UnsupportedSourceError is the typed signal from client/* that the
    # source will never ingest successfully on a retry (bad URI scheme,
    # missing file, unsupported extension, etc.).
    if isinstance(exc, UnsupportedSourceError | FileTooLargeError):
        return PermanentError(str(exc))

    if isinstance(exc, DocumentRejected):
        return PermanentError(str(exc), rejected=True)

    if isinstance(exc, HookError):
        # A hook's own bug repeats on every retry. Its call to a service that
        # failed transiently does not.
        if any(_is_transient_cause(cause) for cause in _causes(exc)):
            return TransientError(str(exc))
        return PermanentError(str(exc))

    if isinstance(exc, ConversionTimeoutError):
        # The document exceeded the deadline; a fresh process would very likely
        # stall on it again. `converter_wedged` decides whether this process can
        # still convert anything else.
        return PermanentError(
            f"conversion deadline: {exc}",
            conversion_stalled=True,
            fatal_to_process=exc.converter_wedged,
        )

    if isinstance(exc, ConverterWedgedError):
        # Another document wedged the converter. This one is not at fault and
        # must survive to be retried after the restart.
        return TransientError(f"converter wedged: {exc}")

    if isinstance(exc, ValueError):
        # Some downstream libraries (e.g. docling) raise plain ValueError
        # for "couldn't parse this file"; default to transient so the queue
        # retries up to max_attempts in case the issue is intermittent.
        return TransientError(str(exc))

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        # 401/403/404/410 are unrecoverable without operator action; 408/429/5xx
        # are transient. Everything else in 4xx is treated as permanent — better
        # to DLQ a misconfigured URL than to retry it forever.
        if is_transient_status(status):
            return TransientError(f"HTTP {status}: {exc}")
        return PermanentError(f"HTTP {status}: {exc}")

    if isinstance(exc, httpx.TransportError):
        # Umbrella for ConnectError, NetworkError, TimeoutException, ProtocolError,
        # ProxyError — every transport-layer failure that's worth retrying.
        return TransientError(f"network: {exc}")

    if isinstance(
        exc,
        PermissionDeniedError
        | UnauthenticatedError
        | UnknownConfigurationKeyError
        | InvalidPathError,
    ):
        # These subclass neither OSError nor httpx, so without this branch
        # they'd fall through to the transient default and retry the whole
        # backoff ladder. A missing object arrives as a builtin
        # FileNotFoundError instead, handled below.
        return PermanentError(f"object store: {exc}")

    if isinstance(exc, FileNotFoundError):
        return PermanentError(f"file not found: {exc}")

    if isinstance(exc, PermissionError):
        return PermanentError(f"permission denied: {exc}")

    if isinstance(exc, IsADirectoryError | NotADirectoryError):
        return PermanentError(f"path error: {exc}")

    if isinstance(exc, asyncio.TimeoutError | TimeoutError | OSError):
        return TransientError(f"timeout/io: {exc}")

    # Unknown errors default to transient — retry up to max_attempts gives the
    # operator visibility into the failure mode without dropping data on the
    # first hiccup.
    return TransientError(f"unexpected: {exc!r}")


async def run_job(
    client: "HaikuRAG",
    job: Job,
    *,
    sources: list["Source"] | None = None,
    metadata_providers: "Mapping[str, MetadataProvider] | None" = None,
) -> JobResult:
    """Execute the work described by `job`. `sources` is the list of
    configured Source adapters; the client looks up `job.source_id`
    against it via `resolve_configured_source` so workers reuse the
    authenticated/pre-configured fetch context the pollers used at
    discovery. `metadata_providers` maps source_id to a provider whose
    output is attached as document metadata on UPSERT. Raises PermanentError
    or TransientError; the worker uses that to decide dead vs retry."""
    extra = job.extra or {}
    parent_ctx = extra.get("_otel")
    attach = attach_context(parent_ctx) if parent_ctx else nullcontext()

    with (
        attach,
        logfire.span(
            "ingester.job",
            job_id=job.id,
            source_id=job.source_id,
            uri=job.uri,
            op=job.op.value,
            attempt=job.attempts,
        ),
    ):
        try:
            manifest_context = extra.get("_manifest")
            if job.op is JobOp.DELETE:
                # An atomic-rename save can let a spurious DELETE win the
                # enqueue race while the file is mid-rewrite. If the resource
                # is already back, skip the delete (it would blackhole a live
                # document) and let the next sweep re-ingest it. Manifest
                # replay intentionally follows the frozen dry-run changeset.
                if manifest_context is None:
                    try:
                        source = resolve_configured_source(
                            job.uri, job.source_id, sources
                        )
                        restored = await source.head(job.uri) is not None
                    except Exception:
                        restored = False
                    if restored:
                        return JobResult(resource_restored=True)
                doc = await client.get_document_by_uri(job.uri)
                if doc is None or doc.id is None:
                    return JobResult(deleted=True)
                owner = doc.metadata.get("source_id")
                if owner != job.source_id:
                    logger.warning(
                        "Not deleting %s (%s) for source %r: it belongs to %s",
                        doc.id,
                        doc.uri,
                        job.source_id,
                        f"source {owner!r}" if owner else "no source",
                    )
                    return JobResult(deleted=False)
                deleted = await client.delete_document(doc.id, source_id=job.source_id)
                if not deleted:
                    logger.warning(
                        "Not deleting %s (%s) for source %r: it moved to another "
                        "source or was removed while the job ran",
                        doc.id,
                        doc.uri,
                        job.source_id,
                    )
                return JobResult(deleted=deleted)

            if manifest_context is not None and job.revision is not None:
                source = resolve_configured_source(job.uri, job.source_id, sources)
                current_revision = await source.head(job.uri)
                # An HTTP failure marker reads as None, and the fetch decides.
                failed = isinstance(source, HTTPSource) and is_failure_marker(
                    job.revision
                )
                if current_revision != job.revision and not failed:
                    raise PermanentError(
                        "manifest revision is stale for "
                        f"{job.uri}: expected {job.revision!r}, "
                        f"current {current_revision!r}"
                    )

            result = await client._ingest_observed(
                job.uri,
                sources=sources,
                source_id=job.source_id,
                metadata_provider=(metadata_providers or {}).get(job.source_id),
                # The revision observed when the UPSERT was enqueued.
                observed_revision=job.revision,
            )
            # Directory ingestion returns list[Document] — workers ingest single
            # resources, so a list here is a programming error in the caller.
            if isinstance(result, list):
                raise PermanentError(
                    f"Job {job.id} resolved to a directory; queue jobs must "
                    f"reference a single document URI."
                )

            metadata = result.metadata or {}
            return JobResult(
                document_id=result.id,
                revision=metadata.get("source_revision"),
                content_hash=metadata.get("md5"),
            )
        except Exception as exc:
            # CancelledError, KeyboardInterrupt, SystemExit are BaseException
            # subclasses; they signal the runtime is shutting us down, not a
            # job-level failure, so we let them propagate untouched.
            raise _classify(exc) from exc

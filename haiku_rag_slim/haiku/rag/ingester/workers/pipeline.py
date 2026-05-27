import asyncio
from contextlib import nullcontext
from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel

from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.models import Job, JobOp
from haiku.rag.telemetry import attach_context, logfire

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.ingester.sources.base import Source


class JobResult(BaseModel):
    """What the worker needs after a successful job: enough metadata to
    update sync_state. document_id is None for DELETE ops."""

    document_id: str | None = None
    revision: str | None = None
    content_hash: str | None = None
    deleted: bool = False


def _classify(exc: BaseException) -> Exception:
    """Wrap an unclassified exception into Permanent or Transient. Already-
    classified errors pass through unchanged."""
    if isinstance(exc, PermanentError | TransientError):
        return exc

    # UnsupportedSourceError is the typed signal from client/* that the
    # source will never ingest successfully on a retry (bad URI scheme,
    # missing file, unsupported extension, etc.).
    if isinstance(exc, UnsupportedSourceError):
        return PermanentError(str(exc))

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
        if status in (408, 429) or status >= 500:
            return TransientError(f"HTTP {status}: {exc}")
        return PermanentError(f"HTTP {status}: {exc}")

    if isinstance(exc, httpx.TransportError):
        # Umbrella for ConnectError, NetworkError, TimeoutException, ProtocolError,
        # ProxyError — every transport-layer failure that's worth retrying.
        return TransientError(f"network: {exc}")

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
) -> JobResult:
    """Execute the work described by `job`. `sources` is the list of
    configured Source adapters — `resolve_fetcher` prefers them over
    URI-scheme adhoc adapters so workers reuse the same authenticated /
    pre-configured fetch context the pollers used at discovery. Raises
    PermanentError or TransientError; the worker uses that to decide
    dead vs retry."""
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
            if job.op is JobOp.DELETE:
                doc = await client.get_document_by_uri(job.uri)
                if doc is not None and doc.id is not None:
                    await client.delete_document(doc.id)
                return JobResult(deleted=True)

            result = await client.create_document_from_source(
                job.uri,
                sources=sources,
                source_id=job.source_id,
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

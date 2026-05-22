import asyncio
from contextlib import nullcontext
from typing import TYPE_CHECKING

import httpx
import logfire
from pydantic import BaseModel

from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.models import Job, JobOp

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG

# ValueError messages from create_document_from_source that mean "this job
# will never succeed". Anything else from ValueError defaults to transient.
_PERMANENT_VALUE_MARKERS = (
    "Unsupported file extension",
    "Unsupported content type",
    "Invalid S3 URI",
    "File does not exist",
    "uri override is not supported",
)


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

    if isinstance(exc, ValueError):
        message = str(exc)
        if any(marker in message for marker in _PERMANENT_VALUE_MARKERS):
            return PermanentError(message)
        return TransientError(message)

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        # 401/403/404/410 are unrecoverable without operator action; 408/429/5xx
        # are transient. Everything else in 4xx is treated as permanent — better
        # to DLQ a misconfigured URL than to retry it forever.
        if status in (408, 429) or status >= 500:
            return TransientError(f"HTTP {status}: {exc}")
        return PermanentError(f"HTTP {status}: {exc}")

    if isinstance(
        exc,
        httpx.ConnectError | httpx.ReadTimeout | httpx.WriteTimeout | httpx.PoolTimeout,
    ):
        return TransientError(f"network: {exc}")

    if isinstance(exc, asyncio.TimeoutError | TimeoutError | OSError):
        return TransientError(f"timeout/io: {exc}")

    # Unknown errors default to transient — retry up to max_attempts gives the
    # operator visibility into the failure mode without dropping data on the
    # first hiccup.
    return TransientError(f"unexpected: {exc!r}")


async def run_job(client: "HaikuRAG", job: Job) -> JobResult:
    """Execute the work described by `job`. Raises PermanentError or
    TransientError; the worker uses that to decide dead vs retry."""
    extra = job.extra or {}
    storage_options = extra.get("storage_options")
    user_metadata = extra.get("metadata", {})
    # Restore the poller's trace context (if any) so the job span nests
    # under the `ingester.poller.sweep` that enqueued it.
    parent_ctx = extra.get("_otel")
    attach = logfire.attach_context(parent_ctx) if parent_ctx else nullcontext()

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
                metadata=user_metadata,
                storage_options=storage_options,
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

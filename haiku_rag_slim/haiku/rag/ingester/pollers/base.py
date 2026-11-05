import asyncio
import logging
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from haiku.rag.circuit_breaker import CircuitBreaker
from haiku.rag.config import SourceConfig
from haiku.rag.ingester.batch import BatchChange, BatchSourceSummary
from haiku.rag.ingester.exceptions import EmptyListingError
from haiku.rag.ingester.queue.models import JobOp, JobStatus, SyncRow
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.sources.base import (
    Source,
    SourceEvent,
    SourceEventKind,
)
from haiku.rag.telemetry import get_context, logfire
from haiku.rag.utils.sql import build_uri_filter

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG

logger = logging.getLogger(__name__)

_STAGGER_FRACTION = 0.25

# URIs per ownership lookup, bounding the filter's length.
OWNERSHIP_BATCH = 500


def _enqueue_extra() -> dict | None:
    """Per-job context the worker can't reconstruct from config alone.
    Currently only the active logfire trace carrier so `ingester.job`
    nests under the sweep/watch span that enqueued it. Connection details
    (headers, auth, storage_options) come from the configured Source
    instance the worker resolves at run time."""
    carrier = get_context()
    return {"_otel": dict(carrier)} if carrier else None


def max_attempts(cfg: SourceConfig, default: int) -> int:
    return cfg.retry.max_attempts if cfg.retry is not None else default


class BasePoller:
    """Shared lifecycle: build a discover() coroutine + process its events
    into queue jobs and sync_state updates. Subclasses provide the loop
    (FS uses watchfiles + initial discover; periodic uses sleep+discover)."""

    def __init__(
        self,
        *,
        source: Source,
        config: SourceConfig,
        job_repo: JobRepo,
        sync_repo: SyncStateRepo,
        breaker: CircuitBreaker | None = None,
        default_max_attempts: int = 5,
    ):
        self.source = source
        self.config = config
        self._jobs = job_repo
        self._sync = sync_repo
        self._breaker = breaker or CircuitBreaker(config.circuit_breaker)
        self._stop = asyncio.Event()
        # Held for a sweep, a watch event's enqueue and a retry, so `stop`
        # returns only once none of them can still enqueue, and a retry never
        # overlaps discovery.
        self._sweep_lock = asyncio.Lock()
        self._last_polled_at: datetime | None = None
        self._last_skip_reason: str | None = None
        self._last_error: str | None = None
        self._last_error_at: datetime | None = None
        self._unsupported_files: dict[str, int] = {}
        self._default_max_attempts = default_max_attempts
        # The database receiving the source's documents. Unset on a dry-run,
        # which opens no store.
        self.documents: HaikuRAG | None = None

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def last_polled_at(self) -> datetime | None:
        return self._last_polled_at

    @property
    def is_circuit_open(self) -> bool:
        return self._breaker.is_open

    @property
    def last_skip_reason(self) -> str | None:
        """Reason the most recent sweep attempt skipped (e.g. "pending_work",
        "circuit_open"), or None when the most recent attempt actually polled.
        Cleared on the next successful sweep."""
        return self._last_skip_reason

    @property
    def last_error(self) -> str | None:
        """The error of the most recent failed discovery, None once a sweep
        succeeds after it."""
        return self._last_error

    @property
    def last_error_at(self) -> datetime | None:
        return self._last_error_at

    @property
    def unsupported_files(self) -> dict[str, int]:
        """Files the last successful sweep skipped for their extension, by
        extension. Empty for a source that does not count them."""
        return self._unsupported_files

    def _record_unsupported_files(self) -> None:
        unsupported = dict(getattr(self.source, "unsupported_files", {}))
        if unsupported and unsupported != self._unsupported_files:
            ranked = sorted(unsupported.items(), key=lambda item: -item[1])
            logger.info(
                "Skipped %d files with unsupported extensions in %s: %s",
                sum(unsupported.values()),
                self.source_id,
                ", ".join(f"{ext or 'no extension'} ({n})" for ext, n in ranked),
            )
        self._unsupported_files = unsupported

    def _record_error(self, exc: Exception) -> None:
        self._last_error = str(exc) or type(exc).__name__
        self._last_error_at = datetime.now(UTC)

    def _clear_error(self) -> None:
        self._last_error = None
        self._last_error_at = None

    async def run(self) -> None:  # pragma: no cover - subclasses override
        raise NotImplementedError

    async def stop(self) -> None:
        self._stop.set()
        async with self._sweep_lock:
            pass

    async def _stagger_start(self) -> bool:
        """Sleep a random fraction of the interval so pollers sharing an
        interval don't sweep in lockstep. Returns True if stop was
        signalled during the wait."""
        jitter = random.uniform(0, self.config.poll_interval_s * _STAGGER_FRACTION)
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=jitter)
            return True
        except TimeoutError:
            return False

    async def _sweep_once(self, *, skip_if_pending: bool = False) -> bool:
        """One discover() sweep. Returns True on success, False if the poller
        is stopped, the breaker is open, the sweep failed (and was recorded),
        or `skip_if_pending` is set and the source has a queued or claimed
        job."""
        async with self._sweep_lock:
            if self._stop.is_set():
                return False
            return await self._sweep(skip_if_pending=skip_if_pending)

    async def retry_failed(self, statuses: set[JobStatus]) -> int | None:
        """Re-queue this source's latest failures in `statuses` with no
        discovery running. Returns how many, or None once the poller stopped."""
        async with self._sweep_lock:
            if self._stop.is_set():
                return None
            return await self._jobs.retry_failed(statuses, source_ids=[self.source_id])

    async def _sweep(self, *, skip_if_pending: bool = False) -> bool:
        if self._breaker.is_open:
            self._last_skip_reason = "circuit_open"
            logger.debug(
                "Skipping discover() — circuit breaker open for %s", self.source_id
            )
            return False
        with logfire.span("ingester.poller.sweep", source_id=self.source_id) as span:
            if skip_if_pending and await self._jobs.has_pending(self.source_id):
                # A one-shot sweep runs once: an event deduplicated against a
                # live job would never be discovered again.
                self._last_skip_reason = "pending_work"
                span.set_attribute("skipped", True)
                span.set_attribute("skip_reason", "pending_work")
                logger.debug(
                    "Skipping discover() — %s has pending work in the queue",
                    self.source_id,
                )
                return False
            try:
                started = datetime.now(UTC)
                revisions = await self._sync.get_revision_snapshot(self.source_id)
                known = await self._sync.list_known_uris(self.source_id)
                counts = {
                    SourceEventKind.UPSERT: 0,
                    SourceEventKind.DELETE: 0,
                    SourceEventKind.UNCHANGED: 0,
                }
                enqueued = {SourceEventKind.UPSERT: 0, SourceEventKind.DELETE: 0}
                sync_batch: list[SyncRow] = []
                deletes: list[SourceEvent] = []
                async for event in self.source.discover(
                    since=revisions, known_uris=known
                ):
                    counts[event.kind] += 1
                    if event.kind is SourceEventKind.DELETE:
                        deletes.append(event)
                    elif await self._handle_event(event, sync_batch):
                        enqueued[event.kind] += 1
                await self._check_listing(
                    listed=counts[SourceEventKind.UPSERT]
                    + counts[SourceEventKind.UNCHANGED],
                    deletes=[event.uri for event in deletes],
                )
                for event in deletes:
                    if await self._handle_event(event, sync_batch):
                        enqueued[SourceEventKind.DELETE] += 1
                await self._sync.batch_upsert(sync_batch)
                self._record_unsupported_files()
                self._breaker.record_success()
                self._last_polled_at = started
                self._last_skip_reason = None
                self._clear_error()
                span.set_attribute("upsert", counts[SourceEventKind.UPSERT])
                span.set_attribute("delete", counts[SourceEventKind.DELETE])
                span.set_attribute("unchanged", counts[SourceEventKind.UNCHANGED])
                span.set_attribute("enqueued", sum(enqueued.values()))
                if any(enqueued.values()):
                    logger.info(
                        "Swept %s: %d upsert, %d delete, %d unchanged",
                        self.source_id,
                        enqueued[SourceEventKind.UPSERT],
                        enqueued[SourceEventKind.DELETE],
                        counts[SourceEventKind.UNCHANGED],
                    )
                return True
            except Exception as exc:
                self._breaker.record_failure()
                self._record_error(exc)
                span.set_attribute(
                    "consecutive_failures", self._breaker.consecutive_failures
                )
                span.record_exception(exc)
                logger.exception(
                    "discover() failed for %s (consecutive=%d): %s",
                    self.source_id,
                    self._breaker.consecutive_failures,
                    exc,
                )
                return False

    async def _dry_run_once(self) -> tuple[bool, BatchSourceSummary, list[BatchChange]]:
        """Collect what one discover() sweep would enqueue without writing
        jobs or sync_state."""
        summary = BatchSourceSummary(source_id=self.source_id)
        changes: list[BatchChange] = []
        if self._breaker.is_open:
            self._last_skip_reason = "circuit_open"
            logger.debug(
                "Skipping dry-run discover() — circuit breaker open for %s",
                self.source_id,
            )
            return False, summary, changes
        with logfire.span("ingester.poller.dry_run", source_id=self.source_id) as span:
            if await self._jobs.has_pending(self.source_id):
                self._last_skip_reason = "pending_work"
                span.set_attribute("skipped", True)
                span.set_attribute("skip_reason", "pending_work")
                logger.debug(
                    "Skipping dry-run discover() — %s has pending work in the queue",
                    self.source_id,
                )
                return False, summary, changes
            try:
                started = datetime.now(UTC)
                revisions = await self._sync.get_revision_snapshot(self.source_id)
                known = await self._sync.list_known_uris(self.source_id)
                async for event in self.source.discover(
                    since=revisions, known_uris=known
                ):
                    if event.kind is SourceEventKind.UPSERT:
                        summary.upsert_count += 1
                        changes.append(
                            BatchChange(
                                op=JobOp.UPSERT,
                                source_id=event.source_id,
                                uri=event.uri,
                                revision=event.revision,
                                discovered_at=event.discovered_at,
                            )
                        )
                    elif event.kind is SourceEventKind.UNCHANGED:
                        summary.unchanged_count += 1
                    elif (
                        event.kind is SourceEventKind.DELETE
                    ):  # pragma: no branch - DELETE is last
                        if self.config.delete_orphans:
                            summary.delete_count += 1
                            changes.append(
                                BatchChange(
                                    op=JobOp.DELETE,
                                    source_id=event.source_id,
                                    uri=event.uri,
                                    revision=event.revision,
                                    discovered_at=event.discovered_at,
                                )
                            )
                        else:
                            summary.ignored_delete_count += 1
                await self._check_listing(
                    listed=summary.upsert_count + summary.unchanged_count,
                    deletes=[c.uri for c in changes if c.op is JobOp.DELETE],
                )
                self._breaker.record_success()
                self._last_polled_at = started
                self._last_skip_reason = None
                self._clear_error()
                span.set_attribute("upsert", summary.upsert_count)
                span.set_attribute("delete", summary.delete_count)
                span.set_attribute("unchanged", summary.unchanged_count)
                return True, summary, changes
            except Exception as exc:
                self._breaker.record_failure()
                self._record_error(exc)
                span.set_attribute(
                    "consecutive_failures", self._breaker.consecutive_failures
                )
                span.record_exception(exc)
                logger.exception(
                    "dry-run discover() failed for %s (consecutive=%d): %s",
                    self.source_id,
                    self._breaker.consecutive_failures,
                    exc,
                )
                return False, BatchSourceSummary(source_id=self.source_id), []

    async def _check_listing(self, *, listed: int, deletes: list[str]) -> None:
        """Raise for a listing that returned nothing but deletes of stored
        documents owned by this source. Without a store, every delete counts."""
        if not deletes or listed or not self.config.delete_orphans:
            return
        if self.documents is None:
            raise EmptyListingError(self.source_id, len(deletes))
        owned = 0
        for start in range(0, len(deletes), OWNERSHIP_BATCH):
            batch = deletes[start : start + OWNERSHIP_BATCH]
            stored = await self.documents.list_documents(filter=build_uri_filter(batch))
            owned += sum(
                1 for doc in stored if doc.metadata.get("source_id") == self.source_id
            )
        if owned:
            raise EmptyListingError(self.source_id, owned, documents=True)

    async def _handle_event(
        self,
        event: SourceEvent,
        sync_batch: list[SyncRow],
    ) -> bool:
        """Enqueue the event's job and note its `sync_state` touch. Returns
        whether a job was enqueued."""
        if event.kind is SourceEventKind.UPSERT:
            job = await self._jobs.enqueue(
                event.source_id,
                event.uri,
                op=JobOp.UPSERT,
                revision=event.revision,
                max_attempts=max_attempts(self.config, self._default_max_attempts),
                extra=_enqueue_extra(),
            )
            # Don't write revision to sync_state here — the worker writes it
            # after a successful ingestion. last_seen_at gets bumped to keep
            # orphan detection accurate.
            sync_batch.append(SyncRow(event.source_id, event.uri, None, None, False))
            return job is not None
        if event.kind is SourceEventKind.UNCHANGED:
            # Touch last_seen_at without changing the stored revision, which a
            # worker may have written since the sweep took its snapshot.
            sync_batch.append(SyncRow(event.source_id, event.uri, None, None, False))
            return False
        if not self.config.delete_orphans:
            return False
        job = await self._jobs.enqueue(
            event.source_id,
            event.uri,
            op=JobOp.DELETE,
            revision=event.revision,
            max_attempts=max_attempts(self.config, self._default_max_attempts),
            extra=_enqueue_extra(),
        )
        return job is not None

import asyncio
import logging
import random
from datetime import UTC, datetime

from haiku.rag.circuit_breaker import CircuitBreaker
from haiku.rag.config import SourceConfig
from haiku.rag.ingester.batch import BatchChange, BatchSourceSummary
from haiku.rag.ingester.exceptions import EmptyListingError
from haiku.rag.ingester.queue.models import JobOp, SyncRow
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.sources.base import (
    Source,
    SourceEvent,
    SourceEventKind,
)
from haiku.rag.telemetry import get_context, logfire

logger = logging.getLogger(__name__)

_STAGGER_FRACTION = 0.25


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
        # Held for a whole sweep, whether the loop or a refresh request runs
        # it, so `stop` returns only once no sweep can still list or enqueue.
        self._sweep_lock = asyncio.Lock()
        self._last_polled_at: datetime | None = None
        self._last_skip_reason: str | None = None
        self._last_error: str | None = None
        self._last_error_at: datetime | None = None
        self._default_max_attempts = default_max_attempts

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

    async def _sweep_once(self) -> bool:
        """One discover() sweep. Returns True on success, False if the poller
        is stopped, the breaker is open, the source has pending work already
        queued, or the sweep failed (and was recorded)."""
        async with self._sweep_lock:
            if self._stop.is_set():
                return False
            return await self._sweep()

    async def _sweep(self) -> bool:
        if self._breaker.is_open:
            self._last_skip_reason = "circuit_open"
            logger.debug(
                "Skipping discover() — circuit breaker open for %s", self.source_id
            )
            return False
        with logfire.span("ingester.poller.sweep", source_id=self.source_id) as span:
            if await self._jobs.has_pending(self.source_id):
                # The unique index would dedupe a re-sweep into a saturated
                # queue anyway; skipping saves the listing round-trip
                # (PROPFIND / S3 LIST / FS walk) and keeps Logfire readable.
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
                sync_batch: list[SyncRow] = []
                deletes: list[SourceEvent] = []
                async for event in self.source.discover(
                    since=revisions, known_uris=known
                ):
                    counts[event.kind] += 1
                    if event.kind is SourceEventKind.DELETE:
                        deletes.append(event)
                    else:
                        await self._handle_event(event, sync_batch)
                self._check_listing(
                    listed=counts[SourceEventKind.UPSERT]
                    + counts[SourceEventKind.UNCHANGED],
                    deletes=len(deletes),
                )
                for event in deletes:
                    await self._handle_event(event, sync_batch)
                await self._sync.batch_upsert(sync_batch)
                self._breaker.record_success()
                self._last_polled_at = started
                self._last_skip_reason = None
                self._clear_error()
                span.set_attribute("upsert", counts[SourceEventKind.UPSERT])
                span.set_attribute("delete", counts[SourceEventKind.DELETE])
                span.set_attribute("unchanged", counts[SourceEventKind.UNCHANGED])
                if counts[SourceEventKind.UPSERT] or counts[SourceEventKind.DELETE]:
                    logger.info(
                        "Swept %s: %d upsert, %d delete, %d unchanged",
                        self.source_id,
                        counts[SourceEventKind.UPSERT],
                        counts[SourceEventKind.DELETE],
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
                    elif event.kind is SourceEventKind.DELETE:
                        if self.config.delete_orphans:
                            summary.delete_count += 1
                            changes.append(
                                BatchChange(
                                    op=JobOp.DELETE,
                                    source_id=event.source_id,
                                    uri=event.uri,
                                    revision=None,
                                    discovered_at=event.discovered_at,
                                )
                            )
                        else:
                            summary.ignored_delete_count += 1
                self._check_listing(
                    listed=summary.upsert_count + summary.unchanged_count,
                    deletes=summary.delete_count,
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

    def _check_listing(self, *, listed: int, deletes: int) -> None:
        """Raise for a listing that returned nothing but deletes."""
        if deletes and not listed and self.config.delete_orphans:
            raise EmptyListingError(self.source_id, deletes)

    async def _handle_event(
        self,
        event: SourceEvent,
        sync_batch: list[SyncRow],
    ) -> None:
        if event.kind is SourceEventKind.UPSERT:
            await self._jobs.enqueue(
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
        elif event.kind is SourceEventKind.UNCHANGED:
            # Touch last_seen_at without changing the stored revision.
            sync_batch.append(
                SyncRow(event.source_id, event.uri, event.revision, None, False)
            )
        elif event.kind is SourceEventKind.DELETE:
            if not self.config.delete_orphans:
                return
            await self._jobs.enqueue(
                event.source_id,
                event.uri,
                op=JobOp.DELETE,
                max_attempts=max_attempts(self.config, self._default_max_attempts),
                extra=_enqueue_extra(),
            )

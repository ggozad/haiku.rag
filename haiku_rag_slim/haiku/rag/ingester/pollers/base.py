import asyncio
import logging
import random
from datetime import UTC, datetime

from haiku.rag.config import SourceConfig
from haiku.rag.ingester.batch import BatchChange, BatchSourceSummary
from haiku.rag.ingester.pollers.circuit_breaker import CircuitBreaker
from haiku.rag.ingester.queue.models import JobOp, SyncRow
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.sources.base import (
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


def _max_attempts(cfg: SourceConfig, default: int) -> int:
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
        self._task: asyncio.Task | None = None
        self._last_polled_at: datetime | None = None
        self._last_skip_reason: str | None = None
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

    async def run(self) -> None:  # pragma: no cover - subclasses override
        raise NotImplementedError

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

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
        """One discover() sweep. Returns True on success, False if the
        breaker is open, the source has pending work already queued, or the
        sweep failed (and was recorded)."""
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
                revisions = await self._sync.get_revision_snapshot(self.source_id)
                known = await self._sync.list_known_uris(self.source_id)
                counts = {
                    SourceEventKind.UPSERT: 0,
                    SourceEventKind.DELETE: 0,
                    SourceEventKind.UNCHANGED: 0,
                }
                sync_batch: list[SyncRow] = []
                async for event in self.source.discover(
                    since=revisions, known_uris=known
                ):
                    counts[event.kind] += 1
                    await self._handle_event(event, sync_batch)
                await self._sync.batch_upsert(sync_batch)
                self._breaker.record_success()
                self._last_polled_at = datetime.now(UTC)
                self._last_skip_reason = None
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
                self._breaker.record_success()
                self._last_polled_at = datetime.now(UTC)
                self._last_skip_reason = None
                span.set_attribute("upsert", summary.upsert_count)
                span.set_attribute("delete", summary.delete_count)
                span.set_attribute("unchanged", summary.unchanged_count)
                return True, summary, changes
            except Exception as exc:
                self._breaker.record_failure()
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
                return False, summary, changes

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
                max_attempts=_max_attempts(self.config, self._default_max_attempts),
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
                max_attempts=_max_attempts(self.config, self._default_max_attempts),
                extra=_enqueue_extra(),
            )

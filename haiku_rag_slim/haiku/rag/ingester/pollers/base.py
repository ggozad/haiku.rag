import asyncio
import logging
from datetime import UTC, datetime

import logfire

from haiku.rag.config import SourceConfig
from haiku.rag.ingester.pollers.circuit_breaker import CircuitBreaker
from haiku.rag.ingester.queue.models import JobOp
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.sources.base import (
    Source,
    SourceEvent,
    SourceEventKind,
)

logger = logging.getLogger(__name__)


def _enqueue_extra(cfg: SourceConfig) -> dict | None:
    """Per-source state worth carrying into the job (so the worker can rebuild
    the same fetch context when it processes), plus the current logfire trace
    context so the worker's `ingester.job` span nests under the
    `ingester.poller.sweep` that enqueued it."""
    extra: dict = {}
    storage_options = getattr(cfg, "storage_options", None)
    if storage_options:
        extra["storage_options"] = dict(storage_options)
    headers = getattr(cfg, "headers", None)
    if headers:
        extra["headers"] = dict(headers)
    carrier = logfire.get_context()
    if carrier:
        extra["_otel"] = dict(carrier)
    return extra or None


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
        self._default_max_attempts = default_max_attempts

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def last_polled_at(self) -> datetime | None:
        return self._last_polled_at

    async def run(self) -> None:  # pragma: no cover - subclasses override
        raise NotImplementedError

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _sweep_once(self) -> bool:
        """One discover() sweep. Returns True on success, False if the
        breaker is open, the source has pending work already queued, or the
        sweep failed (and was recorded)."""
        if self._breaker.is_open:
            logger.debug(
                "Skipping discover() — circuit breaker open for %s", self.source_id
            )
            return False
        with logfire.span("ingester.poller.sweep", source_id=self.source_id) as span:
            if await self._jobs.has_pending(self.source_id):
                # The unique index would dedupe a re-sweep into a saturated
                # queue anyway; skipping saves the listing round-trip
                # (PROPFIND / S3 LIST / FS walk) and keeps Logfire readable.
                span.set_attribute("skipped", True)
                span.set_attribute("skip_reason", "pending_work")
                logger.debug(
                    "Skipping discover() — %s has pending work in the queue",
                    self.source_id,
                )
                return False
            try:
                snapshot = await self._sync.get_snapshot(self.source_id)
                counts = {
                    SourceEventKind.UPSERT: 0,
                    SourceEventKind.DELETE: 0,
                    SourceEventKind.UNCHANGED: 0,
                }
                async for event in self.source.discover(since=snapshot):
                    counts[event.kind] += 1
                    await self._handle_event(event)
                self._breaker.record_success()
                self._last_polled_at = datetime.now(UTC)
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

    async def _handle_event(self, event: SourceEvent) -> None:
        if event.kind is SourceEventKind.UPSERT:
            await self._jobs.enqueue(
                event.source_id,
                event.uri,
                op=JobOp.UPSERT,
                revision=event.revision,
                max_attempts=_max_attempts(self.config, self._default_max_attempts),
                extra=_enqueue_extra(self.config),
            )
            # Don't write revision to sync_state here — the worker writes it
            # after a successful ingestion. last_seen_at gets bumped to keep
            # orphan detection accurate.
            await self._sync.upsert(
                event.source_id,
                event.uri,
                revision=None,
                content_hash=None,
            )
        elif event.kind is SourceEventKind.UNCHANGED:
            # Touch last_seen_at without changing the stored revision.
            await self._sync.upsert(
                event.source_id,
                event.uri,
                revision=event.revision,
                content_hash=None,
            )
        elif event.kind is SourceEventKind.DELETE:
            if not self.config.delete_orphans:
                return
            await self._jobs.enqueue(
                event.source_id,
                event.uri,
                op=JobOp.DELETE,
                max_attempts=_max_attempts(self.config, self._default_max_attempts),
                extra=_enqueue_extra(self.config),
            )

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from haiku.rag.config import CircuitBreakerConfig
from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.pollers.circuit_breaker import CircuitBreaker
from haiku.rag.ingester.queue.models import Job, JobOp
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.workers.pipeline import run_job
from haiku.rag.ingester.workers.retry import RetryPolicy, compute_backoff

if TYPE_CHECKING:
    from collections.abc import Mapping

    from haiku.rag.client import HaikuRAG
    from haiku.rag.ingester.metadata import MetadataProvider
    from haiku.rag.ingester.sources.base import Source

logger = logging.getLogger(__name__)

_WORKER_BREAKER_THRESHOLD = 5
_WORKER_BREAKER_COOLDOWN_S = 60.0


class WorkerPool:
    """`worker_count` async tasks each pull jobs from the queue and run them
    through `run_job`. A reaper task resets claims older than
    `claim_timeout_s` so a crashed worker doesn't strand its job. Lifecycle:
    build it, await start(), let it run, await stop().
    """

    def __init__(
        self,
        *,
        client: "HaikuRAG",
        job_repo: JobRepo,
        sync_repo: SyncStateRepo,
        worker_count: int = 4,
        retry_policy: RetryPolicy | None = None,
        poll_idle_interval_s: float = 1.0,
        claim_timeout_s: int = 1800,
        reaper_interval_s: int = 60,
        retention_s: int | None = None,
        sources: "list[Source] | None" = None,
        metadata_providers: "Mapping[str, MetadataProvider] | None" = None,
    ):
        self._client = client
        self._jobs = job_repo
        self._sync = sync_repo
        self._worker_count = worker_count
        self._retry = retry_policy or RetryPolicy()
        self._poll_idle_s = poll_idle_interval_s
        self._claim_timeout_s = claim_timeout_s
        self._reaper_interval_s = reaper_interval_s
        self._retention_s = retention_s
        self._sources: list[Source] = list(sources) if sources else []
        self._metadata_providers: dict[str, MetadataProvider] = (
            dict(metadata_providers) if metadata_providers else {}
        )
        self._stop = asyncio.Event()
        self._workers: list[asyncio.Task] = []
        self._reaper: asyncio.Task | None = None
        self._pending_releases: set[asyncio.Task] = set()
        self._breakers: dict[str, CircuitBreaker] = {}

    @property
    def live_workers(self) -> int:
        """Worker tasks that are still running. Equal to worker_count under
        normal operation; less when a worker has crashed."""
        return sum(1 for t in self._workers if not t.done())

    @property
    def breaker_open(self) -> bool:
        return any(b.is_open for b in self._breakers.values())

    @property
    def breaker_consecutive_failures(self) -> int:
        return max((b.consecutive_failures for b in self._breakers.values()), default=0)

    def _breaker_for(self, source_id: str) -> CircuitBreaker:
        breaker = self._breakers.get(source_id)
        if breaker is None:
            breaker = CircuitBreaker(
                CircuitBreakerConfig(
                    failure_threshold=_WORKER_BREAKER_THRESHOLD,
                    cooldown_s=_WORKER_BREAKER_COOLDOWN_S,
                )
            )
            self._breakers[source_id] = breaker
        return breaker

    def _paused_source_ids(self) -> set[str]:
        return {sid for sid, b in self._breakers.items() if b.is_open}

    async def start(self) -> None:
        if self._workers:
            raise RuntimeError("WorkerPool already started")
        self._stop.clear()
        # Any rows in `claimed` at start time are owned by workers from a
        # previous process that didn't get to release them (SIGKILL, OOM,
        # host reboot). Reset them so fresh workers can claim immediately
        # instead of waiting on the reaper's claim_timeout_s.
        reset = await self._jobs.reap_stale(claim_timeout_seconds=0)
        if reset:
            logger.info("Boot-reaped %d stale claim(s) from previous process", reset)
        for i in range(self._worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(f"worker-{i}")))
        self._reaper = asyncio.create_task(self._reaper_loop())

    async def stop(self) -> None:
        self._stop.set()
        # Wake workers parked on job_available.wait() so they notice _stop
        # immediately instead of sleeping out the full poll_idle interval.
        async with self._jobs.job_available:
            self._jobs.job_available.notify_all()
        tasks = list(self._workers)
        if self._reaper is not None:
            tasks.append(self._reaper)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._reaper = None

    async def drain_pending_releases(self, timeout: float = 2.0) -> int:
        """Wait for any in-flight cancel-cleanup release Tasks to finish.
        Returns how many completed. Called by the lifecycle owner after
        stop() (success or timeout) so orphans land their SQL update before
        the queue connection closes. Tasks left running after `timeout`
        will be reclaimed by the reaper on the next start instead."""
        pending = list(self._pending_releases)
        if not pending:
            return 0
        done, _ = await asyncio.wait(pending, timeout=timeout)
        return len(done)

    async def drain_once(self, worker_id: str = "drain") -> int:
        """Drain every currently-claimable job to completion on the calling
        coroutine. Used by tests; not by `start()`."""
        processed = 0
        while True:
            job = await self._jobs.claim_next(worker_id)
            if job is None:
                return processed
            await self._process(job)
            processed += 1

    async def _worker_loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            job = await self._jobs.claim_next(
                worker_id, exclude_source_ids=self._paused_source_ids()
            )
            if job is None:
                try:
                    async with self._jobs.job_available:
                        await asyncio.wait_for(
                            self._jobs.job_available.wait(),
                            timeout=self._poll_idle_s,
                        )
                except TimeoutError:
                    pass
                continue
            await self._process(job)

    async def _reaper_loop(self) -> None:
        while not self._stop.is_set():
            await self._sleep_or_stop(self._reaper_interval_s)
            if self._stop.is_set():
                return
            reset = await self._jobs.reap_stale(self._claim_timeout_s)
            if reset:
                logger.info("Reaper reset %d stale claim(s)", reset)
            if self._retention_s is not None:
                pruned = await self._jobs.prune_terminal(self._retention_s)
                if pruned:
                    logger.info("Reaper pruned %d terminal job(s)", pruned)

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _process(self, job: Job) -> None:
        assert job.claimed_by is not None, "_process only runs on claimed jobs"
        worker_id = job.claimed_by
        started = time.monotonic()
        logger.info("Processing %s %s (job %s)", job.op.value, job.uri, job.id)
        try:
            result = await run_job(
                self._client,
                job,
                sources=self._sources,
                metadata_providers=self._metadata_providers,
            )
        except asyncio.CancelledError:
            # Graceful shutdown cancelled us mid-flight. Spawn the release
            # as an independent Task tracked in _pending_releases — that
            # way a second cancel (e.g. shutdown_grace_s elapses and
            # wait_for cancels stop() again) can interrupt our await
            # without interrupting the SQL update, and the lifecycle owner
            # can drain the orphans before closing the queue connection.
            release_task = asyncio.create_task(
                self._jobs.release_if_claimed(job.id, worker_id)
            )
            self._pending_releases.add(release_task)
            release_task.add_done_callback(self._pending_releases.discard)
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                logger.info(
                    "Job %s cancel-cleanup interrupted; orphan release Task "
                    "will be drained by the lifecycle owner",
                    job.id,
                )
            else:
                logger.info("Job %s released back to queue on cancel", job.id)
            raise
        except PermanentError as e:
            if not await self._jobs.mark_dead(job.id, str(e), worker_id):
                logger.warning(
                    "Job %s lost claim before mark_dead (likely reaper race); "
                    "letting the re-claiming worker drive",
                    job.id,
                )
                return
            logger.info("Job %s dead (permanent): %s", job.id, e)
            # Record the failed revision so discovery treats the unchanged file as
            # accounted-for and stops re-enqueuing it every sweep. sync_state.revision
            # means "last accounted-for revision" — ingested OR permanently failed.
            # Revision-less sources (no ETag) can't be suppressed this way.
            if job.revision is not None:
                try:
                    await self._sync.upsert(
                        job.source_id,
                        job.uri,
                        revision=job.revision,
                        content_hash=job.content_hash,
                        ingested=False,
                    )
                except Exception:
                    # The job is already dead. A failed marker write only means
                    # the next sweep may re-enqueue this URI — not worth crashing
                    # the worker and shrinking the pool over.
                    logger.exception(
                        "Job %s dead but failure marker write failed for %s; "
                        "next sweep may re-enqueue",
                        job.id,
                        job.uri,
                    )
            return
        except TransientError as e:
            breaker = self._breaker_for(job.source_id)
            was_closed = not breaker.is_open
            breaker.record_failure()
            if was_closed and breaker.is_open:
                logger.warning(
                    "Worker breaker opened for source %s after %d consecutive "
                    "transient failures; pausing its claims for %.0fs",
                    job.source_id,
                    _WORKER_BREAKER_THRESHOLD,
                    _WORKER_BREAKER_COOLDOWN_S,
                )
            if job.attempts >= job.max_attempts:
                await self._jobs.mark_dead(job.id, str(e), worker_id)
                logger.info(
                    "Job %s dead (max attempts %d): %s", job.id, job.max_attempts, e
                )
                return
            delay = compute_backoff(job.attempts, self._retry)
            if not await self._jobs.reschedule(  # pragma: no cover - reaper race
                job.id, delay, str(e), worker_id
            ):
                logger.warning(
                    "Job %s lost claim before reschedule (likely reaper race); "
                    "letting the re-claiming worker drive retry instead",
                    job.id,
                )
                return
            logger.info(
                "Job %s rescheduled in %.1fs (attempt %d/%d): %s",
                job.id,
                delay,
                job.attempts,
                job.max_attempts,
                e,
            )
            return
        # Guard against the reaper race: if our claim was reset and another
        # worker re-claimed the job, mark_succeeded is a no-op. Don't write
        # sync_state in that case — the new worker will write it when it
        # finishes.
        if not await self._jobs.mark_succeeded(job.id, worker_id):
            logger.warning(
                "Job %s lost claim before mark_succeeded (likely reaper race); "
                "skipping sync_state write",
                job.id,
            )
            return
        breaker = self._breaker_for(job.source_id)
        was_open = breaker.is_open
        breaker.record_success()
        if was_open:
            logger.info(
                "Worker breaker closed for source %s after successful probe",
                job.source_id,
            )
        try:
            if job.op is JobOp.DELETE:
                await self._sync.delete(job.source_id, job.uri)
                # A successful DELETE resolves any earlier UPSERT failures for
                # the same (source_id, uri): the document is gone, the original
                # error is no longer actionable, the DLQ entry is visual noise.
                pruned = await self._jobs.prune_dead(job.source_id, job.uri)
                if pruned:
                    logger.info(
                        "Pruned %d dead job(s) for %s after successful DELETE",
                        pruned,
                        job.uri,
                    )
            else:
                await self._sync.upsert(
                    job.source_id,
                    job.uri,
                    revision=result.revision,
                    content_hash=result.content_hash,
                    ingested=True,
                )
        except Exception:
            # The job is already marked succeeded — the document was ingested
            # correctly. A sync_state write failure means the next sweep may
            # redundantly re-ingest this URI, but that's better than crashing
            # the worker and blocking the rest of the queue.
            logger.exception(
                "Job %s succeeded but sync_state write failed for %s; "
                "next sweep may re-ingest",
                job.id,
                job.uri,
            )
        logger.info(
            "Job %s succeeded in %.2fs: %s", job.id, time.monotonic() - started, job.uri
        )

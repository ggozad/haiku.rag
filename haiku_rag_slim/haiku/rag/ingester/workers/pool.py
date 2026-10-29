import asyncio
import itertools
import logging
import os
import time
from collections import Counter
from typing import TYPE_CHECKING
from uuid import uuid4

from haiku.rag.circuit_breaker import CircuitBreaker
from haiku.rag.config import CircuitBreakerConfig
from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.models import Job, JobOp
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.workers.pipeline import run_job
from haiku.rag.ingester.workers.retry import RetryPolicy, compute_backoff
from haiku.rag.telemetry import logfire

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from haiku.rag.client import HaikuRAG
    from haiku.rag.ingester.metadata import MetadataProvider
    from haiku.rag.sources.base import Source

logger = logging.getLogger(__name__)


def _terminate_wedged_process() -> None:  # pragma: no cover - ends the process
    """Leave the process for a supervisor to replace, now, without unwinding.

    The caller has already committed the job's terminal state, and the process
    cannot convert again, so every further claim is a job it will fail.
    """
    try:
        logging.shutdown()
    finally:
        os._exit(1)


_WORKER_BREAKER_THRESHOLD = 5
_WORKER_BREAKER_COOLDOWN_S = 60.0


class WorkerPool:
    """`worker_count` async tasks each pull jobs from the queue and run them
    through `run_job`. `clients` maps a source id to the client that writes
    its jobs, and only served sources' jobs are claimed. `add_sources` and
    `unserve` change what is served while the pool runs. While a worker
    processes a job it renews the job's lease on a `heartbeat_interval_s`
    cadence; a reaper task resets claims whose lease has gone stale
    (`lease_ttl_s`) so a crashed worker doesn't strand its job, without
    reaping jobs that are merely slow. Lifecycle: build it, await
    start(), let it run, await stop().
    """

    def __init__(
        self,
        *,
        clients: "Mapping[str, HaikuRAG]",
        job_repo: JobRepo,
        sync_repo: SyncStateRepo,
        worker_count: int = 4,
        retry_policy: RetryPolicy | None = None,
        source_retry_policies: "Mapping[str, RetryPolicy] | None" = None,
        poll_idle_interval_s: float = 1.0,
        lease_ttl_s: int = 120,
        heartbeat_interval_s: int = 30,
        reaper_interval_s: int = 60,
        retention_s: int | None = None,
        sources: "list[Source] | None" = None,
        metadata_providers: "Mapping[str, MetadataProvider] | None" = None,
    ):
        self._clients: dict[str, HaikuRAG] = dict(clients)
        self._jobs = job_repo
        self._sync = sync_repo
        self._worker_count = worker_count
        self._retry = retry_policy or RetryPolicy()
        self._source_retry: dict[str, RetryPolicy] = dict(source_retry_policies or {})
        self._poll_idle_s = poll_idle_interval_s
        self._lease_ttl_s = lease_ttl_s
        self._heartbeat_interval_s = heartbeat_interval_s
        self._reaper_interval_s = reaper_interval_s
        self._retention_s = retention_s
        self._sources: dict[str, Source] = {
            source.source_id: source for source in sources or ()
        }
        self._served: set[str] = set(self._clients)
        # Claims and `unserve` take it, so once `unserve` returns no claim
        # names its sources, and every claim that did is in `_executions`.
        self.claim_lock = asyncio.Lock()
        self._metadata_providers: dict[str, MetadataProvider] = (
            dict(metadata_providers) if metadata_providers else {}
        )
        # Globally-unique so claimed_by distinguishes this pool's workers from
        # those of any other process sharing the queue; the claimed_by guards on
        # mark_succeeded/reschedule/release rely on it. The uuid guarantees
        # uniqueness; the pid just makes claimed_by readable in logs/dashboard.
        self._instance = f"{os.getpid()}-{uuid4().hex[:8]}"
        self._stop = asyncio.Event()
        self._workers: list[asyncio.Task] = []
        self._reaper: asyncio.Task | None = None
        self._heartbeat: asyncio.Task | None = None
        # job_id -> claimed_by for jobs currently being processed by this pool;
        # the heartbeat renews exactly these leases.
        self._inflight: dict[str, str] = {}
        # worker_id -> source_id of every job this pool is executing. A
        # reclaimed job runs twice, and `_inflight` names only its last claim.
        self._executions: dict[str, str] = {}
        # Set whenever an execution ends, for `wait_idle`.
        self._finished = asyncio.Event()
        # Set whenever _inflight is empty so the heartbeat can leave promptly
        # once stopping instead of sleeping out a full interval.
        self._idle = asyncio.Event()
        self._idle.set()
        self._pending_releases: set[asyncio.Task] = set()
        self._breakers: dict[str, CircuitBreaker] = {}
        # source_id -> sequence number of its last claim, for the tie-break.
        self._last_claim: dict[str, int] = {}
        self._claims = itertools.count()

    def _worker_id(self, i: int) -> str:
        return f"{self._instance}-{i}"

    def _track_inflight(self, job_id: str, worker_id: str) -> None:
        self._inflight[job_id] = worker_id
        self._idle.clear()

    def _untrack_inflight(self, job_id: str, worker_id: str) -> None:
        # Only drop our own entry: if the reaper reset this claim and a sibling
        # worker re-claimed it, the entry now belongs to that worker and must
        # keep being renewed — our exit must not evict it.
        if self._inflight.get(job_id) == worker_id:
            del self._inflight[job_id]
        if self._executions.pop(worker_id, None) is not None:
            self._finished.set()
        if not self._inflight:
            self._idle.set()

    @property
    def live_workers(self) -> int:
        """Worker tasks that are still running. Equal to worker_count under
        normal operation; less when a worker has crashed."""
        return sum(1 for t in self._workers if not t.done())

    @property
    def heartbeat_alive(self) -> bool:
        """Whether the lease-renewal task is running. Once the pool is started
        this stays True until stop(); a False here while workers are live means
        in-flight leases are no longer being renewed and may be reaped."""
        return self._heartbeat is not None and not self._heartbeat.done()

    @property
    def breaker_open(self) -> bool:
        return any(b.is_open for b in self._breakers.values())

    @property
    def breaker_consecutive_failures(self) -> int:
        return max((b.consecutive_failures for b in self._breakers.values()), default=0)

    def breaker_open_for(self, source_id: str) -> bool:
        """Whether `source_id`'s breaker holds back claims of its jobs."""
        breaker = self._breakers.get(source_id)
        return breaker is not None and breaker.is_open

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

    def _claimable_source_ids(self) -> set[str]:
        """Served sources whose breaker is closed."""
        return {
            source_id
            for source_id in self._served
            if not self.breaker_open_for(source_id)
        }

    def add_sources(
        self,
        *,
        clients: "Mapping[str, HaikuRAG]",
        sources: "list[Source]",
        source_retry_policies: "Mapping[str, RetryPolicy] | None" = None,
        metadata_providers: "Mapping[str, MetadataProvider] | None" = None,
    ) -> None:
        """Serve every source in `clients`, whether the pool runs or not."""
        self._clients.update(clients)
        self._sources.update({source.source_id: source for source in sources})
        self._source_retry.update(source_retry_policies or {})
        self._metadata_providers.update(metadata_providers or {})
        self._served.update(clients)

    async def unserve(self, source_ids: "Collection[str]") -> None:
        """Stop claiming jobs of `source_ids`. Their jobs in flight run on:
        `wait_idle` waits for them."""
        async with self.claim_lock:
            self._served.difference_update(source_ids)

    async def wait_idle(self, source_ids: "Collection[str]") -> None:
        """Return once no job of `source_ids` is in flight."""
        while True:
            self._finished.clear()
            if not any(s in source_ids for s in self._executions.values()):
                return
            await self._finished.wait()

    def remove_sources(self, source_ids: "Collection[str]") -> None:
        """Forget `source_ids`, which are unserved and idle."""
        for source_id in source_ids:
            self._clients.pop(source_id, None)
            self._sources.pop(source_id, None)
            self._source_retry.pop(source_id, None)
            self._metadata_providers.pop(source_id, None)
            self._breakers.pop(source_id, None)
            self._last_claim.pop(source_id, None)

    async def _claim(
        self, worker_id: str, *, open_breakers: bool = False
    ) -> Job | None:
        """Claim the oldest job of the ready source with the fewest jobs in
        flight, then the one claimed least recently. The sources are read under
        the lock, so a claim queued behind `unserve` sees its result."""
        async with self.claim_lock:
            source_ids = (
                set(self._served) if open_breakers else self._claimable_source_ids()
            )
            if len(source_ids) > 1:
                source_ids = await self._jobs.ready_sources(source_ids)
            in_flight = Counter(self._executions.values())
            for source_id in sorted(
                source_ids,
                key=lambda s: (in_flight[s], self._last_claim.get(s, -1), s),
            ):
                job = await self._jobs.claim_next(worker_id, source_ids={source_id})
                if job is not None:
                    self._last_claim[source_id] = next(self._claims)
                    self._track(job)
                    return job
            return None

    def _track(self, job: Job) -> None:
        assert job.claimed_by is not None, "only claimed jobs are in flight"
        self._track_inflight(job.id, job.claimed_by)
        self._executions[job.claimed_by] = job.source_id

    async def start(self) -> None:
        if self._workers:
            raise RuntimeError("WorkerPool already started")
        self._stop.clear()
        # Sweep claims left behind by a previous process (SIGKILL, OOM, host
        # reboot) so fresh workers can take them over. Scoped by the lease TTL,
        # not 0: a peer process sharing the queue may hold live claims, and
        # those must not be wiped out from under it on our startup.
        reset = await self._jobs.reap_stale(lease_ttl_seconds=self._lease_ttl_s)
        if reset:
            logger.info("Boot-reaped %d stale claim(s) from previous process", reset)
        for i in range(self._worker_count):
            self._workers.append(
                asyncio.create_task(self._worker_loop(self._worker_id(i)))
            )
        self._reaper = asyncio.create_task(self._reaper_loop())
        self._heartbeat = asyncio.create_task(self._heartbeat_loop())
        self._heartbeat.add_done_callback(self._on_heartbeat_done)

    def _on_heartbeat_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Heartbeat task died: %r", exc)
        elif not self._stop.is_set():
            logger.error("Heartbeat task exited while the pool was still running")

    async def stop(self) -> None:
        self._stop.set()
        # Wake workers parked on job_available.wait() so they notice _stop
        # immediately instead of sleeping out the full poll_idle interval.
        # _stop also wakes the heartbeat's interval sleep.
        async with self._jobs.job_available:
            self._jobs.job_available.notify_all()
        tasks = list(self._workers)
        if self._reaper is not None:
            tasks.append(self._reaper)
        if self._heartbeat is not None:
            tasks.append(self._heartbeat)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._reaper = None
        self._heartbeat = None

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
            job = await self._claim(worker_id, open_breakers=True)
            if job is None:
                return processed
            await self._process(job)
            processed += 1

    async def _worker_loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            job = await self._claim(worker_id)
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
            try:
                await self._process(job)
            finally:
                # The claim tracked the job. `_process` untracks it as well,
                # unless it fails before its own cleanup.
                self._untrack_inflight(job.id, worker_id)

    async def _reaper_loop(self) -> None:
        while not self._stop.is_set():
            await self._sleep_or_stop(self._reaper_interval_s)
            if self._stop.is_set():
                return
            reset = await self._jobs.reap_stale(self._lease_ttl_s)
            if reset:
                logger.info("Reaper reset %d stale claim(s)", reset)
            if self._retention_s is not None:
                pruned = await self._jobs.prune_terminal(self._retention_s)
                if pruned:
                    logger.info("Reaper pruned %d terminal job(s)", pruned)

    async def _heartbeat_loop(self) -> None:
        """Renew the lease on this pool's in-flight jobs so the reaper leaves
        them alone while they are still being processed. Continues renewing
        through graceful shutdown until the last job drains, so a peer process
        doesn't reap a job we're still finishing. A forced shutdown cancels
        this task along with the workers, which is correct — they are no longer
        draining gracefully."""
        while True:
            if self._inflight:
                try:
                    await self._jobs.renew_claims(dict(self._inflight))
                except Exception:
                    logger.exception(
                        "Lease renewal failed; retrying on the next heartbeat"
                    )
            if self._stop.is_set() and not self._inflight:
                return
            if self._stop.is_set():
                # Draining: keep the renewal cadence, but leave as soon as the
                # last in-flight job finishes instead of sleeping it out.
                try:
                    await asyncio.wait_for(
                        self._idle.wait(), timeout=self._heartbeat_interval_s
                    )
                except TimeoutError:
                    pass
            else:
                await self._sleep_or_stop(self._heartbeat_interval_s)

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _process(self, job: Job) -> None:
        assert job.claimed_by is not None, "_process only runs on claimed jobs"
        worker_id = job.claimed_by
        # Track before any await so the heartbeat renews this job's lease for
        # its whole lifetime; untrack on every exit (success, error, cancel).
        self._track(job)
        try:
            await self._run_job_lifecycle(job, worker_id)
        finally:
            self._untrack_inflight(job.id, worker_id)

    async def _finish_permanent(
        self, job: Job, error: PermanentError, worker_id: str
    ) -> None:
        """Record a permanent failure, dead or rejected, then the suppression
        marker."""
        if error.rejected:
            recorded = await self._jobs.mark_rejected(job.id, str(error), worker_id)
        else:
            recorded = await self._jobs.mark_dead(
                job.id,
                str(error),
                worker_id,
                conversion_stalled=error.conversion_stalled,
            )
        if not recorded:
            logger.warning(
                "Job %s lost claim before it was recorded (likely reaper race); "
                "letting the re-claiming worker drive",
                job.id,
            )
            return
        if error.rejected:
            logger.info("Job %s rejected: %s", job.id, error)
        else:
            logger.info("Job %s dead (permanent): %s", job.id, error)
        # Record the failed revision so discovery treats the unchanged file as
        # accounted-for and stops re-enqueuing it every sweep. sync_state.revision
        # means "last accounted-for revision": ingested, permanently failed or
        # rejected. Revision-less sources (no ETag) can't be suppressed this way.
        if job.revision is None:
            return
        try:
            await self._sync.upsert(
                job.source_id,
                job.uri,
                revision=job.revision,
                content_hash=job.content_hash,
                ingested=False,
            )
        except Exception:
            # The job is already recorded. A failed marker write only means the
            # next sweep may re-enqueue this URI — not worth crashing the
            # worker and shrinking the pool over.
            logger.exception(
                "Job %s recorded but failure marker write failed for %s; "
                "next sweep may re-enqueue",
                job.id,
                job.uri,
            )

    async def _run_job_lifecycle(self, job: Job, worker_id: str) -> None:
        started = time.monotonic()
        logger.info("Processing %s %s (job %s)", job.op.value, job.uri, job.id)
        try:
            result = await run_job(
                self._clients[job.source_id],
                job,
                sources=list(self._sources.values()),
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
            # A process that cannot convert has to go however this branch
            # ends: `mark_dead` losing the claim or raising must not leave it
            # claiming jobs it will fail. Persistence still runs first, since
            # `reap_stale` refunds the attempt of a vanished owner.
            try:
                await self._finish_permanent(job, e, worker_id)
            finally:
                if e.fatal_to_process:
                    logger.error(
                        "Job %s left this process unable to convert; exiting "
                        "for a supervisor to replace it: %s",
                        job.id,
                        e,
                    )
                    _terminate_wedged_process()
            return
        except TransientError as e:
            breaker = self._breaker_for(job.source_id)
            was_closed = not breaker.is_open
            breaker.record_failure()
            if was_closed and breaker.is_open:
                logfire.warn(
                    "ingester.worker breaker opened",
                    source_id=job.source_id,
                    threshold=_WORKER_BREAKER_THRESHOLD,
                    cooldown_s=_WORKER_BREAKER_COOLDOWN_S,
                )
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
            delay = compute_backoff(
                job.attempts, self._source_retry.get(job.source_id, self._retry)
            )
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
            if job.op is JobOp.DELETE and not result.resource_restored:
                await self._sync.delete(job.source_id, job.uri)
                # Retiring a source's URI resolves its earlier ingestion failures.
                pruned = await self._jobs.prune_unsuccessful(job.source_id, job.uri)
                if pruned:
                    logger.info(
                        "Pruned %d dead or rejected job(s) for %s after successful DELETE",
                        pruned,
                        job.uri,
                    )
            elif job.op is JobOp.UPSERT:
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

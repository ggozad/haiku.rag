import asyncio
import logging
import time
from typing import TYPE_CHECKING

from haiku.rag.ingester.exceptions import PermanentError, TransientError
from haiku.rag.ingester.queue.models import Job, JobOp
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.workers.pipeline import run_job
from haiku.rag.ingester.workers.retry import RetryPolicy, compute_backoff

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG

logger = logging.getLogger(__name__)


class WorkerPool:
    """Asyncio-based pool. N worker tasks share a bounded Semaphore, each
    pulling jobs from the queue and running them through `run_job`. Reaper
    task resets claims older than `claim_timeout_s` so a crashed worker
    doesn't strand its job.

    Lifecycle: build it, await start(), let it run, await stop().
    """

    def __init__(
        self,
        *,
        client: "HaikuRAG",
        job_repo: JobRepo,
        sync_repo: SyncStateRepo,
        worker_count: int = 4,
        max_concurrent: int = 4,
        retry_policy: RetryPolicy | None = None,
        poll_idle_interval_s: float = 1.0,
        claim_timeout_s: int = 1800,
        reaper_interval_s: int = 60,
    ):
        self._client = client
        self._jobs = job_repo
        self._sync = sync_repo
        self._worker_count = worker_count
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._retry = retry_policy or RetryPolicy()
        self._poll_idle_s = poll_idle_interval_s
        self._claim_timeout_s = claim_timeout_s
        self._reaper_interval_s = reaper_interval_s
        self._stop = asyncio.Event()
        self._workers: list[asyncio.Task] = []
        self._reaper: asyncio.Task | None = None

    async def start(self) -> None:
        if self._workers:
            raise RuntimeError("WorkerPool already started")
        self._stop.clear()
        for i in range(self._worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(f"worker-{i}")))
        self._reaper = asyncio.create_task(self._reaper_loop())

    async def stop(self) -> None:
        self._stop.set()
        tasks = list(self._workers)
        if self._reaper is not None:
            tasks.append(self._reaper)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._workers.clear()
        self._reaper = None

    async def drain_once(self, worker_id: str = "drain") -> int:
        """Drain every currently-claimable job to completion on the calling
        coroutine. Used by run-once and by tests; not by `start()`."""
        processed = 0
        while True:
            job = await self._jobs.claim_next(worker_id)
            if job is None:
                return processed
            await self._process(job)
            processed += 1

    async def _worker_loop(self, worker_id: str) -> None:
        while not self._stop.is_set():
            try:
                job = await self._jobs.claim_next(worker_id)
            except Exception:  # pragma: no cover - defensive against DB hiccups
                logger.exception("claim_next failed in %s", worker_id)
                await self._sleep_or_stop(self._poll_idle_s)
                continue

            if job is None:
                await self._sleep_or_stop(self._poll_idle_s)
                continue

            async with self._semaphore:
                await self._process(job)

    async def _reaper_loop(self) -> None:
        while not self._stop.is_set():
            await self._sleep_or_stop(self._reaper_interval_s)
            if self._stop.is_set():
                return
            try:
                reset = await self._jobs.reap_stale(self._claim_timeout_s)
                if reset:
                    logger.info("Reaper reset %d stale claim(s)", reset)
            except Exception:  # pragma: no cover - defensive against DB hiccups
                logger.exception("reaper failed")

    async def _sleep_or_stop(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _process(self, job: Job) -> None:
        started = time.monotonic()
        logger.info("Processing %s %s (job %s)", job.op.value, job.uri, job.id)
        try:
            result = await run_job(self._client, job)
        except PermanentError as e:
            await self._jobs.mark_dead(job.id, str(e))
            logger.info("Job %s dead (permanent): %s", job.id, e)
            return
        except TransientError as e:
            if job.attempts >= job.max_attempts:
                await self._jobs.mark_dead(job.id, str(e))
                logger.info(
                    "Job %s dead (max attempts %d): %s", job.id, job.max_attempts, e
                )
                return
            delay = compute_backoff(job.attempts, self._retry)
            await self._jobs.reschedule(job.id, delay, str(e))
            logger.info(
                "Job %s rescheduled in %.1fs (attempt %d/%d): %s",
                job.id,
                delay,
                job.attempts,
                job.max_attempts,
                e,
            )
            return
        except Exception as e:  # pragma: no cover - pipeline classifier net
            # Defensive: pipeline classifier should have caught everything.
            await self._jobs.mark_dead(job.id, f"unclassified: {e!r}")
            logger.exception("Unclassified error in job %s", job.id)
            return

        await self._jobs.mark_succeeded(job.id)
        if job.op is JobOp.DELETE:
            await self._sync.delete(job.source_id, job.uri)
        else:
            await self._sync.upsert(
                job.source_id,
                job.uri,
                revision=result.revision,
                content_hash=result.content_hash,
                ingested=True,
            )
        logger.info(
            "Job %s succeeded in %.2fs: %s", job.id, time.monotonic() - started, job.uri
        )

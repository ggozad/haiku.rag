import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from watchfiles import Change, awatch

from haiku.rag.ingester.pollers.base import BasePoller, _enqueue_extra, max_attempts
from haiku.rag.ingester.queue.models import JobOp
from haiku.rag.sources.filter import FileFilter
from haiku.rag.telemetry import logfire

if TYPE_CHECKING:
    from haiku.rag.circuit_breaker import CircuitBreaker
    from haiku.rag.config import FSSourceConfig
    from haiku.rag.sources.fs import FSSource

logger = logging.getLogger(__name__)

_WATCH_RETRY_INITIAL_S = 1.0
_WATCH_RETRY_MAX_S = 300.0


class FSPoller(BasePoller):
    """Filesystem poller: initial discover() sweep plus a watchfiles-driven
    push loop. Periodic sweeps still run so files modified while the watcher
    was offline get picked up too."""

    def __init__(
        self,
        *,
        source: "FSSource",
        config: "FSSourceConfig",
        job_repo,
        sync_repo,
        breaker: "CircuitBreaker | None" = None,
        default_max_attempts: int = 5,
    ):
        super().__init__(
            source=source,
            config=config,
            job_repo=job_repo,
            sync_repo=sync_repo,
            breaker=breaker,
            default_max_attempts=default_max_attempts,
        )
        self._fs_source: FSSource = source
        self._fs_config: FSSourceConfig = config
        self._filter = FileFilter(
            ignore_patterns=config.ignore_patterns or None,
            include_patterns=config.include_patterns or None,
            supported_extensions=source.supported_extensions,
        )

    async def run(self) -> None:
        await self._sweep_once()
        watch_task = asyncio.create_task(self._watch_loop())
        sweep_task = asyncio.create_task(self._sweep_loop())
        try:
            await self._stop.wait()
        finally:
            watch_task.cancel()
            sweep_task.cancel()
            await asyncio.gather(watch_task, sweep_task, return_exceptions=True)

    async def _sweep_loop(self) -> None:  # pragma: no cover - event-loop glue
        """Periodic full sweep. Catches files modified while the watcher
        wasn't running (gaps between starts, races, FS events the OS dropped).
        Sweep behaviour is unit-tested via `_sweep_once()` directly."""
        if await self._stagger_start():
            return
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.config.poll_interval_s
                )
                return
            except TimeoutError:
                pass
            await self._sweep_once()

    async def _watch_loop(self) -> None:
        """Push-event loop on top of watchfiles. Each change is translated
        into one queue job — no need to re-stat or re-snapshot.

        A failed watcher, such as one over a root that does not exist yet,
        restarts after a doubling delay until the poller stops. Its failures
        leave the breaker alone: the breaker gates the sweeps.
        """
        delay = _WATCH_RETRY_INITIAL_S
        while not self._stop.is_set():
            try:
                async for changes in awatch(
                    self._fs_source.root,
                    watch_filter=self._filter,
                    stop_event=self._stop,
                ):
                    delay = _WATCH_RETRY_INITIAL_S
                    for change, path in changes:
                        await self._handle_watch_change(change, Path(path))
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_error(exc)
                logger.exception(
                    "watchfiles loop failed for %s: %s", self.source_id, exc
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except TimeoutError:
                pass
            delay = min(delay * 2, _WATCH_RETRY_MAX_S)

    async def _handle_watch_change(self, change: Change, path: Path) -> None:
        uri = path.as_uri()
        # Wrap in a span so the worker's `ingester.job` (and everything it
        # nests) hangs off a watch-event parent. Without this the watchfiles
        # callback runs with no active context, the `_otel` carrier is empty,
        # and the job span surfaces at the trace root — disconnected from
        # the FS event that caused it.
        with logfire.span(
            "ingester.poller.watch_event",
            source_id=self.source_id,
            change=change.name,
            uri=uri,
        ):
            async with self._sweep_lock:
                await self._enqueue_change(change, path, uri)

    async def _enqueue_change(self, change: Change, path: Path, uri: str) -> None:
        if change is Change.deleted:
            if not self._fs_config.delete_orphans:
                return
            # `git checkout`, atomic-rename saves, and similar atomic
            # restores fire (deleted, added) back-to-back. By the time we
            # handle the delete, the file is already back. Enqueuing
            # DELETE here would block the follow-up Change.added's UPSERT
            # via the live-row unique index, then run and remove the
            # document — blackholing it until the next periodic sweep.
            if path.exists():
                return
            await self._jobs.enqueue(
                self.source_id,
                uri,
                op=JobOp.DELETE,
                max_attempts=max_attempts(self.config, self._default_max_attempts),
                extra=_enqueue_extra(),
            )
            return

        if change in (
            Change.added,
            Change.modified,
        ):  # pragma: no branch - deleted returns above
            try:
                revision = str(path.stat().st_mtime_ns)
            except FileNotFoundError:
                # File was deleted between the watchfiles event and our
                # stat() call.  Skip — the deletion event will handle it.
                return
            await self._jobs.enqueue(
                self.source_id,
                uri,
                op=JobOp.UPSERT,
                revision=revision,
                max_attempts=max_attempts(self.config, self._default_max_attempts),
                extra=_enqueue_extra(),
            )
            await self._sync.upsert(
                self.source_id, uri, revision=None, content_hash=None
            )

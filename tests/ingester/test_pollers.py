import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from haiku.rag.circuit_breaker import CircuitBreaker
from haiku.rag.config import (
    CircuitBreakerConfig,
    FSSourceConfig,
    HTTPSourceConfig,
    S3SourceConfig,
    WebDAVSourceConfig,
)
from haiku.rag.ingester.exceptions import EmptyListingError
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.pollers.periodic import PeriodicPoller
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from haiku.rag.sources.base import (
    FetchResult,
    SourceEvent,
    SourceEventKind,
)
from tests.conftest import capture_logs

_POLLER_LOGGER = logging.getLogger("haiku.rag.ingester.pollers.base")


class _StubSource:
    """Test double that yields a scripted sequence of events on each
    discover() call. `fetch` and `supports` aren't exercised by pollers."""

    def __init__(self, source_id: str, sweeps: list[list[SourceEvent]]):
        self.source_id = source_id
        self._sweeps = list(sweeps)
        self.discover_calls = 0
        self.fail_with: Exception | None = None

    def supports(self, uri: str) -> bool:  # pragma: no cover - unused here
        return True

    async def head(self, uri: str) -> str | None:  # pragma: no cover
        return None

    async def fetch(self, uri: str) -> FetchResult:  # pragma: no cover
        raise NotImplementedError

    async def discover(self, since=None, *, known_uris=None):
        self.discover_calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        events = self._sweeps.pop(0) if self._sweeps else []
        for event in events:
            yield event


def _event(
    uri: str,
    kind=SourceEventKind.UPSERT,
    revision: str | None = "v1",
    source_id: str = "src",
):
    return SourceEvent(
        source_id=source_id,
        uri=uri,
        kind=kind,
        revision=None if kind is SourceEventKind.DELETE else revision,
        discovered_at=datetime.now(UTC),
    )


class _FailsMidway(_StubSource):
    """Lists one file and one delete, then raises."""

    def __init__(self, source_id: str):
        super().__init__(source_id, [])

    async def discover(self, since=None, *, known_uris=None):
        yield _event("file:///kept.md", source_id=self.source_id)
        yield _event(
            "file:///gone.md", kind=SourceEventKind.DELETE, source_id=self.source_id
        )
        raise RuntimeError("listing broke")


@pytest.fixture
def fs_config(tmp_path):
    return FSSourceConfig(
        type="fs",
        id="src",
        root=tmp_path,
        delete_orphans=True,
        poll_interval_s=0.05,
    )


def _periodic(source, config, jobs, sync, **kwargs):
    return PeriodicPoller(
        source=source,
        config=config,
        job_repo=jobs,
        sync_repo=sync,
        **kwargs,
    )


# --- _stagger_start ---


async def test_stagger_start_sleeps_fraction_of_interval(
    jobs, sync, fs_config, monkeypatch
):
    """_stagger_start should sleep for a random fraction of poll_interval_s
    and return False (not stopped)."""
    monkeypatch.setattr("random.uniform", lambda a, b: b)  # max jitter
    source = _StubSource("src", [])
    poller = _periodic(source, fs_config, jobs, sync)
    # poll_interval_s=0.05, so max jitter = 0.05 * 0.25 = 0.0125s
    stopped = await poller._stagger_start()
    assert stopped is False


async def test_stagger_start_returns_true_when_stopped(
    jobs, sync, fs_config, monkeypatch
):
    """If _stop is set before the jitter elapses, _stagger_start returns True."""
    monkeypatch.setattr("random.uniform", lambda a, b: 10.0)  # long jitter
    source = _StubSource("src", [])
    poller = _periodic(source, fs_config, jobs, sync)
    poller._stop.set()
    stopped = await poller._stagger_start()
    assert stopped is True


# --- _sweep_once / event handling on the base class via PeriodicPoller ---


async def test_upsert_event_enqueues_job_and_touches_sync_state(fs_config, jobs, sync):
    source = _StubSource("src", [[_event("file:///a.md", revision="r1")]])
    poller = _periodic(source, fs_config, jobs, sync)
    ok = await poller._sweep_once()
    assert ok is True

    queued = await jobs.list_jobs(source_id="src")
    assert len(queued) == 1
    assert queued[0].op is JobOp.UPSERT
    assert queued[0].revision == "r1"

    # Pollers DO NOT write revision to sync_state — the worker does that
    # after a successful ingest. The URI shows up in list_known_uris from
    # the moment the poller emits the event, but the revision_snapshot
    # stays empty until ingestion completes.
    assert await sync.get_revision_snapshot("src") == {}
    assert await sync.list_known_uris("src") == {"file:///a.md"}


async def test_unchanged_event_touches_sync_state_no_job(fs_config, jobs, sync):
    source = _StubSource(
        "src", [[_event("file:///a.md", kind=SourceEventKind.UNCHANGED, revision="r1")]]
    )
    poller = _periodic(source, fs_config, jobs, sync)
    await poller._sweep_once()

    assert await jobs.list_jobs(source_id="src") == []
    assert await sync.get_revision_snapshot("src") == {"file:///a.md": "r1"}


async def test_delete_event_enqueues_delete_job(fs_config, jobs, sync):
    source = _StubSource(
        "src",
        [
            [
                _event("file:///kept.md", kind=SourceEventKind.UNCHANGED),
                _event("file:///gone.md", kind=SourceEventKind.DELETE),
            ]
        ],
    )
    poller = _periodic(source, fs_config, jobs, sync)
    await poller._sweep_once()

    queued = await jobs.list_jobs(source_id="src")
    assert len(queued) == 1
    assert queued[0].op is JobOp.DELETE


async def test_empty_listing_with_deletes_fails_the_sweep(fs_config, jobs, sync):
    await sync.upsert("src", "file:///gone.md", revision="r1", ingested=True)
    source = _StubSource(
        "src", [[_event("file:///gone.md", kind=SourceEventKind.DELETE)]]
    )
    poller = _periodic(source, fs_config, jobs, sync)

    with capture_logs(_POLLER_LOGGER, logging.ERROR) as records:
        assert await poller._sweep_once() is False

    assert await jobs.list_jobs(source_id="src") == []
    assert await sync.get_revision_snapshot("src") == {"file:///gone.md": "r1"}
    assert await sync.list_ingested_uris("src") == {"file:///gone.md"}
    assert poller._breaker.consecutive_failures == 1
    assert poller.last_polled_at is None
    assert any(
        isinstance(r.exc_info[1], EmptyListingError) for r in records if r.exc_info
    )


async def test_empty_listing_with_nothing_known_succeeds(fs_config, jobs, sync):
    poller = _periodic(_StubSource("src", [[]]), fs_config, jobs, sync)
    assert await poller._sweep_once() is True


async def test_empty_listing_succeeds_when_delete_orphans_false(fs_config, jobs, sync):
    config = fs_config.model_copy(update={"delete_orphans": False})
    source = _StubSource(
        "src", [[_event("file:///gone.md", kind=SourceEventKind.DELETE)]]
    )
    poller = _periodic(source, config, jobs, sync)

    assert await poller._sweep_once() is True
    assert await jobs.list_jobs(source_id="src") == []


async def test_deletes_before_a_failed_listing_are_not_enqueued(fs_config, jobs, sync):
    poller = _periodic(_FailsMidway("src"), fs_config, jobs, sync)

    with capture_logs(_POLLER_LOGGER, logging.ERROR):
        assert await poller._sweep_once() is False

    ops = [job.op for job in await jobs.list_jobs(source_id="src")]
    assert JobOp.DELETE not in ops


async def test_delete_event_skipped_when_delete_orphans_false(fs_config, jobs, sync):
    fs_config = fs_config.model_copy(update={"delete_orphans": False})
    source = _StubSource(
        "src", [[_event("file:///gone.md", kind=SourceEventKind.DELETE)]]
    )
    poller = _periodic(source, fs_config, jobs, sync)
    await poller._sweep_once()
    assert await jobs.list_jobs(source_id="src") == []


async def test_repeated_sweep_skipped_when_queue_has_pending(fs_config, jobs, sync):
    """Backpressure: once a job is queued/claimed, the next sweep skips
    discover() entirely instead of churning the listing operation."""
    event = _event("file:///a.md", revision="r1")
    source = _StubSource("src", [[event], [event]])
    poller = _periodic(source, fs_config, jobs, sync)
    assert await poller._sweep_once() is True
    assert source.discover_calls == 1
    # Second sweep: queue still has the live job → skip without calling discover.
    assert await poller._sweep_once() is False
    assert source.discover_calls == 1


async def test_skipped_sweep_records_pending_work_reason(fs_config, jobs, sync):
    """last_skip_reason surfaces 'pending_work' while the queue is saturated
    and clears once the next sweep actually polls."""
    event = _event("file:///a.md", revision="r1")
    source = _StubSource("src", [[event], [event], []])
    poller = _periodic(source, fs_config, jobs, sync)

    await poller._sweep_once()  # first sweep enqueues, succeeds
    assert poller.last_skip_reason is None

    await poller._sweep_once()  # backpressure skips
    assert poller.last_skip_reason == "pending_work"

    # Drain the queue, sweep again, reason clears.
    claimed = await jobs.claim_next("worker")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "worker")
    await poller._sweep_once()
    assert poller.last_skip_reason is None


async def test_dead_job_does_not_clear_sync_state_revision(fs_config, jobs, sync):
    """When a job dies, the URI's previously-ingested revision must remain
    in sync_state so subsequent sweeps still see the URI as known."""
    await sync.upsert("src", "file:///a.md", revision="r1", content_hash="h1")

    changed = _event("file:///a.md", revision="r2")
    source = _StubSource("src", [[changed], [changed]])
    poller = _periodic(source, fs_config, jobs, sync)

    await poller._sweep_once()
    queued = await jobs.list_jobs()
    assert len(queued) == 1
    assert queued[0].revision == "r2"

    claimed = await jobs.claim_next("w")
    assert claimed is not None
    await jobs.mark_dead(claimed.id, "transient blew up", "w")

    row = await sync.get_row("src", "file:///a.md")
    assert row is not None
    assert row.revision == "r1"

    await poller._sweep_once()
    assert await sync.get_revision_snapshot("src") == {"file:///a.md": "r1"}


async def test_sweep_resumes_after_queue_drains(fs_config, jobs, sync):
    """Once the queue clears (success, dead, or cancel), sweeps resume."""
    event = _event("file:///a.md", revision="r1")
    source = _StubSource("src", [[event], []])
    poller = _periodic(source, fs_config, jobs, sync)
    await poller._sweep_once()

    claimed = await jobs.claim_next("worker")
    assert claimed is not None
    await jobs.mark_succeeded(claimed.id, "worker")

    assert await poller._sweep_once() is True
    assert source.discover_calls == 2


async def test_circuit_breaker_pauses_sweeps_after_failures(fs_config, jobs, sync):
    class _Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = _Clock()
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=2, cooldown_s=30.0),
        now_fn=clock,
    )
    source = _StubSource("src", [])
    source.fail_with = RuntimeError("upstream down")

    poller = _periodic(source, fs_config, jobs, sync, breaker=breaker)
    # Two failures open the breaker.
    assert await poller._sweep_once() is False
    assert await poller._sweep_once() is False
    assert breaker.is_open is True

    # Third call should be skipped — discover() is not invoked.
    before = source.discover_calls
    assert await poller._sweep_once() is False
    assert source.discover_calls == before


class _TimedSource(_StubSource):
    """Records when each listing ran."""

    def __init__(self, source_id: str):
        super().__init__(source_id, [])
        self.listed_at: list[datetime] = []

    async def discover(self, since=None, *, known_uris=None):
        self.listed_at.append(datetime.now(UTC))
        async for event in super().discover(since, known_uris=known_uris):
            yield event


async def test_sweep_records_last_polled_at_as_its_start(fs_config, jobs, sync):
    source = _TimedSource("src")
    poller = _periodic(source, fs_config, jobs, sync)
    assert poller.last_polled_at is None

    before = datetime.now(UTC)
    assert await poller._sweep_once() is True
    polled = poller.last_polled_at
    assert polled is not None
    assert before <= polled <= source.listed_at[0]


async def test_failed_or_skipped_sweep_keeps_last_polled_at(fs_config, jobs, sync):
    source = _TimedSource("src")
    poller = _periodic(source, fs_config, jobs, sync)
    assert await poller._sweep_once() is True
    polled = poller.last_polled_at

    source.fail_with = RuntimeError("upstream down")
    with capture_logs(_POLLER_LOGGER, logging.ERROR):
        assert await poller._sweep_once() is False
    assert poller.last_polled_at == polled

    source.fail_with = None
    assert await jobs.enqueue("src", "file:///a.md", JobOp.UPSERT)
    assert await poller._sweep_once() is False
    assert poller.last_skip_reason == "pending_work"
    assert poller.last_polled_at == polled


async def test_dry_run_records_last_polled_at_as_its_start(fs_config, jobs, sync):
    source = _TimedSource("src")
    poller = _periodic(source, fs_config, jobs, sync)

    before = datetime.now(UTC)
    ok, _, _ = await poller._dry_run_once()
    assert ok is True
    polled = poller.last_polled_at
    assert polled is not None
    assert before <= polled <= source.listed_at[0]


async def test_per_source_retry_policy_overrides_default(jobs, sync, tmp_path):
    from haiku.rag.config import RetryPolicyConfig

    cfg = FSSourceConfig(
        type="fs",
        id="src",
        root=tmp_path,
        retry=RetryPolicyConfig(max_attempts=9),
    )
    source = _StubSource("src", [[_event("file:///a.md")]])
    poller = _periodic(source, cfg, jobs, sync, default_max_attempts=3)
    await poller._sweep_once()
    queued = await jobs.list_jobs(source_id="src")
    assert queued[0].max_attempts == 9


# --- dry-run collection ---


async def test_dry_run_reports_changes_without_mutating_queue_or_sync(
    fs_config, jobs, sync
):
    source = _StubSource(
        "src",
        [
            [
                _event("file:///a.md", revision="r1"),
                _event(
                    "file:///b.md",
                    kind=SourceEventKind.UNCHANGED,
                    revision="r2",
                ),
                _event("file:///gone.md", kind=SourceEventKind.DELETE),
            ]
        ],
    )
    poller = _periodic(source, fs_config, jobs, sync)

    ok, summary, changes = await poller._dry_run_once()

    assert ok is True
    assert summary.upsert_count == 1
    assert summary.delete_count == 1
    assert summary.unchanged_count == 1
    assert summary.ignored_delete_count == 0
    assert [(c.op, c.uri, c.revision) for c in changes] == [
        (JobOp.UPSERT, "file:///a.md", "r1"),
        (JobOp.DELETE, "file:///gone.md", None),
    ]
    assert await jobs.list_jobs(source_id="src") == []
    assert await sync.list_known_uris("src") == set()


async def test_a_delete_carries_its_revision(jobs, sync, fs_config):
    delete = SourceEvent(
        source_id="src",
        uri="https://example.com/a.md",
        kind=SourceEventKind.DELETE,
        revision="http-404",
        discovered_at=datetime.now(UTC),
    )
    sweeps = [[_event("https://example.com/b.md"), delete]]
    source = _StubSource("src", sweeps * 2)
    poller = _periodic(source, fs_config, jobs, sync)

    ok, _, changes = await poller._dry_run_once()
    assert ok is True
    assert [(c.op, c.revision) for c in changes if c.op is JobOp.DELETE] == [
        (JobOp.DELETE, "http-404")
    ]

    assert await poller._sweep_once() is True
    [job] = [j for j in await jobs.list_jobs(source_id="src") if j.op is JobOp.DELETE]
    assert job.revision == "http-404"


async def test_dry_run_counts_ignored_deletes_when_orphan_delete_disabled(
    fs_config, jobs, sync
):
    cfg = fs_config.model_copy(update={"delete_orphans": False})
    source = _StubSource(
        "src", [[_event("file:///gone.md", kind=SourceEventKind.DELETE)]]
    )
    poller = _periodic(source, cfg, jobs, sync)

    ok, summary, changes = await poller._dry_run_once()

    assert ok is True
    assert summary.delete_count == 0
    assert summary.ignored_delete_count == 1
    assert changes == []
    assert await jobs.list_jobs(source_id="src") == []


async def test_dry_run_fails_on_empty_listing_with_deletes(fs_config, jobs, sync):
    source = _StubSource(
        "src", [[_event("file:///gone.md", kind=SourceEventKind.DELETE)]]
    )
    poller = _periodic(source, fs_config, jobs, sync)

    with capture_logs(_POLLER_LOGGER, logging.ERROR):
        ok, summary, changes = await poller._dry_run_once()

    assert ok is False
    assert changes == []
    assert summary.delete_count == 0
    assert poller._breaker.consecutive_failures == 1


async def test_dry_run_returns_no_changes_when_listing_fails_midway(
    fs_config, jobs, sync
):
    poller = _periodic(_FailsMidway("src"), fs_config, jobs, sync)

    with capture_logs(_POLLER_LOGGER, logging.ERROR):
        ok, summary, changes = await poller._dry_run_once()

    assert ok is False
    assert changes == []
    assert summary.upsert_count == 0


async def test_dry_run_skips_when_queue_has_pending_work(fs_config, jobs, sync):
    await jobs.enqueue("src", "file:///already.md", op=JobOp.UPSERT)
    source = _StubSource("src", [[_event("file:///a.md")]])
    poller = _periodic(source, fs_config, jobs, sync)

    ok, summary, changes = await poller._dry_run_once()

    assert ok is False
    assert summary.source_id == "src"
    assert changes == []
    assert source.discover_calls == 0
    assert poller.last_skip_reason == "pending_work"


# --- PollerManager lifecycle ---


async def test_manager_builds_pollers_per_source(tmp_path, jobs, sync):
    """When SourceConfig.id is set, the poller's source uses it verbatim;
    when omitted, the adapter auto-derives one from its target."""
    configs = [
        FSSourceConfig(type="fs", root=tmp_path),
        S3SourceConfig(type="s3", uri="s3://bucket/"),
        HTTPSourceConfig(type="http", id="urls", urls=[]),
        WebDAVSourceConfig(
            type="webdav", id="nc", base_url="https://nc.example.com/dav/"
        ),
    ]
    manager = PollerManager(
        configs=configs,
        job_repo=jobs,
        sync_repo=sync,
    )
    built = manager.pollers
    assert len(built) == 4
    assert {p.source_id for p in built} == {
        f"fs:{tmp_path.resolve()}",
        "s3:bucket/",
        "urls",
        "nc",
    }


async def test_manager_sources_available_at_construction(tmp_path, jobs, sync):
    """PollerManager builds Sources eagerly so callers (WorkerPool) can
    receive them by plain construction order."""
    from haiku.rag.config import SourceConfig
    from haiku.rag.sources.http import HTTPSource

    configs: list[SourceConfig] = [
        FSSourceConfig(type="fs", id="docs", root=tmp_path),
        HTTPSourceConfig(
            type="http",
            id="urls",
            urls=[],
            headers={"Authorization": "Bearer abc"},
        ),
    ]
    manager = PollerManager(configs=configs, job_repo=jobs, sync_repo=sync)
    sources = manager.sources
    assert len(sources) == 2
    assert {s.source_id for s in sources} == {"docs", "urls"}
    http = next(s for s in sources if s.source_id == "urls")
    assert isinstance(http, HTTPSource)
    assert http.headers == {"Authorization": "Bearer abc"}


async def test_manager_double_start_raises(tmp_path, jobs, sync):
    cfg = FSSourceConfig(
        type="fs",
        id="local",
        root=tmp_path,
        poll_interval_s=60.0,
    )
    manager = PollerManager(configs=[cfg], job_repo=jobs, sync_repo=sync)
    await manager.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            await manager.start()
    finally:
        await manager.stop()


async def test_manager_restart_resumes_polling(tmp_path, jobs, sync):
    """stop() then start() produces a working poller that stays alive after
    its initial sweep — the second cycle's stop event is fresh, not the
    set state left over from the previous stop()."""
    (tmp_path / "a.md").write_text("hello")
    cfg = FSSourceConfig(
        type="fs",
        id="local",
        root=tmp_path,
        poll_interval_s=60.0,
    )
    manager = PollerManager(
        configs=[cfg], job_repo=jobs, sync_repo=sync, supported_extensions=[".md"]
    )

    await manager.start()
    await asyncio.sleep(0.1)
    await manager.stop()

    await manager.start()
    try:
        # The poller task must STAY alive after its initial sweep so the
        # watchfiles + periodic-sweep loops keep running. live_pollers
        # drops to 0 immediately if run() exited because _stop was set.
        await asyncio.sleep(0.1)
        assert manager.live_pollers == 1
    finally:
        await manager.stop()


# --- FSPoller._handle_watch_change ---


def _fs_poller(tmp_path, jobs, sync):
    """Construct an FSPoller for unit-testing the watch-change handler.
    Doesn't start the watch loop — tests call `_handle_watch_change` directly."""
    from haiku.rag.ingester.pollers.fs import FSPoller
    from haiku.rag.sources.fs import FSSource

    cfg = FSSourceConfig(
        type="fs",
        id="local",
        root=tmp_path,
        delete_orphans=True,
        poll_interval_s=60.0,
    )
    source = FSSource(root=tmp_path, supported_extensions=[".md"], source_id="local")
    return FSPoller(
        source=source,
        config=cfg,
        job_repo=jobs,
        sync_repo=sync,
    )


async def test_watch_deleted_enqueues_when_file_truly_gone(tmp_path, jobs, sync):
    from watchfiles import Change

    poller = _fs_poller(tmp_path, jobs, sync)
    missing = tmp_path / "gone.md"
    # file doesn't exist; deleted event should enqueue DELETE
    await poller._handle_watch_change(Change.deleted, missing)

    queued = await jobs.list_jobs(source_id="local")
    assert len(queued) == 1
    assert queued[0].op is JobOp.DELETE


async def test_watch_deleted_skipped_when_file_already_back(tmp_path, jobs, sync):
    """`git checkout` and atomic-rename saves fire (deleted, added) in quick
    succession; by the time the deleted event reaches us the file is back.
    Skipping the DELETE keeps the follow-up Change.added's UPSERT from being
    blocked by the live-row index."""
    from watchfiles import Change

    poller = _fs_poller(tmp_path, jobs, sync)
    present = tmp_path / "restored.md"
    present.write_text("restored")

    await poller._handle_watch_change(Change.deleted, present)

    assert await jobs.list_jobs(source_id="local") == []


async def test_watch_deleted_then_added_enqueues_upsert(tmp_path, jobs, sync):
    """End-to-end of the git-checkout scenario: (deleted, added) for an
    existing file leaves a single UPSERT job, not a DELETE."""
    from watchfiles import Change

    poller = _fs_poller(tmp_path, jobs, sync)
    path = tmp_path / "doc.md"
    path.write_text("contents")

    await poller._handle_watch_change(Change.deleted, path)
    await poller._handle_watch_change(Change.added, path)

    queued = await jobs.list_jobs(source_id="local")
    assert len(queued) == 1
    assert queued[0].op is JobOp.UPSERT


async def test_watch_change_waits_for_the_sweep_lock(tmp_path, jobs, sync):
    from watchfiles import Change

    poller = _fs_poller(tmp_path, jobs, sync)
    path = tmp_path / "doc.md"
    path.write_text("contents")

    async with poller._sweep_lock:
        handling = asyncio.create_task(poller._handle_watch_change(Change.added, path))
        await asyncio.sleep(0.05)
        assert await jobs.list_jobs(source_id="local") == []
    await asyncio.wait_for(handling, timeout=5.0)

    assert len(await jobs.list_jobs(source_id="local")) == 1


async def test_watch_added_file_deleted_before_stat_does_not_crash(
    tmp_path, jobs, sync
):
    """If a file is deleted between the watchfiles event and the stat()
    call, the handler should return silently instead of raising
    FileNotFoundError and killing the watch loop."""
    from watchfiles import Change

    poller = _fs_poller(tmp_path, jobs, sync)
    missing = tmp_path / "vanished.md"
    # File doesn't exist — simulate Change.added arriving after deletion.
    await poller._handle_watch_change(Change.added, missing)

    # No job should be enqueued, and no exception should have propagated.
    assert await jobs.list_jobs(source_id="local") == []


# --- FSPoller end-to-end smoke ---


async def test_fs_poller_enqueues_initial_files(tmp_path, jobs, sync):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    cfg = FSSourceConfig(type="fs", id="local", root=tmp_path, poll_interval_s=60.0)
    manager = PollerManager(
        configs=[cfg], job_repo=jobs, sync_repo=sync, supported_extensions=[".md"]
    )
    await manager.start()
    try:
        # Wait for the initial sweep to land jobs.
        for _ in range(40):
            queued = await jobs.list_jobs(source_id="local")
            if len(queued) == 2:
                break
            await asyncio.sleep(0.05)
    finally:
        await manager.stop()

    queued = await jobs.list_jobs(source_id="local")
    assert {Path(j.uri).name for j in queued} == {"a.md", "b.md"}
    assert all(j.status is JobStatus.QUEUED for j in queued)


# --- _dry_run_once ---


async def test_dry_run_collects_changes_without_writing(fs_config, jobs, sync):
    source = _StubSource(
        "src",
        [
            [
                _event("file:///a.md"),
                _event("file:///b.md", kind=SourceEventKind.UNCHANGED),
                _event("file:///c.md", kind=SourceEventKind.DELETE),
            ]
        ],
    )
    poller = _periodic(source, fs_config, jobs, sync)

    ok, summary, changes = await poller._dry_run_once()

    assert ok is True
    assert summary.upsert_count == 1
    assert summary.unchanged_count == 1
    assert summary.delete_count == 1
    assert {c.op for c in changes} == {JobOp.UPSERT, JobOp.DELETE}
    # A dry run must not touch the queue.
    assert await jobs.list_jobs(source_id="src") == []


async def test_dry_run_ignores_deletes_when_delete_orphans_false(
    fs_config, jobs, sync, tmp_path
):
    config = FSSourceConfig(
        type="fs",
        id="src",
        root=tmp_path,
        delete_orphans=False,
        poll_interval_s=0.05,
    )
    source = _StubSource("src", [[_event("file:///c.md", kind=SourceEventKind.DELETE)]])
    poller = _periodic(source, config, jobs, sync)

    ok, summary, changes = await poller._dry_run_once()

    assert ok is True
    assert summary.delete_count == 0
    assert summary.ignored_delete_count == 1
    assert changes == []


async def test_dry_run_skipped_when_circuit_open(fs_config, jobs, sync):
    class _Clock:
        now = 0.0

        def __call__(self):
            return self.now

    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=1, cooldown_s=30.0),
        now_fn=_Clock(),
    )
    source = _StubSource("src", [])
    source.fail_with = RuntimeError("upstream down")
    poller = _periodic(source, fs_config, jobs, sync, breaker=breaker)

    assert await poller._sweep_once() is False
    assert breaker.is_open is True

    before = source.discover_calls
    ok, summary, changes = await poller._dry_run_once()

    assert ok is False
    assert changes == []
    assert source.discover_calls == before
    assert poller.last_skip_reason == "circuit_open"


async def test_dry_run_records_failure_when_discover_raises(fs_config, jobs, sync):
    source = _StubSource("src", [])
    source.fail_with = RuntimeError("upstream down")
    poller = _periodic(source, fs_config, jobs, sync)

    ok, summary, changes = await poller._dry_run_once()

    assert ok is False
    assert changes == []
    assert poller._breaker.consecutive_failures == 1


async def test_dry_run_skipped_when_queue_has_pending_work(fs_config, jobs, sync):
    source = _StubSource("src", [[_event("file:///a.md")]])
    poller = _periodic(source, fs_config, jobs, sync)
    await jobs.enqueue("src", "file:///pending.md", JobOp.UPSERT)

    ok, _summary, changes = await poller._dry_run_once()

    assert ok is False
    assert changes == []
    assert poller.last_skip_reason == "pending_work"


async def test_watch_deleted_skipped_when_delete_orphans_false(tmp_path, jobs, sync):
    from watchfiles import Change

    from haiku.rag.ingester.pollers.fs import FSPoller
    from haiku.rag.sources.fs import FSSource

    cfg = FSSourceConfig(
        type="fs",
        id="local",
        root=tmp_path,
        delete_orphans=False,
        poll_interval_s=60.0,
    )
    poller = FSPoller(
        source=FSSource(root=tmp_path, supported_extensions=[".md"], source_id="local"),
        config=cfg,
        job_repo=jobs,
        sync_repo=sync,
    )

    await poller._handle_watch_change(Change.deleted, tmp_path / "gone.md")

    assert await jobs.list_jobs(source_id="local") == []


async def test_dry_run_manifest_reports_failed_sources(tmp_path, jobs, sync):
    """A source whose discover() raises is named in the failed list while the
    manifest still carries the sources that succeeded."""
    manager = PollerManager(
        configs=[FSSourceConfig(type="fs", id="ok", root=tmp_path)],
        job_repo=jobs,
        sync_repo=sync,
    )
    broken = _StubSource("broken", [])
    broken.fail_with = RuntimeError("upstream down")
    manager._pollers.append(
        _periodic(
            broken,
            FSSourceConfig(type="fs", id="broken", root=tmp_path, poll_interval_s=60.0),
            jobs,
            sync,
        )
    )

    manifest, failed = await manager.dry_run_manifest()

    assert failed == ["broken"]
    assert {s.source_id for s in manifest.sources} == {"ok", "broken"}


async def test_dry_run_manifest_leaves_out_a_failed_sources_partial_changes(
    tmp_path, jobs, sync
):
    (tmp_path / "a.md").write_text("alpha")
    manager = PollerManager(
        configs=[FSSourceConfig(type="fs", id="ok", root=tmp_path)],
        job_repo=jobs,
        sync_repo=sync,
    )
    manager._pollers.append(
        _periodic(
            _FailsMidway("broken"),
            FSSourceConfig(type="fs", id="broken", root=tmp_path, poll_interval_s=60.0),
            jobs,
            sync,
        )
    )

    with capture_logs(_POLLER_LOGGER, logging.ERROR):
        manifest, failed = await manager.dry_run_manifest()

    assert failed == ["broken"]
    assert {c.source_id for c in manifest.changes} == {"ok"}


# --- last error ---


async def test_a_failed_sweep_records_its_error_until_a_sweep_succeeds(
    fs_config, jobs, sync
):
    source = _StubSource("src", [])
    source.fail_with = RuntimeError("upstream down")
    poller = _periodic(source, fs_config, jobs, sync)
    before = datetime.now(UTC)

    with capture_logs(_POLLER_LOGGER, logging.ERROR) as records:
        assert await poller._sweep_once() is False

    assert [r.getMessage() for r in records] == [
        "discover() failed for src (consecutive=1): upstream down"
    ]
    assert poller.last_error == "upstream down"
    assert poller.last_error_at is not None and poller.last_error_at >= before

    source.fail_with = None
    assert await poller._sweep_once() is True

    assert (poller.last_error, poller.last_error_at) == (None, None)


async def test_a_failed_dry_run_records_its_error_until_one_succeeds(
    fs_config, jobs, sync
):
    source = _StubSource("src", [])
    source.fail_with = EmptyListingError("src", 3)
    poller = _periodic(source, fs_config, jobs, sync)

    with capture_logs(_POLLER_LOGGER, logging.ERROR) as records:
        ok, _summary, _changes = await poller._dry_run_once()

    assert ok is False
    assert [r.getMessage() for r in records] == [
        "dry-run discover() failed for src (consecutive=1): "
        + str(EmptyListingError("src", 3))
    ]
    assert poller.last_error == str(EmptyListingError("src", 3))

    source.fail_with = None
    ok, _summary, _changes = await poller._dry_run_once()

    assert ok is True
    assert poller.last_error is None


async def test_an_error_without_a_message_is_named_by_its_type(fs_config, jobs, sync):
    source = _StubSource("src", [])
    source.fail_with = TimeoutError()
    poller = _periodic(source, fs_config, jobs, sync)

    with capture_logs(_POLLER_LOGGER, logging.ERROR) as records:
        await poller._sweep_once()

    assert [r.getMessage() for r in records] == [
        "discover() failed for src (consecutive=1): "
    ]

    assert poller.last_error == "TimeoutError"


async def test_a_failed_watch_loop_records_its_error(tmp_path, jobs, sync, monkeypatch):
    async def _failing_watch(*_, **__):
        raise RuntimeError("inotify limit reached")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("haiku.rag.ingester.pollers.fs.awatch", _failing_watch)
    poller = _fs_poller(tmp_path, jobs, sync)

    with capture_logs(
        logging.getLogger("haiku.rag.ingester.pollers.fs"), logging.ERROR
    ) as records:
        await poller._watch_loop()

    assert [r.getMessage() for r in records] == [
        "watchfiles loop failed for local: inotify limit reached"
    ]
    assert poller.last_error == "inotify limit reached"


# --- stopping and removal ---


class _GatedSource(_StubSource):
    """Lists one file once `gate` is set, recording that discovery began."""

    def __init__(self, source_id: str):
        super().__init__(source_id, [])
        self.listing = asyncio.Event()
        self.gate = asyncio.Event()

    async def discover(self, since=None, *, known_uris=None):
        self.discover_calls += 1
        self.listing.set()
        await self.gate.wait()
        yield _event("file:///a.md", source_id=self.source_id)


async def test_stop_waits_for_the_sweep_in_progress(jobs, sync, fs_config):
    source = _GatedSource("src")
    poller = _periodic(source, fs_config, jobs, sync)
    sweep = asyncio.create_task(poller._sweep_once())
    await source.listing.wait()

    stop = asyncio.create_task(poller.stop())
    await asyncio.sleep(0.05)
    assert not stop.done()

    source.gate.set()
    await asyncio.wait_for(stop, timeout=5.0)
    assert sweep.done() and sweep.result() is True


async def test_a_sweep_after_stop_does_not_list(jobs, sync, fs_config):
    source = _StubSource("src", [[_event("file:///a.md")]])
    poller = _periodic(source, fs_config, jobs, sync)

    await poller.stop()

    assert await poller._sweep_once() is False
    assert source.discover_calls == 0
    assert await jobs.list_jobs(source_id="src") == []


def _idle_config(tmp_path, source_id: str) -> FSSourceConfig:
    return FSSourceConfig(type="fs", id=source_id, root=tmp_path, poll_interval_s=3600)


async def test_remove_stops_one_poller_and_keeps_the_others(tmp_path, jobs, sync):
    manager = PollerManager(
        configs=[_idle_config(tmp_path, "kept"), _idle_config(tmp_path, "gone")],
        job_repo=jobs,
        sync_repo=sync,
    )
    await manager.start()
    try:
        await manager.remove(["gone"])

        assert [poller.source_id for poller in manager.pollers] == ["kept"]
        assert manager.live_pollers == 1
    finally:
        await manager.stop()


async def test_a_poller_added_to_a_started_manager_runs(tmp_path, jobs, sync):
    manager = PollerManager(
        configs=[_idle_config(tmp_path, "first")], job_repo=jobs, sync_repo=sync
    )
    await manager.start()
    try:
        manager.add([manager.build(_idle_config(tmp_path, "second"))])

        assert [poller.source_id for poller in manager.pollers] == ["first", "second"]
        assert manager.live_pollers == 2
    finally:
        await manager.stop()
    assert manager.live_pollers == 0

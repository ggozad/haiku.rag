import asyncio
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest

from haiku.rag.config import (
    CircuitBreakerConfig,
    FSSourceConfig,
    HTTPSourceConfig,
    S3SourceConfig,
    WebDAVSourceConfig,
)
from haiku.rag.ingester.pollers.circuit_breaker import CircuitBreaker
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.pollers.periodic import PeriodicPoller
from haiku.rag.ingester.queue.migrations import apply_migrations
from haiku.rag.ingester.queue.models import JobOp, JobStatus
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.sources.base import (
    FetchResult,
    SourceEvent,
    SourceEventKind,
)


@pytest.fixture
async def conn(tmp_path):
    path = tmp_path / "queue.db"
    connection = await aiosqlite.connect(str(path))
    connection.row_factory = aiosqlite.Row
    await apply_migrations(connection)
    yield connection
    await connection.close()


@pytest.fixture
def jobs(conn):
    return JobRepo(conn)


@pytest.fixture
def sync(conn):
    return SyncStateRepo(conn)


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

    async def discover(self, since=None):
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


# --- _sweep_once / event handling on the base class via PeriodicPoller ---


@pytest.mark.asyncio
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
    # after a successful ingest. But last_seen_at is bumped.
    snapshot = await sync.get_snapshot("src")
    assert snapshot == {}  # revision left empty by the poller


@pytest.mark.asyncio
async def test_unchanged_event_touches_sync_state_no_job(fs_config, jobs, sync):
    source = _StubSource(
        "src", [[_event("file:///a.md", kind=SourceEventKind.UNCHANGED, revision="r1")]]
    )
    poller = _periodic(source, fs_config, jobs, sync)
    await poller._sweep_once()

    assert await jobs.list_jobs(source_id="src") == []
    assert await sync.get_snapshot("src") == {"file:///a.md": "r1"}


@pytest.mark.asyncio
async def test_delete_event_enqueues_delete_job(fs_config, jobs, sync):
    source = _StubSource(
        "src", [[_event("file:///gone.md", kind=SourceEventKind.DELETE)]]
    )
    poller = _periodic(source, fs_config, jobs, sync)
    await poller._sweep_once()

    queued = await jobs.list_jobs(source_id="src")
    assert len(queued) == 1
    assert queued[0].op is JobOp.DELETE


@pytest.mark.asyncio
async def test_delete_event_skipped_when_delete_orphans_false(fs_config, jobs, sync):
    fs_config = fs_config.model_copy(update={"delete_orphans": False})
    source = _StubSource(
        "src", [[_event("file:///gone.md", kind=SourceEventKind.DELETE)]]
    )
    poller = _periodic(source, fs_config, jobs, sync)
    await poller._sweep_once()
    assert await jobs.list_jobs(source_id="src") == []


@pytest.mark.asyncio
async def test_repeated_sweep_does_not_duplicate_jobs(fs_config, jobs, sync):
    """Live-uniqueness: enqueueing the same (source_id, uri, op) when a job
    is already queued/claimed is a no-op."""
    event = _event("file:///a.md", revision="r1")
    source = _StubSource("src", [[event], [event]])
    poller = _periodic(source, fs_config, jobs, sync)
    await poller._sweep_once()
    await poller._sweep_once()
    queued = await jobs.list_jobs(source_id="src")
    assert len(queued) == 1


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_sweep_records_last_polled_at_on_success(fs_config, jobs, sync):
    source = _StubSource("src", [[]])
    poller = _periodic(source, fs_config, jobs, sync)
    assert poller.last_polled_at is None
    await poller._sweep_once()
    assert poller.last_polled_at is not None
    assert poller.last_polled_at.tzinfo is not None


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_storage_options_thread_through_to_job_extra(jobs, sync):
    cfg = S3SourceConfig(
        type="s3",
        id="bucket",
        uri="s3://bucket/",
        storage_options={"endpoint": "http://seaweed:8333"},
    )
    source = _StubSource(
        "bucket", [[_event("s3://bucket/file.md", source_id="bucket")]]
    )
    poller = _periodic(source, cfg, jobs, sync)
    await poller._sweep_once()
    queued = await jobs.list_jobs(source_id="bucket")
    assert queued[0].extra == {"storage_options": {"endpoint": "http://seaweed:8333"}}


@pytest.mark.asyncio
async def test_http_headers_thread_through_to_job_extra(jobs, sync):
    cfg = HTTPSourceConfig(
        type="http",
        id="auth",
        urls=["https://example.com/a.md"],
        headers={"Authorization": "Bearer abc"},
    )
    source = _StubSource(
        "auth", [[_event("https://example.com/a.md", source_id="auth")]]
    )
    poller = _periodic(source, cfg, jobs, sync)
    await poller._sweep_once()
    queued = await jobs.list_jobs(source_id="auth")
    assert queued[0].extra == {"headers": {"Authorization": "Bearer abc"}}


# --- PollerManager lifecycle ---


@pytest.mark.asyncio
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
    built = manager.build_pollers()
    assert len(built) == 4
    assert {p.source_id for p in built} == {
        f"fs:{tmp_path.resolve()}",
        "s3:bucket/",
        "urls",
        "nc",
    }


@pytest.mark.asyncio
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


# --- FSPoller end-to-end smoke ---


@pytest.mark.asyncio
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

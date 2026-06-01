"""IngesterApp lifecycle: run_batch (one sweep -> drain -> exit) and serve
(pollers + workers + API until shutdown). The document store (HaikuRAG) is
patched out — the behavior under test is the orchestration and orphan
pruning, not embedding."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import unquote, urlparse

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.exceptions import UnsupportedSourceError
from haiku.rag.config import (
    APIConfig,
    AppConfig,
    FSSourceConfig,
    IngesterConfig,
    QueueConfig,
    RetryPolicyConfig,
    WorkerConfig,
)
from haiku.rag.ingester.app import IngesterApp
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.store.models.document import Document


def _config(tmp_path, **worker_kwargs) -> AppConfig:
    workers = WorkerConfig(
        worker_count=2,
        poll_idle_interval_s=0.05,
        retry=RetryPolicyConfig(max_attempts=1),
        **worker_kwargs,
    )
    return AppConfig(
        ingester=IngesterConfig(
            queue=QueueConfig(path=tmp_path / "queue.db"),
            sources=[
                FSSourceConfig(
                    type="fs",
                    id="local",
                    root=tmp_path,
                    poll_interval_s=3600.0,
                )
            ],
            workers=workers,
            api=APIConfig(enabled=False),
        ),
    )


def _mock_client() -> AsyncMock:
    """A HaikuRAG mock whose upserts return a fresh Document (with an md5 so
    the worker records sync_state) and whose lookups return a Document with a
    deterministic id so the DELETE path resolves."""
    client = AsyncMock(spec=HaikuRAG)
    counter = {"n": 0}

    async def _create(uri, *_, **__):
        counter["n"] += 1
        # Mirror the real client: persist the FS revision (mtime_ns) so the
        # poller's change-detection skips unchanged files on the next sweep.
        path = Path(unquote(urlparse(uri).path))
        return Document(
            id=f"doc-{counter['n']}",
            content="x",
            uri=uri,
            metadata={
                "content_type": "text/markdown",
                "md5": f"md5-{counter['n']}",
                "source_revision": str(path.stat().st_mtime_ns),
            },
        )

    async def _get_by_uri(uri):
        return Document(id=f"doc-for-{uri}", content="x", uri=uri, metadata={})

    client.create_document_from_source.side_effect = _create
    client.get_document_by_uri.side_effect = _get_by_uri
    return client


@pytest.fixture
def use_client(monkeypatch):
    """Make IngesterApp's internally-created HaikuRAG resolve to the given
    mock client. Call again to swap the client for a later run in the same
    test."""

    def _install(client):
        @asynccontextmanager
        async def _cm(*_, **__):
            yield client

        monkeypatch.setattr("haiku.rag.client.HaikuRAG", lambda *a, **k: _cm())

    return _install


@pytest.mark.asyncio
async def test_run_batch_drains_upserts(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    client = _mock_client()
    use_client(client)

    report = await IngesterApp(
        config=_config(tmp_path), db_path=tmp_path / "db.lancedb"
    ).run_batch()

    assert report.succeeded == 2
    assert report.dead == 0
    assert client.create_document_from_source.await_count == 2
    ingested = {
        call.args[0] for call in client.create_document_from_source.await_args_list
    }
    assert ingested == {
        (tmp_path / "a.md").as_uri(),
        (tmp_path / "b.md").as_uri(),
    }


@pytest.mark.asyncio
async def test_run_batch_prunes_orphans(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")
    (tmp_path / "b.md").write_text("world")

    client = _mock_client()
    use_client(client)
    config = _config(tmp_path)
    db_path = tmp_path / "db.lancedb"

    # First batch ingests both files and records sync_state for each.
    first = await IngesterApp(config=config, db_path=db_path).run_batch()
    assert first.succeeded == 2
    client.delete_document.assert_not_awaited()

    # b.md disappears from the source tree. The next sweep sees it in
    # sync_state but not on disk -> enqueues a DELETE for it.
    (tmp_path / "b.md").unlink()

    second = await IngesterApp(config=config, db_path=db_path).run_batch()

    # a.md is unchanged (same mtime) so it's not re-ingested; only the orphan
    # delete runs.
    assert client.create_document_from_source.await_count == 2
    client.delete_document.assert_awaited_once()
    assert second.succeeded == 1
    assert second.dead == 0


@pytest.mark.asyncio
async def test_run_batch_reports_dead_on_permanent_failure(tmp_path, use_client):
    (tmp_path / "a.md").write_text("hello")

    client = _mock_client()
    client.create_document_from_source.side_effect = UnsupportedSourceError("nope")
    use_client(client)

    report = await IngesterApp(
        config=_config(tmp_path), db_path=tmp_path / "db.lancedb"
    ).run_batch()

    assert report.succeeded == 0
    assert report.dead == 1


@pytest.mark.asyncio
async def test_run_batch_recovered_doc_is_not_counted_as_dead(tmp_path, use_client):
    """A doc that dead-lettered in an earlier run and succeeds in this one
    prunes its dead row. The report counts only this run's terminal jobs, so
    the recovery shows as a success, never a negative or stale dead count."""
    (tmp_path / "a.md").write_text("hello")
    config = _config(tmp_path)
    db_path = tmp_path / "db.lancedb"

    failing = _mock_client()
    failing.create_document_from_source.side_effect = UnsupportedSourceError("nope")
    use_client(failing)
    first = await IngesterApp(config=config, db_path=db_path).run_batch()
    assert first.dead == 1

    healthy = _mock_client()
    use_client(healthy)
    second = await IngesterApp(config=config, db_path=db_path).run_batch()
    assert second.dead == 0
    assert second.succeeded == 1


@pytest.mark.asyncio
async def test_run_batch_reports_failed_sweep(
    tmp_path, use_client, monkeypatch, caplog
):
    """A source whose discover() raises is reported in failed_sweeps so the
    run can be treated as failed rather than a silent empty success."""
    (tmp_path / "a.md").write_text("hello")
    use_client(_mock_client())

    async def _failing_discover(self, **kwargs):
        raise RuntimeError("discover blew up")
        yield  # unreachable; makes this an async generator

    monkeypatch.setattr(
        "haiku.rag.ingester.sources.fs.FSSource.discover", _failing_discover
    )

    with caplog.at_level("ERROR", logger="haiku.rag.ingester.pollers.base"):
        report = await IngesterApp(
            config=_config(tmp_path), db_path=tmp_path / "db.lancedb"
        ).run_batch()

    assert report.failed_sweeps == ["local"]
    assert report.succeeded == 0
    assert report.dead == 0
    assert "discover() failed" in caplog.text


@pytest.mark.asyncio
async def test_run_batch_empty_source_returns_immediately(tmp_path, use_client):
    client = _mock_client()
    use_client(client)

    report = await asyncio.wait_for(
        IngesterApp(
            config=_config(tmp_path), db_path=tmp_path / "db.lancedb"
        ).run_batch(),
        timeout=5.0,
    )

    assert report.succeeded == 0
    assert report.dead == 0
    client.create_document_from_source.assert_not_awaited()


async def _wait_until(predicate, *, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached within timeout")
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.parametrize("api", [True, False])
async def test_serve_starts_workers_pollers_and_shuts_down(tmp_path, use_client, api):
    """serve brings up pollers, workers and (when enabled) the HTTP API, then
    tears them all down. Shutdown is driven here by cancelling the serve task,
    which runs the same drain-and-close path as a SIGINT/SIGTERM."""
    use_client(_mock_client())
    config = _config(tmp_path)
    config.ingester.api = APIConfig(enabled=api, host="127.0.0.1", port=0)
    app = IngesterApp(config=config, db_path=tmp_path / "db.lancedb")

    task = asyncio.create_task(app.serve(api=api))
    try:
        await _wait_until(
            lambda: (
                app._pool is not None
                and app._pool.live_workers > 0
                and app._pollers is not None
                and app._pollers.live_pollers > 0
            )
        )
    finally:
        task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)

    # Workers and pollers are stopped after the shutdown path runs.
    assert app._pool is not None and app._pollers is not None
    assert app._pool.live_workers == 0
    assert app._pollers.live_pollers == 0


class _SlowPool(WorkerPool):
    """A WorkerPool whose stop never finishes within the grace, to exercise
    _stop_pool's timeout path. The real wiring is bypassed since _stop_pool
    only calls stop() and drain_pending_releases()."""

    def __init__(self) -> None:
        self.released = 0

    async def stop(self) -> None:
        await asyncio.sleep(1.0)

    async def drain_pending_releases(self, timeout: float = 2.0) -> int:
        self.released += 1
        return 2


@pytest.mark.asyncio
async def test_stop_pool_warns_when_shutdown_grace_elapses(tmp_path, caplog):
    """When a worker doesn't stop within the shutdown grace, _stop_pool logs a
    warning and still drains any pending cancel-cleanup releases."""
    config = _config(tmp_path, shutdown_grace_s=0.01)
    app = IngesterApp(config=config, db_path=tmp_path / "db.lancedb")
    pool = _SlowPool()
    app._pool = pool

    with caplog.at_level("WARNING", logger="haiku.rag.ingester.app"):
        await app._stop_pool()

    assert pool.released == 1
    assert "Shutdown grace" in caplog.text


def _record_close_order(monkeypatch) -> list[str]:
    """Record the order of _stop_pool and PollerManager.close_sources.

    Workers share the pollers' Source instances for fetch(), so the source
    clients must be closed only after the pool has stopped — otherwise an
    in-flight fetch during the shutdown grace hits a closed client.
    """
    order: list[str] = []
    orig_stop_pool = IngesterApp._stop_pool
    orig_close = PollerManager.close_sources

    async def rec_stop(self):
        order.append("stop_pool")
        await orig_stop_pool(self)

    async def rec_close(self):
        order.append("close_sources")
        await orig_close(self)

    monkeypatch.setattr(IngesterApp, "_stop_pool", rec_stop)
    monkeypatch.setattr(PollerManager, "close_sources", rec_close)
    return order


@pytest.mark.asyncio
async def test_run_batch_closes_sources_after_pool_stops(
    tmp_path, use_client, monkeypatch
):
    (tmp_path / "a.md").write_text("hello")
    use_client(_mock_client())
    app = IngesterApp(config=_config(tmp_path), db_path=tmp_path / "db.lancedb")
    order = _record_close_order(monkeypatch)

    await app.run_batch()

    assert order == ["stop_pool", "close_sources"]


@pytest.mark.asyncio
async def test_serve_closes_sources_after_pool_stops(tmp_path, use_client, monkeypatch):
    use_client(_mock_client())
    config = _config(tmp_path)
    config.ingester.api = APIConfig(enabled=False)
    app = IngesterApp(config=config, db_path=tmp_path / "db.lancedb")
    order = _record_close_order(monkeypatch)

    task = asyncio.create_task(app.serve(api=False))
    try:
        await _wait_until(lambda: app._pool is not None and app._pool.live_workers > 0)
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)

    assert order == ["stop_pool", "close_sources"]

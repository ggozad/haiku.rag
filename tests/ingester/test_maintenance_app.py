import asyncio

import pytest

from haiku.rag.config import (
    APIConfig,
    AppConfig,
    EmbeddingModelConfig,
    EmbeddingsConfig,
    FSSourceConfig,
    IngestedDatabase,
    IngesterConfig,
    LanceDBConfig,
    MaintenanceConfig,
    QueueConfig,
    WorkerConfig,
)
from haiku.rag.ingester import maintenance
from haiku.rag.ingester.api.routes.stats import stats
from haiku.rag.ingester.app import FailedDatabase, IngesterApp
from tests.ingester.conftest import _wait_until

NAME = "haiku.rag"


def _config(tmp_path, *, enabled: bool = True, quiet_s: float = 0.1) -> AppConfig:
    root = tmp_path / "src"
    root.mkdir(exist_ok=True)
    return AppConfig(
        embeddings=EmbeddingsConfig(model=EmbeddingModelConfig(vector_dim=16)),
        lancedb=LanceDBConfig(databases={NAME: str(tmp_path / "db.lancedb")}),
        ingester=IngesterConfig(
            queue=QueueConfig(path=tmp_path / "queue.db"),
            databases={
                NAME: IngestedDatabase(
                    sources=[
                        FSSourceConfig(
                            type="fs", id="local", root=root, poll_interval_s=3600.0
                        )
                    ]
                )
            },
            workers=WorkerConfig(worker_count=1, poll_idle_interval_s=0.05),
            api=APIConfig(enabled=False),
            maintenance=MaintenanceConfig(
                enabled=enabled, quiet_s=quiet_s, max_wait_s=quiet_s * 10
            ),
        ),
    )


def _session(app: IngesterApp):
    return app._running[NAME].client._single_session("test")


@pytest.mark.parametrize("enabled", [True, False])
async def test_maintained_clients_do_not_auto_vacuum(tmp_path, enabled):
    app = IngesterApp(config=_config(tmp_path, enabled=enabled))

    async with app._resources():
        assert _session(app).config.storage.auto_vacuum is not enabled
        assert (NAME in app._maintenance) is enabled
    assert app._config.storage.auto_vacuum is True


async def test_serve_maintains_each_database_and_reports_it(tmp_path):
    app = IngesterApp(config=_config(tmp_path))
    task = asyncio.create_task(app.serve(api=False))
    try:
        await _wait_until(
            lambda: (
                NAME in app._maintenance
                and app._maintenance[NAME].status.finished_at is not None
            ),
            serving=task,
        )
        maintenance_task = app._running[NAME].maintenance_task
        assert maintenance_task is not None and not maintenance_task.done()
        reported = (await stats(app._api_state())).maintenance_by_database[NAME]
        assert reported.vacuumed is True
        assert reported.index_built is False
        assert reported.error is None
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)
    assert maintenance_task.cancelled()


def _write_during_drain(app: IngesterApp, monkeypatch) -> None:
    drain = app._drain_batch

    async def _drain(*args, **kwargs):
        _session(app).record_write()
        return await drain(*args, **kwargs)

    monkeypatch.setattr(app, "_drain_batch", _drain)


async def test_run_batch_maintains_a_written_database(tmp_path, monkeypatch):
    app = IngesterApp(config=_config(tmp_path, quiet_s=3600))
    _write_during_drain(app, monkeypatch)

    report = await app.run_batch()

    assert report.failed_maintenance == []
    status = app._maintenance[NAME].status
    assert status.finished_at is not None and status.error is None


async def test_run_batch_leaves_an_unwritten_database_alone(tmp_path):
    app = IngesterApp(config=_config(tmp_path, quiet_s=3600))

    report = await app.run_batch()

    assert report.failed_maintenance == []
    assert app._maintenance[NAME].status.finished_at is None


async def test_run_batch_reports_a_failed_maintenance(tmp_path, monkeypatch):
    app = IngesterApp(config=_config(tmp_path, quiet_s=3600))
    drain = app._drain_batch

    async def _drain(*args, **kwargs):
        _session(app).record_write()
        # A vacuum already running makes this one skip.
        await app._running[NAME].client.store._vacuum_lock.acquire()
        return await drain(*args, **kwargs)

    monkeypatch.setattr(app, "_drain_batch", _drain)

    report = await app.run_batch()

    assert report.failed_maintenance == [
        FailedDatabase(database=NAME, error=f"vacuum of {NAME!r} did not complete")
    ]


def _block_passes(monkeypatch) -> asyncio.Event:
    """Passes start and never finish. The event is set once one starts."""
    started = asyncio.Event()

    async def _maintain(client):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(maintenance, "maintain", _maintain)
    return started


async def test_shutdown_cancels_a_pass_in_progress(tmp_path, monkeypatch):
    started = _block_passes(monkeypatch)
    app = IngesterApp(config=_config(tmp_path))
    task = asyncio.create_task(app.serve(api=False))
    try:
        await _wait_until(started.is_set, serving=task)
        maintenance_task = app._running[NAME].maintenance_task
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)

    assert maintenance_task is not None and maintenance_task.cancelled()
    assert app._maintenance[NAME].status.error is None
    assert app._pool is not None and app._pool._paused == set()


async def test_removing_a_database_cancels_its_pass(tmp_path, monkeypatch):
    started = _block_passes(monkeypatch)
    app = IngesterApp(config=_config(tmp_path))
    task = asyncio.create_task(app.serve(api=False))
    try:
        await _wait_until(started.is_set, serving=task)
        maintenance_task = app._running[NAME].maintenance_task

        await asyncio.wait_for(app.remove_database(NAME), timeout=5.0)

        assert maintenance_task is not None and maintenance_task.cancelled()
        assert NAME not in app._maintenance
        assert app._pool is not None and app._pool._paused == set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

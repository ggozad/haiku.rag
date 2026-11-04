import asyncio
import logging
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from haiku.rag.client import DocumentImport, HaikuRAG
from haiku.rag.config import MaintenanceConfig, get_config
from haiku.rag.ingester import maintenance
from haiku.rag.ingester.maintenance import (
    DatabaseMaintenance,
    MaintenanceError,
    MaintenanceResult,
    has_vector_index,
    maintain,
)
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.store.models.chunk import Chunk
from tests.conftest import capture_logs
from tests.documents import docling_doc

_MAINTENANCE_LOGGER = logging.getLogger("haiku.rag.ingester.maintenance")

DIM = 16


@pytest.fixture
async def client(temp_db_path):
    config = get_config().model_copy(deep=True)
    config.embeddings.model.vector_dim = DIM
    config.storage.auto_vacuum = False
    async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
        yield rag


async def add_chunks(client: HaikuRAG, count: int) -> None:
    rng = random.Random(count)
    chunks = [
        Chunk(
            content=f"chunk {i}",
            embedding=[rng.uniform(-1, 1) for _ in range(DIM)],
            order=i,
        )
        for i in range(count)
    ]
    await client.import_documents(
        [
            DocumentImport(
                docling_document=docling_doc("d", "body"),
                chunks=chunks,
                uri=f"mem://{count}",
            )
        ]
    )


async def test_a_small_table_is_vacuumed_and_left_unindexed(client):
    await add_chunks(client, 10)

    assert await maintain(client) == MaintenanceResult(vacuumed=True, index_built=False)


async def test_a_table_without_an_index_gets_one(client):
    await add_chunks(client, 300)
    assert await has_vector_index(client.store) is False

    assert await maintain(client) == MaintenanceResult(vacuumed=True, index_built=True)
    assert await has_vector_index(client.store) is True


async def test_an_indexed_table_is_vacuumed_and_not_rebuilt(client):
    await add_chunks(client, 300)
    await maintain(client)
    await add_chunks(client, 1000)

    assert await maintain(client) == MaintenanceResult(vacuumed=True, index_built=False)
    stats = await client.store.chunks_table.index_stats("vector_idx")
    assert stats is not None
    assert stats.num_unindexed_rows == 0


async def test_a_vacuum_that_does_not_complete_fails_the_pass(client):
    await add_chunks(client, 300)
    async with client.store._vacuum_lock:
        with pytest.raises(MaintenanceError, match="vacuum"):
            await asyncio.wait_for(maintain(client), timeout=5)

    assert await has_vector_index(client.store) is False


async def test_a_cloud_database_is_left_alone():
    store = AsyncMock(location="db://cloud")
    cloud = SimpleNamespace(store=store, create_vector_index=AsyncMock())

    assert await maintain(cloud) == MaintenanceResult(vacuumed=False, index_built=False)  # ty: ignore[invalid-argument-type]
    store.vacuum.assert_not_awaited()
    store.chunks_table.list_indices.assert_not_awaited()
    cloud.create_vector_index.assert_not_awaited()


def _pool(client, jobs, sync, *source_ids: str) -> WorkerPool:
    return WorkerPool(
        clients={source_id: client for source_id in source_ids},
        job_repo=jobs,
        sync_repo=sync,
    )


class _Spy:
    """Wraps `maintain`, recording the paused sources each pass saw."""

    def __init__(self, monkeypatch, pool: WorkerPool, fail: int = 0):
        self.paused: list[set[str]] = []
        self._pool = pool
        self._fail = fail
        real = maintenance.maintain

        async def _maintain(client):
            self.paused.append(set(pool._paused))
            if len(self.paused) <= self._fail:
                raise MaintenanceError("vacuum did not complete")
            return await real(client)

        monkeypatch.setattr(maintenance, "maintain", _maintain)


def _maintenance(client, pool, quiet_s=60.0, max_wait_s=600.0) -> DatabaseMaintenance:
    return DatabaseMaintenance(
        "db",
        client,
        pool,
        ["a"],
        MaintenanceConfig(enabled=True, quiet_s=quiet_s, max_wait_s=max_wait_s),
    )


async def test_a_pass_holds_its_databases_claims(client, jobs, sync, monkeypatch):
    pool = _pool(client, jobs, sync, "a", "b")
    spy = _Spy(monkeypatch, pool)
    session = client._single_session("test")
    session.record_write()

    result = await _maintenance(client, pool).run_once()

    assert spy.paused == [{"a"}]
    assert pool._paused == set()
    assert session.unmaintained_since is None
    assert result == MaintenanceResult(vacuumed=True, index_built=False)


async def test_a_failed_pass_resumes_claims_and_stays_due(
    client, jobs, sync, monkeypatch
):
    pool = _pool(client, jobs, sync, "a")
    _Spy(monkeypatch, pool, fail=1)
    session = client._single_session("test")
    session.record_write()
    since = session.unmaintained_since
    database = _maintenance(client, pool)

    with pytest.raises(MaintenanceError):
        await database.run_once()

    assert pool._paused == set()
    assert session.unmaintained_since == since
    assert database.status.error == "vacuum did not complete"
    assert database.status.result is None


async def test_a_write_during_a_failed_pass_keeps_its_time(
    client, jobs, sync, monkeypatch
):
    pool = _pool(client, jobs, sync, "a")
    session = client._single_session("test")
    session.record_write()

    async def _maintain(client):
        session.record_write()
        raise MaintenanceError("vacuum did not complete")

    monkeypatch.setattr(maintenance, "maintain", _maintain)
    written_during = None
    with pytest.raises(MaintenanceError):
        try:
            await _maintenance(client, pool).run_once()
        finally:
            written_during = session.last_write_at

    assert session.unmaintained_since == written_during


async def test_a_pass_runs_once_the_database_is_quiet(client, jobs, sync, monkeypatch):
    pool = _pool(client, jobs, sync, "a")
    spy = _Spy(monkeypatch, pool)
    task = asyncio.create_task(_maintenance(client, pool, quiet_s=0.2).run())
    try:
        # A start makes the database due.
        await _until(lambda: len(spy.paused) == 1)
        client._single_session("test").record_write()
        await asyncio.sleep(0.1)
        assert len(spy.paused) == 1
        await _until(lambda: len(spy.paused) == 2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_a_pass_runs_at_max_wait_under_constant_writes(
    client, jobs, sync, monkeypatch
):
    pool = _pool(client, jobs, sync, "a")
    spy = _Spy(monkeypatch, pool)
    session = client._single_session("test")
    task = asyncio.create_task(
        _maintenance(client, pool, quiet_s=0.3, max_wait_s=0.6).run()
    )
    try:
        for _ in range(20):
            session.record_write()
            await asyncio.sleep(0.05)
        assert len(spy.paused) >= 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_a_failed_pass_is_retried_after_quiet(client, jobs, sync, monkeypatch):
    pool = _pool(client, jobs, sync, "a")
    spy = _Spy(monkeypatch, pool, fail=1)
    database = _maintenance(client, pool, quiet_s=0.2)
    with capture_logs(_MAINTENANCE_LOGGER, logging.ERROR) as records:
        task = asyncio.create_task(database.run())
        try:
            await _until(lambda: len(spy.paused) == 1)
            await asyncio.sleep(0.1)
            assert len(spy.paused) == 1
            await _until(lambda: database.status.result is not None)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert len(spy.paused) == 2
    assert database.status.error is None
    assert [r.getMessage() for r in records] == [
        "Maintenance of database 'db' failed: vacuum did not complete"
    ]


async def _until(predicate, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_idle_polling_sees_a_write_within_max_wait(
    client, jobs, sync, monkeypatch
):
    pool = _pool(client, jobs, sync, "a")
    spy = _Spy(monkeypatch, pool)
    database = _maintenance(client, pool, quiet_s=60.0, max_wait_s=0.2)
    task = asyncio.create_task(database.run())
    try:
        await _until(lambda: database.status.result is not None)
        # The task is now idle, waiting for a write.
        await asyncio.sleep(0.05)
        client._single_session("test").record_write()
        await _until(lambda: len(spy.paused) == 2, timeout=2.0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

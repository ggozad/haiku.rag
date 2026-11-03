"""How the service starts and stops: on an inherited socket, and on its own
once idle."""

import asyncio
import os
import socket

import httpx
import pytest

from haiku.rag.config import APIConfig
from haiku.rag.ingester.app import LISTEN_FD_VARIABLE, IngesterApp
from haiku.rag.ingester.queue.repository import JobRepo
from tests.ingester.conftest import _mock_client, _wait_until
from tests.ingester.test_run_batch import _config


async def test_the_api_serves_an_inherited_socket(tmp_path, use_client, monkeypatch):
    use_client(_mock_client())
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    monkeypatch.setenv(LISTEN_FD_VARIABLE, str(os.dup(listener.fileno())))
    listener.close()
    config = _config(tmp_path)
    # Bound elsewhere than the socket: the inherited one wins.
    config.ingester.api = APIConfig(enabled=True, host="127.0.0.1", port=1)
    config.ingester.workers.shutdown_grace_s = 0.2
    app = IngesterApp(config=config)

    task = asyncio.create_task(app.serve())
    try:
        async with httpx.AsyncClient() as client:
            for _ in range(100):
                try:
                    response = await client.get(f"http://127.0.0.1:{port}/health")
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.05)
        assert response.status_code == 200
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.parametrize("api", [False, True])
async def test_an_idle_service_exits(tmp_path, use_client, monkeypatch, api):
    use_client(_mock_client())
    config = _config(tmp_path, shutdown_grace_s=0.2)
    config.ingester.api = APIConfig(enabled=api, host="127.0.0.1", port=0)
    config.ingester.service.idle_exit_s = 0.2
    counts = iter([{"queued": 1}])

    async def counts_by_status(self):  # noqa: ARG001
        return next(counts, {})

    monkeypatch.setattr(JobRepo, "counts_by_status", counts_by_status)
    app = IngesterApp(config=config)

    await asyncio.wait_for(app.serve(api=api), timeout=10)


async def test_a_busy_service_stays_up(tmp_path, use_client, monkeypatch):
    use_client(_mock_client())
    config = _config(tmp_path, shutdown_grace_s=0.2)
    config.ingester.service.idle_exit_s = 0.1

    async def counts_by_status(self):  # noqa: ARG001
        return {"claimed": 1}

    monkeypatch.setattr(JobRepo, "counts_by_status", counts_by_status)
    app = IngesterApp(config=config)
    task = asyncio.create_task(app.serve(api=False))
    try:
        await _wait_until(
            lambda: app._pool is not None and app._pool.live_workers > 0,
            serving=task,
        )
        await asyncio.sleep(0.3)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)


def _vacuuming_config(tmp_path, **settings):
    config = _config(tmp_path, shutdown_grace_s=0.2)
    vacuum = config.ingester.service.vacuum_after_indexing
    vacuum.min_documents = 3
    vacuum.quiet_s = 0.04
    for key, value in settings.items():
        setattr(vacuum, key, value)
    return config


def _indexed(monkeypatch, succeeded: int, busy=({"claimed": 1},)):
    """Jobs as a run that finished `succeeded` documents left them."""
    counts = iter(busy)

    async def counts_by_status(self):  # noqa: ARG001
        return next(counts, {})

    first: list = []

    async def counts_by_status_since(self, since, *, source_ids=None):  # noqa: ARG001
        if not first:
            first.append(since)
        return {"succeeded": succeeded} if since == first[0] else {}

    monkeypatch.setattr(JobRepo, "counts_by_status", counts_by_status)
    monkeypatch.setattr(JobRepo, "counts_by_status_since", counts_by_status_since)


async def _serving(app, until, *, api=False):
    task = asyncio.create_task(app.serve(api=api))
    try:
        await _wait_until(until, serving=task)
        await asyncio.sleep(0.3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)


async def test_a_drained_indexing_run_is_vacuumed_once(
    tmp_path, use_client, monkeypatch
):
    client = _mock_client()
    client.store.vacuuming = False
    use_client(client)
    _indexed(monkeypatch, succeeded=3)
    app = IngesterApp(config=_vacuuming_config(tmp_path, retention_seconds=0))

    await _serving(app, lambda: client.vacuum.await_count == 1)

    client.vacuum.assert_awaited_once_with(retention_seconds=0)


async def test_a_small_indexing_run_is_not_vacuumed(tmp_path, use_client, monkeypatch):
    client = _mock_client()
    client.store.vacuuming = False
    use_client(client)
    _indexed(monkeypatch, succeeded=2)
    app = IngesterApp(config=_vacuuming_config(tmp_path))
    seen = []
    real = JobRepo.counts_by_status_since

    async def counting(self, since, *, source_ids=None):
        seen.append(since)
        return await real(self, since, source_ids=source_ids)

    monkeypatch.setattr(JobRepo, "counts_by_status_since", counting)

    await _serving(app, lambda: len(seen) >= 2)

    client.vacuum.assert_not_awaited()


async def test_no_vacuum_runs_beside_another(tmp_path, use_client, monkeypatch):
    client = _mock_client()
    client.store.vacuuming = True
    use_client(client)
    _indexed(monkeypatch, succeeded=5)
    app = IngesterApp(config=_vacuuming_config(tmp_path))
    ticks = []
    real = JobRepo.counts_by_status

    async def counting(self):
        ticks.append(1)
        return await real(self)

    monkeypatch.setattr(JobRepo, "counts_by_status", counting)

    await _serving(app, lambda: len(ticks) >= 4)

    client.vacuum.assert_not_awaited()


async def test_a_failed_vacuum_is_not_retried(tmp_path, use_client, monkeypatch):
    client = _mock_client()
    client.store.vacuuming = False
    client.vacuum.side_effect = RuntimeError("lance")
    use_client(client)
    _indexed(monkeypatch, succeeded=3)
    app = IngesterApp(config=_vacuuming_config(tmp_path))

    await _serving(app, lambda: client.vacuum.await_count == 1)

    assert client.vacuum.await_count == 1


async def test_the_vacuum_waits_for_running_searches(tmp_path, use_client, monkeypatch):
    from haiku.rag.service.search import SearchService

    client = _mock_client()
    client.store.vacuuming = False
    use_client(client)
    _indexed(monkeypatch, succeeded=3)
    config = _vacuuming_config(tmp_path)
    config.ingester.api = APIConfig(enabled=True, host="127.0.0.1", port=0)
    busy = iter([True, True])
    real = SearchService.while_idle

    async def while_idle(self, work):
        if next(busy, False):
            return False
        return await real(self, work)

    monkeypatch.setattr(SearchService, "while_idle", while_idle)
    app = IngesterApp(config=config)

    await _serving(app, lambda: client.vacuum.await_count == 1, api=True)

    assert client.vacuum.await_count == 1


def test_vacuum_after_indexing_can_be_off(tmp_path):
    config = _vacuuming_config(tmp_path, min_documents=None)

    assert IngesterApp(config=config)._vacuum_after_indexing() is None

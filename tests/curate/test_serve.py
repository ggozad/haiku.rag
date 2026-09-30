import asyncio
import logging
import socket
import threading

import httpx
import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import APIConfig
from haiku.rag.curate import app as app_module
from haiku.rag.curate.app import APIServerStopped, describe, serve
from haiku.rag.curate.store.models import DatabaseSweep, SweepStatus
from tests.conftest import capture_logs
from tests.curate.test_sweep import (
    _import,
    _writer_config,
    curate,  # noqa: F401
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _until(condition, timeout: float = 30.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not await condition():
        assert asyncio.get_running_loop().time() < deadline, "condition never held"
        await asyncio.sleep(0.05)


async def test_serve_sweeps_on_an_interval_until_stopped(curate, request):  # noqa: F811
    paths, config, repository = curate
    app_module.logger.setLevel(logging.INFO)
    request.addfinalizer(lambda: app_module.logger.setLevel(logging.NOTSET))
    async with HaikuRAG(paths["wiki"], _writer_config()) as rag:
        await _import(rag, "file:///wiki/a.pdf", ["Alpha."], [0])
    config.curate.api = APIConfig(enabled=False)
    config.curate.sweep_interval_s = 0.05
    stop = asyncio.Event()

    with capture_logs(app_module.logger, logging.INFO) as records:
        task = asyncio.create_task(serve(config, stop=stop))

        async def swept_again() -> bool:
            messages = [r.getMessage() for r in records]
            return messages.count("wiki: unchanged") >= 1

        await _until(swept_again)
        stop.set()
        await task

    messages = [r.getMessage() for r in records]
    assert "wiki: ok, 1 documents, 1 new, 0 deleted" in messages
    assert await repository.last_ok_sweep("wiki") is not None


async def test_serve_answers_the_api_without_sweeping(curate):  # noqa: F811
    paths, config, repository = curate
    port = _free_port()
    config.curate.api = APIConfig(port=port, auth_token="secret")
    stop = asyncio.Event()
    task = asyncio.create_task(serve(config, sweeping=False, stop=stop))

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:

        async def healthy() -> bool:
            try:
                return (await client.get("/health")).status_code == 200
            except httpx.TransportError:
                return False

        await _until(healthy)
        health = (await client.get("/health")).json()
        unauthorised = await client.get("/databases")
    stop.set()
    await task

    assert [d["last_status"] for d in health["databases"]] == [None, None]
    assert unauthorised.status_code == 401
    assert await repository.last_ok_sweep("wiki") is None


async def test_serve_warns_when_the_api_has_no_token(curate):  # noqa: F811
    _, config, _ = curate
    port = _free_port()
    config.curate.api = APIConfig(port=port)
    stop = asyncio.Event()

    with capture_logs(app_module.logger, logging.WARNING) as records:
        task = asyncio.create_task(serve(config, sweeping=False, stop=stop))
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:

            async def healthy() -> bool:
                try:
                    return (await client.get("/health")).status_code == 200
                except httpx.TransportError:
                    return False

            await _until(healthy)
        stop.set()
        await task

    assert [r.getMessage() for r in records] == [
        "curate.api.auth_token is unset: the API is unauthenticated"
    ]


def _result(status: SweepStatus, **values) -> DatabaseSweep:
    return DatabaseSweep(
        database="wiki", started_at="", finished_at="", status=status, **values
    )


def test_describe_each_status():
    ok = _result(SweepStatus.OK, documents=3, rebaseline=True)
    assert describe(ok) == "wiki: ok, 3 documents, 0 new, 0 deleted, embedder changed"
    assert describe(_result(SweepStatus.ERROR, error="boom")) == "wiki: error: boom"
    assert describe(_result(SweepStatus.UNCHANGED)) == "wiki: unchanged"


async def test_the_api_answers_while_a_sweep_runs(curate, monkeypatch):  # noqa: F811
    _, config, _ = curate
    port = _free_port()
    config.curate.api = APIConfig(port=port, auth_token="secret")
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    async def blocking_sweep(config, repository):
        started.set()
        release.wait(timeout=30)
        finished.set()
        return []

    monkeypatch.setattr(app_module, "sweep", blocking_sweep)
    stop = asyncio.Event()
    task = asyncio.create_task(serve(config, stop=stop))
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", timeout=5
        ) as client:

            async def healthy_mid_sweep() -> bool:
                if not started.is_set():
                    return False
                try:
                    answered = (await client.get("/health")).status_code == 200
                except httpx.TransportError:
                    return False
                return answered and not finished.is_set()

            await _until(healthy_mid_sweep, timeout=10)
    finally:
        release.set()
        stop.set()
        await task


async def test_serve_fails_when_the_api_server_stops(curate, monkeypatch):  # noqa: F811
    _, config, _ = curate
    config.curate.sweep_interval_s = 3600

    class StoppingServer:
        should_exit = False

        async def serve(self):
            return None

    monkeypatch.setattr(
        app_module, "_api_server", lambda config, repository: StoppingServer()
    )

    with pytest.raises(APIServerStopped, match="the API server stopped"):
        await asyncio.wait_for(serve(config, sweeping=False), timeout=10)


async def test_serve_logs_a_failed_database_as_a_warning(curate, tmp_path):  # noqa: F811
    _, config, _ = curate
    config.lancedb.databases["gone"] = str(tmp_path / "gone.lancedb")
    config.curate.api = APIConfig(enabled=False)
    stop = asyncio.Event()

    with capture_logs(app_module.logger, logging.WARNING) as records:
        task = asyncio.create_task(serve(config, stop=stop))

        async def reported() -> bool:
            return any(r.getMessage().startswith("gone: error:") for r in records)

        await _until(reported)
        stop.set()
        await task

    [record] = records
    assert record.levelno == logging.WARNING

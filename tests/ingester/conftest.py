import asyncio
import contextlib
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import httpx
import pytest
from httpx import ASGITransport

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, QueueConfig
from haiku.rag.ingester.app import IngesterApp
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.queue.migrations import open_queue
from haiku.rag.ingester.queue.models import Job, JobOp, JobStatus
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.store.models.document import Document
from haiku.rag.uri import uri_to_path


@pytest.fixture
async def engine(tmp_path):
    eng = await open_queue(QueueConfig(path=tmp_path / "queue.db"))
    yield eng
    await eng.dispose()


@pytest.fixture
async def conn(tmp_path, engine):
    """Raw connection to the queue file for test-only SQL inspection and
    backdating. Shares the WAL database the engine writes through."""
    connection = await aiosqlite.connect(str(tmp_path / "queue.db"))
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA busy_timeout=30000")
    yield connection
    await connection.close()


@pytest.fixture
def jobs(engine):
    return JobRepo(engine)


@pytest.fixture
def sync(engine):
    return SyncStateRepo(engine)


@pytest.fixture
def use_client(monkeypatch):
    """Make IngesterApp's internally-created HaikuRAG resolve to the given
    mock client. Call again to swap the client for a later run in the same
    test."""

    def _install(client):
        """`client` serves every database, or a dict maps a database name to
        its client."""

        @asynccontextmanager
        async def _cm(scope):
            yield client[scope.names[0]] if isinstance(client, dict) else client

        # The app opens each database through `_covering`: the double stands
        # in for it.
        monkeypatch.setattr(
            "haiku.rag.client.HaikuRAG",
            covering_stub(lambda scope, *a, **k: _cm(scope)),
        )

    return _install


async def finished_job(
    repo: JobRepo,
    status: JobStatus,
    uri: str = "u",
    *,
    source_id: str = "s",
    op: JobOp = JobOp.UPSERT,
    error: str = "boom",
    conversion_stalled: bool = False,
) -> Job:
    """Enqueue `uri`, claim it and finish it in `status`. Returns the claim."""
    job = await repo.enqueue(source_id, uri, op)
    assert job is not None
    claimed = await repo.claim_next("w", source_ids={source_id})
    assert claimed is not None and claimed.id == job.id
    if status is JobStatus.SUCCEEDED:
        assert await repo.mark_succeeded(claimed.id, "w")
    elif status is JobStatus.REJECTED:
        assert await repo.mark_rejected(claimed.id, error, "w")
    else:
        assert await repo.mark_dead(
            claimed.id, error, "w", conversion_stalled=conversion_stalled
        )
    return claimed


async def status_of(repo: JobRepo, job_id: str) -> JobStatus:
    job = await repo.get_job(job_id)
    assert job is not None
    return job.status


class FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory
        self.loaded = False

    def load(self):
        self.loaded = True
        return self._factory


def covering_stub(covering) -> MagicMock:
    """A HaikuRAG double whose `_covering` is `covering`."""
    stub = MagicMock()
    stub._covering = covering
    return stub


def _mock_client() -> AsyncMock:
    """A HaikuRAG double that keeps ingested documents for lookup and listing."""
    client = AsyncMock(spec=HaikuRAG)
    counter = {"n": 0}
    stored: dict[str, Document] = {}

    async def _create(uri, *_, source_id=None, **__):
        counter["n"] += 1
        # FS source_revision is the file's mtime_ns.
        path = uri_to_path(uri)
        metadata = {
            "content_type": "text/markdown",
            "md5": f"md5-{counter['n']}",
            "source_revision": str(path.stat().st_mtime_ns),
        }
        if source_id is not None:
            metadata["source_id"] = source_id
        document = Document(
            id=f"doc-{counter['n']}", content="x", uri=uri, metadata=metadata
        )
        stored[uri] = document
        return document

    async def _get_by_uri(uri):
        return stored.get(uri) or Document(
            id=f"doc-for-{uri}", content="x", uri=uri, metadata={}
        )

    async def _delete(document_id, *, source_id=None):
        for uri, document in list(stored.items()):
            if document.id == document_id:
                del stored[uri]
        return True

    async def _list_documents(*_, **__):
        return list(stored.values())

    client._ingest_observed.side_effect = _create
    client.get_document_by_uri.side_effect = _get_by_uri
    client.delete_document.side_effect = _delete
    client.list_documents.side_effect = _list_documents
    return client


async def _wait_until(
    predicate, *, timeout: float = 5.0, serving: asyncio.Task | None = None
):
    """Wait for `predicate`, raising the error of a `serving` task that ended."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if serving is not None and serving.done():
            serving.result()
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached within timeout")
        await asyncio.sleep(0.02)


def _two_databases(tmp_path) -> AppConfig:
    for name in ("papers", "wiki"):
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.md").write_text(name)
    return AppConfig.model_validate(
        {
            "lancedb": {
                "databases": {
                    name: str(tmp_path / f"{name}.lancedb")
                    for name in ("papers", "wiki")
                }
            },
            "ingester": {
                "queue": {"path": str(tmp_path / "queue.db")},
                "workers": {
                    "worker_count": 2,
                    "poll_idle_interval_s": 0.05,
                    "retry": {"max_attempts": 1},
                },
                "api": {"enabled": False},
                "databases": {
                    name: {
                        "sources": [
                            {
                                "type": "fs",
                                "id": name,
                                "root": str(tmp_path / name),
                                "poll_interval_s": 3600.0,
                            }
                        ]
                    }
                    for name in ("papers", "wiki")
                },
            },
        }
    )


def _ingested(client) -> list[str]:
    return [
        uri_to_path(call.args[0]).name
        for call in client._ingest_observed.await_args_list
    ]


async def _seed_unconfigured_jobs(config) -> JobRepo:
    engine = await open_queue(config.ingester.queue)
    jobs = JobRepo(engine)
    claimed = await jobs.enqueue("removed", "claimed", JobOp.UPSERT)
    assert claimed is not None
    assert await jobs.claim_next("w") is not None
    dead = await jobs.enqueue("removed", "dead", JobOp.UPSERT)
    assert dead is not None and await jobs.claim_next("w") is not None
    assert await jobs.mark_dead(dead.id, "boom", "w")
    assert await jobs.enqueue("removed", "queued", JobOp.UPSERT) is not None
    await engine.dispose()
    return jobs


async def _jobs_of(config, source_id: str) -> dict[str, JobStatus]:
    engine = await open_queue(config.ingester.queue)
    try:
        return {
            job.uri: job.status
            for job in await JobRepo(engine).list_jobs(source_id=source_id)
        }
    finally:
        await engine.dispose()


def _pool(app: IngesterApp) -> WorkerPool:
    assert app._pool is not None
    return app._pool


def _pollers(app: IngesterApp) -> PollerManager:
    assert app._pollers is not None
    return app._pollers


def _gate(client, gate: asyncio.Event) -> None:
    """Hold every ingestion through `client` until `gate` is set."""
    ingest = client._ingest_observed.side_effect

    async def _held(uri, *args, **kwargs):
        await gate.wait()
        return await ingest(uri, *args, **kwargs)

    client._ingest_observed.side_effect = _held


@asynccontextmanager
async def _serving(app: IngesterApp):
    # Shutdown cancels a held job 0.2s after the pool stops.
    app._config.ingester.workers.shutdown_grace_s = 0.2
    task = asyncio.create_task(app.serve(api=False))
    try:
        await _wait_until(
            lambda: app._pool is not None and app._pool.live_workers > 0,
            serving=task,
        )
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5.0)


class _Opened:
    """Opens `client`, counting its closes."""

    def __init__(self, client):
        self.client = client
        self.closes = 0

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *exc_info):
        self.closes += 1
        return False


class _FailsToCloseOnce(_Opened):
    async def __aexit__(self, *exc_info):
        await super().__aexit__(*exc_info)
        if self.closes == 1:
            raise RuntimeError("close failed")
        return False


def _api(app: IngesterApp, auth_token: str | None = None) -> httpx.AsyncClient:
    from haiku.rag.ingester.api.server import build_app

    return httpx.AsyncClient(
        transport=ASGITransport(app=build_app(app._api_state(), auth_token=auth_token)),
        base_url="http://testserver",
    )

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from httpx import ASGITransport

from haiku.rag.client.scope import DatabaseScope
from haiku.rag.config import AppConfig, FSSourceConfig, SourceConfig
from haiku.rag.ingester.app import IngesterApp
from haiku.rag.ingester.exceptions import PermanentError
from haiku.rag.ingester.pollers.manager import PollerManager
from haiku.rag.ingester.queue.migrations import open_queue
from haiku.rag.ingester.queue.models import Job, JobOp, JobStatus
from haiku.rag.ingester.queue.repository import JobRepo, SyncStateRepo
from haiku.rag.ingester.reconcile import reconcile
from haiku.rag.ingester.workers.pool import WorkerPool
from haiku.rag.sources.base import SourceEvent, SourceEventKind
from haiku.rag.store.exceptions import MigrationRequiredError, UpgradeRequiredError
from tests.conftest import capture_logs
from tests.ingester.test_run_batch import (
    _ingested,
    _jobs_of,
    _mock_client,
    _seed_unconfigured_jobs,
    _two_databases,
    _wait_until,
)


def _papers_only(tmp_path) -> AppConfig:
    """Both databases placed, only papers ingested."""
    config = _two_databases(tmp_path)
    del config.ingester.databases["wiki"]
    return config


def _wiki(config: AppConfig, tmp_path):
    """The wiki database placed by `config`, and its source, as
    `add_database` takes them."""
    ref = DatabaseScope.resolve(config, database_name="wiki").databases[0]
    sources: list[SourceConfig] = [
        FSSourceConfig(
            type="fs", id="wiki", root=tmp_path / "wiki", poll_interval_s=3600.0
        )
    ]
    return ref, sources


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
    # A held job left in flight is cancelled at shutdown instead of waited for.
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


@asynccontextmanager
async def _queue(config: AppConfig):
    engine = await open_queue(config.ingester.queue)
    try:
        yield JobRepo(engine), SyncStateRepo(engine)
    finally:
        await engine.dispose()


async def _live(config: AppConfig, source_id: str) -> list[Job]:
    async with _queue(config) as (jobs, _):
        return await jobs.live_jobs({source_id})


def _api(app: IngesterApp) -> httpx.AsyncClient:
    from haiku.rag.ingester.api.server import build_app

    return httpx.AsyncClient(
        transport=ASGITransport(app=build_app(app._api_state())),
        base_url="http://testserver",
    )


async def test_a_database_added_while_serving_is_ingested_beside_running_jobs(
    tmp_path, use_client
):
    papers, wiki = _mock_client(), _mock_client()
    gate = asyncio.Event()
    _gate(papers, gate)
    use_client({"papers": papers, "wiki": wiki})
    config = _papers_only(tmp_path)
    app = IngesterApp(config=config)

    async with _serving(app):
        await _wait_until(lambda: bool(_pool(app)._inflight))
        await app.add_database(*_wiki(config, tmp_path))

        await _wait_until(lambda: _ingested(wiki) == ["wiki.md"])
        assert await papers.list_documents() == []
        gate.set()
        await _wait_until(lambda: not _pool(app)._inflight)
        assert len(await papers.list_documents()) == 1

    assert sorted(app._source_databases) == ["papers", "wiki"]


async def test_removing_a_database_waits_for_its_jobs_and_deletes_its_queued_ones(
    tmp_path, use_client
):
    config = _papers_only(tmp_path)
    for name in ("b", "c"):
        (tmp_path / "papers" / f"{name}.md").write_text(name)
    papers = _mock_client()
    gate = asyncio.Event()
    _gate(papers, gate)
    use_client({"papers": papers})
    app = IngesterApp(config=config)

    async with _serving(app):
        await _wait_until(lambda: len(_pool(app)._inflight) == 2)
        removal = asyncio.create_task(app.remove_database("papers"))
        await _wait_until(lambda: "papers" not in _pool(app)._served)
        await asyncio.sleep(0.1)
        assert not removal.done()

        gate.set()
        await asyncio.wait_for(removal, timeout=5.0)

        assert len(papers._ingest_observed.await_args_list) == 2
        assert "papers" not in app._running
        assert await _live(config, "papers") == []
        async with _queue(config) as (_, sync):
            assert await sync.list_known_uris("papers") == set()


async def test_after_a_removal_the_other_sources_keep_every_worker(
    tmp_path, use_client
):
    config = _two_databases(tmp_path)
    (tmp_path / "papers" / "papers.md").unlink()
    papers, wiki = _mock_client(), _mock_client()
    gate = asyncio.Event()
    _gate(papers, gate)
    use_client({"papers": papers, "wiki": wiki})
    app = IngesterApp(config=config)

    async with _serving(app):
        await _wait_until(lambda: _ingested(wiki) == ["wiki.md"])
        await app.remove_database("wiki")

        for name in ("a", "b"):
            (tmp_path / "papers" / f"{name}.md").write_text(name)
        assert await _pollers(app).pollers[0]._sweep_once()

        await _wait_until(lambda: len(_pool(app)._inflight) == 2)
        gate.set()


async def test_a_retry_of_a_job_of_a_database_under_removal_answers_409(
    tmp_path, use_client
):
    papers = _mock_client()
    gate = asyncio.Event()
    _gate(papers, gate)
    use_client({"papers": papers})
    config = _papers_only(tmp_path)
    async with _queue(config) as (jobs, _):
        dead = await jobs.enqueue("papers", "file:///old.md", JobOp.UPSERT)
        assert dead is not None and await jobs.claim_next("w") is not None
        assert await jobs.mark_dead(dead.id, "boom", "w")
    app = IngesterApp(config=config)

    async with _serving(app), _api(app) as api:
        await _wait_until(lambda: bool(_pool(app)._inflight))
        removal = asyncio.create_task(app.remove_database("papers"))
        await _wait_until(lambda: "papers" not in _pool(app)._served)

        resp = await api.post(f"/jobs/{dead.id}/retry")

        gate.set()
        await asyncio.wait_for(removal, timeout=5.0)
    assert resp.status_code == 409


async def test_a_retry_landing_before_a_removal_is_deleted_by_it(
    tmp_path, use_client, monkeypatch
):
    use_client({"papers": _mock_client()})
    config = _papers_only(tmp_path)
    async with _queue(config) as (jobs, _):
        dead = await jobs.enqueue("papers", "file:///old.md", JobOp.UPSERT)
        assert dead is not None and await jobs.claim_next("w") is not None
        assert await jobs.mark_dead(dead.id, "boom", "w")
    app = IngesterApp(config=config)

    async with _serving(app), _api(app) as api:
        assert app._jobs is not None
        retrying, release = asyncio.Event(), asyncio.Event()
        retry = app._jobs.retry

        async def _paused(job_id):
            retrying.set()
            await release.wait()
            return await retry(job_id)

        monkeypatch.setattr(app._jobs, "retry", _paused)
        request = asyncio.create_task(api.post(f"/jobs/{dead.id}/retry"))
        await asyncio.wait_for(retrying.wait(), timeout=5.0)
        removal = asyncio.create_task(app.remove_database("papers"))
        await asyncio.sleep(0.05)
        assert not removal.done()

        release.set()
        resp = await asyncio.wait_for(request, timeout=5.0)
        await asyncio.wait_for(removal, timeout=5.0)

    assert resp.status_code == 200
    assert await _live(config, "papers") == []


async def test_a_source_retry_racing_a_removal_completes_first_or_is_refused(
    tmp_path, use_client, monkeypatch
):
    use_client({"papers": _mock_client()})
    config = _papers_only(tmp_path)
    async with _queue(config) as (jobs, _):
        dead = await jobs.enqueue("papers", "file:///old.md", JobOp.UPSERT)
        assert dead is not None and await jobs.claim_next("w") is not None
        assert await jobs.mark_dead(dead.id, "boom", "w")
    app = IngesterApp(config=config)

    async with _serving(app), _api(app) as api:
        assert app._jobs is not None
        retrying, release = asyncio.Event(), asyncio.Event()
        retry_failed = app._jobs.retry_failed

        async def _paused(statuses, *, source_ids=None):
            retrying.set()
            await release.wait()
            return await retry_failed(statuses, source_ids=source_ids)

        monkeypatch.setattr(app._jobs, "retry_failed", _paused)
        retry = partial(api.post, "/sources/papers/retry", params={"status": "dead"})
        first = asyncio.create_task(retry())
        await asyncio.wait_for(retrying.wait(), timeout=5.0)
        waiting = asyncio.create_task(retry())
        await asyncio.sleep(0.05)
        removal = asyncio.create_task(app.remove_database("papers"))
        await asyncio.sleep(0.05)
        assert not removal.done() and not waiting.done()
        late = await asyncio.wait_for(retry(), timeout=5.0)

        release.set()
        first_resp, waiting_resp = await asyncio.wait_for(
            asyncio.gather(first, waiting), timeout=5.0
        )
        await asyncio.wait_for(removal, timeout=5.0)

    assert first_resp.json()["retried"] == 1
    assert waiting_resp.status_code == 409
    assert late.status_code == 409
    assert await _live(config, "papers") == []


async def test_a_refresh_racing_a_removal_completes_first_or_is_refused(
    tmp_path, use_client, monkeypatch
):
    papers = _mock_client()
    use_client({"papers": papers})
    config = _papers_only(tmp_path)
    app = IngesterApp(config=config)

    async with _serving(app), _api(app) as api:
        # A refresh skips a source with pending work, so papers.md finishes first.
        await _wait_until(lambda: _ingested(papers) == ["papers.md"])
        await _wait_until(lambda: not _pool(app)._inflight)
        poller = _pollers(app).pollers[0]
        listing, gate = asyncio.Event(), asyncio.Event()
        calls = []

        async def _discover(since=None, *, known_uris=None):
            calls.append(True)
            listing.set()
            await gate.wait()
            yield SourceEvent(
                source_id="papers",
                uri=(tmp_path / "papers" / "new.md").as_uri(),
                kind=SourceEventKind.UPSERT,
                revision="r1",
                discovered_at=datetime.now(UTC),
            )

        monkeypatch.setattr(poller.source, "discover", _discover)
        first = asyncio.create_task(api.post("/sources/papers/refresh"))
        await asyncio.wait_for(listing.wait(), timeout=5.0)
        removal = asyncio.create_task(app.remove_database("papers"))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(api.post("/sources/papers/refresh"))
        await asyncio.sleep(0.05)

        gate.set()
        first_resp, second_resp = await asyncio.wait_for(
            asyncio.gather(first, second), timeout=5.0
        )
        await asyncio.wait_for(removal, timeout=5.0)

    assert first_resp.json()["refreshed"] is True
    assert second_resp.json()["refreshed"] is False
    assert len(calls) == 1
    assert await _live(config, "papers") == []


async def test_a_readded_database_rediscovers_its_source(tmp_path, use_client):
    config = _papers_only(tmp_path)
    stalled = tmp_path / "papers" / "stalled.md"
    stalled.write_text("stalls")
    papers = _mock_client()
    ingest = papers._ingest_observed.side_effect

    async def _stalls(uri, *args, **kwargs):
        if uri == stalled.as_uri():
            raise PermanentError("conversion stalled", conversion_stalled=True)
        return await ingest(uri, *args, **kwargs)

    papers._ingest_observed.side_effect = _stalls
    use_client({"papers": papers})
    ref = DatabaseScope.resolve(config, database_name="papers").databases[0]
    sources = list(config.ingester.databases["papers"].sources)
    app = IngesterApp(config=config)

    async with _serving(app):
        await _wait_until(lambda: len(papers._ingest_observed.await_args_list) == 2)
        await _wait_until(lambda: not _pool(app)._inflight)
        await app.remove_database("papers")

        await app.add_database(ref, sources)
        await _wait_until(lambda: len(papers._ingest_observed.await_args_list) == 3)
        await _wait_until(lambda: not _pool(app)._inflight)

        documents = await papers.list_documents()

    attempts = _ingested(papers)
    assert sorted(attempts[:2]) == ["papers.md", "stalled.md"]
    assert attempts[2:] == ["papers.md"]
    assert [document.uri for document in documents] == [
        (tmp_path / "papers" / "papers.md").as_uri()
    ]
    papers.delete_document.assert_not_awaited()
    assert await _jobs_of(config, "papers") == {
        (tmp_path / "papers" / "papers.md").as_uri(): JobStatus.SUCCEEDED,
        stalled.as_uri(): JobStatus.DEAD,
    }


@pytest.mark.parametrize("error", [MigrationRequiredError, UpgradeRequiredError])
async def test_a_database_failing_to_open_on_add_leaves_the_app_as_it_was(
    tmp_path, monkeypatch, error
):
    papers = _mock_client()

    @asynccontextmanager
    async def _open(scope):
        if scope.names == ("wiki",):
            raise error("Database cannot be opened by this release")
        yield papers

    stub = MagicMock()
    stub._covering = lambda scope, *a, **k: _open(scope)
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)
    config = _papers_only(tmp_path)
    app = IngesterApp(config=config)

    async with _serving(app):
        with pytest.raises(error, match="database 'wiki'"):
            await app.add_database(*_wiki(config, tmp_path))

        assert list(app._running) == ["papers"]
        assert [poller.source_id for poller in _pollers(app).pollers] == ["papers"]
        assert _pool(app)._served == {"papers"}
        assert app._source_databases == {"papers": "papers"}


async def test_a_database_failing_to_reconcile_on_add_is_closed(
    tmp_path, use_client, monkeypatch
):
    papers, wiki = _mock_client(), _mock_client()
    use_client({"papers": papers, "wiki": wiki})
    config = _papers_only(tmp_path)
    app = IngesterApp(config=config)
    ref, sources = _wiki(config, tmp_path)
    closed = []

    async def _fails_for_wiki(rag, sync, source_ids):
        if rag is wiki:
            raise RuntimeError("store unreachable")
        return await reconcile(rag, sync, source_ids)

    async with _serving(app):
        monkeypatch.setattr("haiku.rag.ingester.app.reconcile", _fails_for_wiki)
        monkeypatch.setattr(
            "haiku.rag.sources.fs.FSSource.aclose",
            lambda self: _record(closed, self.source_id),
        )
        with pytest.raises(RuntimeError, match="store unreachable"):
            await app.add_database(ref, sources)

        assert closed == ["wiki"]
        assert list(app._running) == ["papers"]
        assert _pool(app)._served == {"papers"}


async def _record(closed: list, source_id: str) -> None:
    closed.append(source_id)


async def test_a_shutdown_during_a_removals_wait_completes(tmp_path, use_client):
    papers = _mock_client()
    _gate(papers, asyncio.Event())
    use_client({"papers": papers})
    config = _papers_only(tmp_path)
    config.ingester.workers.shutdown_grace_s = 0.2
    app = IngesterApp(config=config)

    task = asyncio.create_task(app.serve(api=False))
    await _wait_until(lambda: app._pool is not None and bool(app._pool._inflight))
    removal = asyncio.create_task(app.remove_database("papers"))
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5.0)
    removal.cancel()
    with contextlib.suppress(BaseException):
        await removal

    assert app._pool is not None and app._pool.live_workers == 0
    assert all(
        job.status is not JobStatus.CLAIMED for job in await _live(config, "papers")
    )


@pytest.mark.parametrize("step", ["delete", "close"])
async def test_a_failed_removal_stays_removing_until_the_next_completes(
    tmp_path, use_client, monkeypatch, step
):
    use_client({"papers": _mock_client()})
    config = _papers_only(tmp_path)
    app = IngesterApp(config=config)

    async with _serving(app):
        assert app._jobs is not None
        database = app._running["papers"]
        failures = [RuntimeError(f"{step} failed")]

        def _once(original):
            async def _maybe_fail(*args, **kwargs):
                if failures:
                    raise failures.pop()
                return await original(*args, **kwargs)

            return _maybe_fail

        if step == "delete":
            monkeypatch.setattr(
                app._jobs, "cancel_sources", _once(app._jobs.cancel_sources)
            )
        else:
            source = database.pollers[0].source
            monkeypatch.setattr(source, "aclose", _once(source.aclose))

        with pytest.raises(RuntimeError, match=f"{step} failed"):
            await app.remove_database("papers")
        assert database.removing
        assert "papers" in app._running

        await app.remove_database("papers")
        assert "papers" not in app._running
        assert "papers" not in app._ingested


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


async def test_a_failed_client_close_is_retried_by_the_next_removal(
    tmp_path, monkeypatch
):
    opened = _FailsToCloseOnce(_mock_client())
    stub = MagicMock()
    stub._covering = lambda scope, *a, **k: opened
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)
    app = IngesterApp(config=_papers_only(tmp_path))

    async with _serving(app):
        with pytest.raises(RuntimeError, match="close failed"):
            await app.remove_database("papers")
        assert app._running["papers"].removing

        await app.remove_database("papers")

        assert opened.closes == 2
        assert "papers" not in app._running


async def test_a_database_with_an_unknown_metadata_provider_is_never_opened_on_add(
    tmp_path, monkeypatch
):
    opened = []
    papers = _Opened(_mock_client())

    def _covering(scope, *a, **k):
        opened.append(scope.names[0])
        return papers

    stub = MagicMock()
    stub._covering = _covering
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)
    config = _papers_only(tmp_path)
    app = IngesterApp(config=config)
    ref, sources = _wiki(config, tmp_path)
    sources[0].metadata_provider = "missing"

    async with _serving(app):
        with pytest.raises(ValueError, match="missing"):
            await app.add_database(ref, sources)

        assert opened == ["papers"]
        assert list(app._running) == ["papers"]


async def test_a_failed_close_at_shutdown_still_closes_the_other_databases(
    tmp_path, monkeypatch
):
    opened = {
        "papers": _FailsToCloseOnce(_mock_client()),
        "wiki": _FailsToCloseOnce(_mock_client()),
    }
    stub = MagicMock()
    stub._covering = lambda scope, *a, **k: opened[scope.names[0]]
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)

    with pytest.raises(RuntimeError, match="close failed"):
        await IngesterApp(config=_two_databases(tmp_path)).run_batch()

    assert {name: o.closes for name, o in opened.items()} == {"papers": 1, "wiki": 1}


async def test_a_failed_close_after_a_failed_startup_still_closes_the_others(
    tmp_path, monkeypatch
):
    from haiku.rag.store.exceptions import AmbiguousDatabaseError

    opened = {
        "papers": _FailsToCloseOnce(_mock_client()),
        "wiki": _Opened(_mock_client()),
    }
    stub = MagicMock()
    stub._covering = lambda scope, *a, **k: opened[scope.names[0]]
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)
    app = IngesterApp(config=_two_databases(tmp_path))

    def _shared(refs):
        raise AmbiguousDatabaseError("one location")

    monkeypatch.setattr("haiku.rag.ingester.app._refuse_shared_locations", _shared)

    with pytest.raises(RuntimeError, match="close failed") as raised:
        await app.run_batch()

    assert isinstance(raised.value.__context__, AmbiguousDatabaseError)
    assert {name: o.closes for name, o in opened.items()} == {"papers": 1, "wiki": 1}


async def test_a_shutdown_closes_the_sources_of_an_interrupted_removal(
    tmp_path, use_client, monkeypatch
):
    use_client({"papers": _mock_client()})
    app = IngesterApp(config=_papers_only(tmp_path))
    deleting = asyncio.Event()
    closed = []

    async def _stalls(source_ids):
        deleting.set()
        await asyncio.Event().wait()

    async with _serving(app):
        assert app._jobs is not None
        monkeypatch.setattr(app._jobs, "cancel_sources", _stalls)
        monkeypatch.setattr(
            "haiku.rag.sources.fs.FSSource.aclose",
            lambda self: _record(closed, self.source_id),
        )
        removal = asyncio.create_task(app.remove_database("papers"))
        await asyncio.wait_for(deleting.wait(), timeout=5.0)
        assert _pollers(app).pollers == []
    removal.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await removal

    assert closed == ["papers"]


def _opening(opened: dict, refused: dict[str, type[Exception]] | None = None):
    """A `_covering` double opening each database through `opened`, and
    raising `refused[name]` for the names it holds."""

    @asynccontextmanager
    async def _refuse(error):
        raise error("Database requires migration")
        yield

    def _covering(scope, *a, **k):
        name = scope.names[0]
        if refused and name in refused:
            return _refuse(refused[name])
        return opened[name]

    stub = MagicMock()
    stub._covering = _covering
    return stub


def _reconcile_failing(names: set[str]):
    async def _reconcile(rag, sync, source_ids):
        if names & set(source_ids):
            raise RuntimeError("store unreachable")
        return await reconcile(rag, sync, source_ids)

    return _reconcile


_APP_LOGGER = logging.getLogger("haiku.rag.ingester.app")


@pytest.mark.parametrize("error", [MigrationRequiredError, UpgradeRequiredError])
async def test_a_database_failing_to_open_at_startup_is_skipped(
    tmp_path, monkeypatch, error
):
    papers = _mock_client()
    monkeypatch.setattr(
        "haiku.rag.client.HaikuRAG",
        _opening({"papers": _Opened(papers)}, refused={"wiki": error}),
    )
    closed = []
    monkeypatch.setattr(
        "haiku.rag.sources.fs.FSSource.aclose",
        lambda self: _record(closed, self.source_id),
    )
    app = IngesterApp(config=_two_databases(tmp_path))

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        async with _serving(app):
            await _wait_until(lambda: _ingested(papers) == ["papers.md"])

            assert list(app._running) == ["papers"]
            assert app._failed == {
                "wiki": "database 'wiki': Database requires migration"
            }
            assert [p.source_id for p in _pollers(app).pollers] == ["papers"]
            assert closed == ["wiki"]
            async with _api(app) as api:
                health = (await api.get("/health")).json()
            assert health["status"] == "degraded"
            assert health["failed_databases"] == [
                {
                    "database": "wiki",
                    "error": "database 'wiki': Database requires migration",
                }
            ]

    assert [record.getMessage() for record in records] == [
        "Database 'wiki' failed to start and is skipped: database 'wiki': "
        "Database requires migration"
    ]


async def test_a_database_failing_to_reconcile_at_startup_is_skipped(
    tmp_path, monkeypatch
):
    papers, wiki = _Opened(_mock_client()), _Opened(_mock_client())
    monkeypatch.setattr(
        "haiku.rag.client.HaikuRAG", _opening({"papers": papers, "wiki": wiki})
    )
    monkeypatch.setattr(
        "haiku.rag.ingester.app.reconcile", _reconcile_failing({"wiki"})
    )
    app = IngesterApp(config=_two_databases(tmp_path))

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        async with _serving(app):
            await _wait_until(lambda: _ingested(papers.client) == ["papers.md"])

            assert list(app._running) == ["papers"]
            assert app._failed == {"wiki": "store unreachable"}
            assert [p.source_id for p in _pollers(app).pollers] == ["papers"]
            assert wiki.closes == 1
            assert _ingested(wiki.client) == []
    assert [record.getMessage() for record in records] == [
        "Database 'wiki' failed to start and is skipped: store unreachable"
    ]


async def test_a_skipped_database_failing_to_close_is_logged_and_dropped(
    tmp_path, monkeypatch
):
    papers, wiki = _Opened(_mock_client()), _FailsToCloseOnce(_mock_client())
    monkeypatch.setattr(
        "haiku.rag.client.HaikuRAG", _opening({"papers": papers, "wiki": wiki})
    )
    monkeypatch.setattr(
        "haiku.rag.ingester.app.reconcile", _reconcile_failing({"wiki"})
    )
    app = IngesterApp(config=_two_databases(tmp_path))

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        async with _serving(app):
            await _wait_until(lambda: _ingested(papers.client) == ["papers.md"])

            assert app._failed == {"wiki": "store unreachable"}

    assert wiki.closes == 1
    assert papers.closes == 1
    assert [record.getMessage() for record in records] == [
        "Database 'wiki' failed to start and is skipped: store unreachable",
        "Database 'wiki' failed to close: close failed",
    ]


async def test_a_database_with_an_unknown_metadata_provider_is_never_opened(
    tmp_path, monkeypatch
):
    opened = []
    papers = _Opened(_mock_client())

    def _covering(scope, *a, **k):
        opened.append(scope.names[0])
        return papers

    stub = MagicMock()
    stub._covering = _covering
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)
    config = _two_databases(tmp_path)
    config.ingester.databases["wiki"].sources[0].metadata_provider = "missing"
    app = IngesterApp(config=config)

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        async with _serving(app):
            assert opened == ["papers"]
            assert "unknown metadata provider 'missing'" in app._failed["wiki"]
    assert [record.getMessage() for record in records] == [
        "Database 'wiki' failed to start and is skipped: Source 'wiki' references unknown metadata provider 'missing'; no entry point registered under 'haiku.rag.metadata_providers'."
    ]


async def test_every_adapter_of_a_skipped_database_is_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "haiku.rag.client.HaikuRAG",
        _opening(
            {"papers": _Opened(_mock_client())},
            refused={"wiki": MigrationRequiredError},
        ),
    )
    config = _two_databases(tmp_path)
    (tmp_path / "drafts").mkdir()
    config.ingester.databases["wiki"].sources.append(
        FSSourceConfig(
            type="fs", id="drafts", root=tmp_path / "drafts", poll_interval_s=3600.0
        )
    )
    closed = []

    async def _aclose(self):
        closed.append(self.source_id)
        if self.source_id == "wiki":
            raise RuntimeError("adapter close failed")

    monkeypatch.setattr("haiku.rag.sources.fs.FSSource.aclose", _aclose)
    app = IngesterApp(config=config)

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        async with _serving(app):
            assert sorted(closed) == ["drafts", "wiki"]

    assert "Database 'wiki' failed to close: adapter close failed" in [
        record.getMessage() for record in records
    ]


def _closing_database(config, tmp_path, opened, *failures: BaseException | None):
    """A database whose adapters fail to close with `failures`, in order."""
    from haiku.rag.ingester.app import _Database

    async def _aclose(failure):
        if failure is not None:
            raise failure

    pollers = [
        SimpleNamespace(source=SimpleNamespace(aclose=partial(_aclose, failure)))
        for failure in failures
    ]
    ref, _ = _wiki(config, tmp_path)
    return _Database(ref, opened, opened.client, pollers, {})  # ty: ignore[invalid-argument-type]


async def test_closing_a_database_runs_every_close_and_raises_the_first_failure(
    tmp_path,
):
    config = _papers_only(tmp_path)
    opened = _FailsToCloseOnce(_mock_client())
    database = _closing_database(
        config, tmp_path, opened, RuntimeError("first"), None, RuntimeError("second")
    )

    with pytest.raises(RuntimeError, match="first"):
        await IngesterApp(config=config)._close(database)

    assert opened.closes == 1


async def test_closing_a_database_raises_a_cancellation_over_a_failure(tmp_path):
    config = _papers_only(tmp_path)
    opened = _FailsToCloseOnce(_mock_client())
    database = _closing_database(config, tmp_path, opened, asyncio.CancelledError())
    app = IngesterApp(config=config)

    with pytest.raises(asyncio.CancelledError):
        await app._discard(database, RuntimeError("store unreachable"))

    assert opened.closes == 1


@pytest.mark.parametrize("step", ["open", "reconcile"])
async def test_startup_fails_when_every_database_fails_and_deletes_no_job(
    tmp_path, monkeypatch, step
):
    from haiku.rag.ingester.exceptions import NoDatabaseStartedError

    if step == "open":
        stub = _opening(
            {},
            refused={"papers": MigrationRequiredError, "wiki": MigrationRequiredError},
        )
    else:
        stub = _opening(
            {"papers": _Opened(_mock_client()), "wiki": _Opened(_mock_client())}
        )
        monkeypatch.setattr(
            "haiku.rag.ingester.app.reconcile",
            _reconcile_failing({"papers", "wiki"}),
        )
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)
    config = _two_databases(tmp_path)
    await _seed_unconfigured_jobs(config)

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        with pytest.raises(NoDatabaseStartedError) as raised:
            await IngesterApp(config=config).run_batch()
    assert [record.getMessage() for record in records] == [
        f"Database {name!r} failed to start and is skipped: "
        + (
            f"database {name!r}: Database requires migration"
            if step == "open"
            else "store unreachable"
        )
        for name in ("papers", "wiki")
    ]

    assert "'papers': " in str(raised.value) and "'wiki': " in str(raised.value)
    assert await _jobs_of(config, "removed") == {
        "claimed": JobStatus.CLAIMED,
        "dead": JobStatus.DEAD,
        "queued": JobStatus.QUEUED,
    }


async def test_run_batch_ingests_the_databases_that_started(tmp_path, monkeypatch):
    from haiku.rag.ingester.app import FailedDatabase

    papers = _mock_client()
    monkeypatch.setattr(
        "haiku.rag.client.HaikuRAG",
        _opening({"papers": _Opened(papers)}, refused={"wiki": MigrationRequiredError}),
    )

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        report = await IngesterApp(config=_two_databases(tmp_path)).run_batch()
    assert [record.getMessage() for record in records] == [
        "Database 'wiki' failed to start and is skipped: database 'wiki': Database requires migration"
    ]

    assert report.succeeded == 1
    assert _ingested(papers) == ["papers.md"]
    assert report.failed_databases == [
        FailedDatabase(
            database="wiki", error="database 'wiki': Database requires migration"
        )
    ]


async def test_a_manifest_replay_refuses_a_change_of_a_skipped_database(
    tmp_path, monkeypatch
):
    config = _two_databases(tmp_path)
    manifest = (await IngesterApp(config=config).run_batch_dry_run()).manifest
    monkeypatch.setattr(
        "haiku.rag.client.HaikuRAG",
        _opening(
            {"papers": _Opened(_mock_client())},
            refused={"wiki": MigrationRequiredError},
        ),
    )

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        with pytest.raises(ValueError, match="does not ingest: wiki"):
            await IngesterApp(config=config).run_batch_from_manifest(manifest)
    assert [record.getMessage() for record in records] == [
        "Database 'wiki' failed to start and is skipped: database 'wiki': Database requires migration"
    ]


async def test_a_manifest_replay_without_changes_of_a_skipped_database_reports_it(
    tmp_path, use_client, monkeypatch
):
    papers, wiki = _mock_client(), _mock_client()
    use_client({"papers": papers, "wiki": wiki})
    config = _two_databases(tmp_path)
    await IngesterApp(config=config).run_batch()
    (tmp_path / "papers" / "more.md").write_text("more")
    manifest = (await IngesterApp(config=config).run_batch_dry_run()).manifest
    assert {change.source_id for change in manifest.changes} == {"papers"}
    monkeypatch.setattr(
        "haiku.rag.client.HaikuRAG",
        _opening({"papers": _Opened(papers)}, refused={"wiki": MigrationRequiredError}),
    )

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        report = await IngesterApp(config=config).run_batch_from_manifest(manifest)
    assert [record.getMessage() for record in records] == [
        "Database 'wiki' failed to start and is skipped: database 'wiki': Database requires migration"
    ]

    assert report.succeeded == 1
    assert [failed.database for failed in report.failed_databases] == ["wiki"]

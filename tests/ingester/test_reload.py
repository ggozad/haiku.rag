import asyncio
import contextlib
import logging
import os
import signal
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import MagicMock

import httpx
import pytest
import yaml
from httpx import ASGITransport

from haiku.rag.config import AppConfig, load_yaml_config
from haiku.rag.ingester.app import IngesterApp
from haiku.rag.ingester.exceptions import (
    InvalidConfigError,
    NothingToIngestError,
    ReloadRefusedError,
)
from haiku.rag.sources.base import SourceEvent, SourceEventKind
from tests.conftest import capture_logs
from tests.ingester.test_lifecycle import _gate, _pollers, _pool, _serving
from tests.ingester.test_run_batch import _ingested, _mock_client, _wait_until

_APP_LOGGER = logging.getLogger("haiku.rag.ingester.app")

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


def _fs(tmp_path, source_id: str, root: str | None = None) -> dict:
    """An FS source over `tmp_path / root`, holding one file named after it."""
    directory = tmp_path / (root or source_id)
    directory.mkdir(exist_ok=True)
    (directory / f"{source_id}.md").write_text(source_id)
    return {
        "type": "fs",
        "id": source_id,
        "root": str(directory),
        "poll_interval_s": 3600.0,
    }


def _settings(tmp_path, databases: dict[str, tuple[str, list[dict]]]) -> dict:
    """A configuration placing each database at `tmp_path / location` and
    ingesting it from its sources."""
    return {
        "lancedb": {
            "databases": {
                name: str(tmp_path / location)
                for name, (location, _) in databases.items()
            }
        },
        "ingester": {
            "queue": {"path": str(tmp_path / "queue.db")},
            "workers": {
                "worker_count": 2,
                "poll_idle_interval_s": 0.05,
                "retry": {"max_attempts": 1},
                # What `_serving` sets on the running configuration.
                "shutdown_grace_s": 0.2,
            },
            "api": {"enabled": False},
            "databases": {
                name: {"sources": sources} for name, (_, sources) in databases.items()
            },
        },
    }


def _write(path, settings: dict) -> None:
    path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")


def _app(path, **kwargs) -> IngesterApp:
    config = AppConfig.model_validate(load_yaml_config(path))
    return IngesterApp(config=config, config_path=path, **kwargs)


def _changes(report) -> set[tuple[str, str]]:
    return {(change.database, change.change) for change in report.databases}


def _api(app: IngesterApp, auth_token: str | None = None) -> httpx.AsyncClient:
    from haiku.rag.ingester.api.server import build_app

    return httpx.AsyncClient(
        transport=ASGITransport(app=build_app(app._api_state(), auth_token=auth_token)),
        base_url="http://testserver",
    )


def _refusing(names: set[str], clients: dict):
    """A `_covering` double whose open of a database in `names` fails."""
    from haiku.rag.store.exceptions import MigrationRequiredError

    @asynccontextmanager
    async def _open(scope):
        if scope.names[0] in names:
            raise MigrationRequiredError("Database requires migration")
        yield clients[scope.names[0]]

    stub = MagicMock()
    stub._covering = lambda scope, *a, **k: _open(scope)
    return stub


@pytest.fixture
def clients(use_client):
    clients = defaultdict(_mock_client)
    use_client(clients)
    return clients


# --- what a reload applies ---


async def test_a_reload_applies_each_kind_of_change(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    papers, wiki, notes, kept, legacy = (
        _fs(tmp_path, name) for name in ("papers", "wiki", "notes", "kept", "legacy")
    )
    _write(
        path,
        _settings(
            tmp_path,
            {
                "papers": ("papers.lancedb", [papers]),
                "wiki": ("wiki.lancedb", [wiki]),
                "notes": ("notes.lancedb", [notes]),
                "kept": ("kept.lancedb", [kept]),
                "legacy": ("legacy.lancedb", [legacy]),
            },
        ),
    )
    app = _app(path)

    async with _serving(app):
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "papers": ("papers-moved.lancedb", [papers]),
                    "encyclopedia": ("wiki.lancedb", [wiki]),
                    "notes": ("notes.lancedb", [_fs(tmp_path, "notes", "drafts")]),
                    "kept": ("kept.lancedb", [kept]),
                    "fresh": ("fresh.lancedb", [_fs(tmp_path, "fresh")]),
                },
            ),
        )
        report = await app.reload()

        assert report.outcome == "applied"
        assert _changes(report) == {
            ("legacy", "removed"),
            ("papers", "removed"),
            ("papers", "added"),
            ("wiki", "removed"),
            ("encyclopedia", "added"),
            ("notes", "removed"),
            ("notes", "added"),
            ("fresh", "added"),
        }
        assert sorted(app._running) == [
            "encyclopedia",
            "fresh",
            "kept",
            "notes",
            "papers",
        ]
        assert str(app._running["papers"].ref.location).endswith("papers-moved.lancedb")
        assert app._source_databases["wiki"] == "encyclopedia"
        await _wait_until(lambda: _ingested(clients["fresh"]) == ["fresh.md"])
    assert clients["kept"]._ingest_observed.await_count == 1


async def test_a_change_outside_the_databases_is_refused(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    settings = _settings(
        tmp_path, {"papers": ("papers.lancedb", [_fs(tmp_path, "papers")])}
    )
    _write(path, settings)
    app = _app(path)

    async with _serving(app):
        settings["ingester"]["databases"]["wiki"] = {"sources": [_fs(tmp_path, "wiki")]}
        settings["lancedb"]["databases"]["wiki"] = str(tmp_path / "wiki.lancedb")
        settings["ingester"]["workers"]["worker_count"] = 3
        settings["processing"] = {"chunk_size": 128}
        settings["environment"] = "development"
        _write(path, settings)

        with pytest.raises(
            ReloadRefusedError, match=r"ingester\.workers.*processing|processing.*"
        ) as refused:
            await app.reload()

        assert "ingester.workers" in str(refused.value)
        assert "processing.chunk_size" in str(refused.value)
        assert "environment" in str(refused.value)
        assert list(app._running) == ["papers"]


@pytest.mark.parametrize(
    "overrides",
    [{"host": "0.0.0.0"}, {"port": 9999}, {"root_path": "/ingester"}],
)
async def test_a_reload_under_api_overrides_applies_a_database_change(
    tmp_path, clients, overrides
):
    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    config = AppConfig.model_validate(load_yaml_config(path))
    for field, value in overrides.items():
        setattr(config.ingester.api, field, value)
    app = IngesterApp(config=config, config_path=path, api_overrides=overrides)

    async with _serving(app):
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "papers": ("papers.lancedb", [papers]),
                    "wiki": ("wiki.lancedb", [_fs(tmp_path, "wiki")]),
                },
            ),
        )
        report = await app.reload()

    assert _changes(report) == {("wiki", "added")}


async def test_a_reload_with_an_unreadable_addition_changes_nothing(
    tmp_path, monkeypatch
):
    clients = defaultdict(_mock_client)
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", _refusing({"broken"}, clients))
    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    app = _app(path)

    async with _serving(app):
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "papers": ("papers.lancedb", [papers]),
                    "good": ("good.lancedb", [_fs(tmp_path, "good")]),
                    "broken": ("broken.lancedb", [_fs(tmp_path, "broken")]),
                },
            ),
        )
        report = await app.reload()

        assert report.outcome == "failed"
        assert _changes(report) == {("broken", "failed")}
        assert "database 'broken'" in (report.databases[0].error or "")
        assert list(app._running) == ["papers"]
        assert sorted(_pool(app)._served) == ["papers"]
        assert [p.source_id for p in _pollers(app).pollers] == ["papers"]


async def test_a_reload_failing_after_removals_began_applies_the_rest_next_time(
    tmp_path, monkeypatch
):
    clients = defaultdict(_mock_client)
    refused = {"renamed"}
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", _refusing(refused, clients))
    path = tmp_path / "haiku.rag.yaml"
    papers, legacy = _fs(tmp_path, "papers"), _fs(tmp_path, "legacy")
    _write(
        path,
        _settings(
            tmp_path,
            {
                "papers": ("papers.lancedb", [papers]),
                "legacy": ("legacy.lancedb", [legacy]),
            },
        ),
    )
    app = _app(path)

    async with _serving(app), _api(app) as api:
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "papers": ("papers.lancedb", [papers]),
                    "renamed": ("legacy.lancedb", [legacy]),
                },
            ),
        )
        first_resp = await api.post("/reload")
        first = first_resp.json()

        assert first_resp.status_code == 500
        assert first["outcome"] == "partial"
        assert {(c["database"], c["change"]) for c in first["databases"]} == {
            ("legacy", "removed"),
            ("renamed", "failed"),
        }
        assert "legacy:" in (await api.get("/config")).json()["yaml"]
        sources = (await api.get("/sources")).json()
        assert [(s["source_id"], s["database"]) for s in sources] == [
            ("papers", "papers")
        ]

        refused.clear()
        second_resp = await api.post("/reload")
        second = second_resp.json()

        assert second_resp.status_code == 200
        assert second["outcome"] == "applied"
        assert {(c["database"], c["change"]) for c in second["databases"]} == {
            ("renamed", "added")
        }
        assert "renamed:" in (await api.get("/config")).json()["yaml"]


async def test_a_removal_left_removing_is_completed_first_by_the_next_reload(
    tmp_path, clients, monkeypatch
):
    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    app = _app(path)

    async with _serving(app):
        assert app._jobs is not None
        cancel_sources = app._jobs.cancel_sources
        failures = [RuntimeError("queue unreachable")] * 2

        async def _fails_twice(source_ids):
            if failures:
                raise failures.pop()
            return await cancel_sources(source_ids)

        monkeypatch.setattr(app._jobs, "cancel_sources", _fails_twice)
        _write(
            path,
            _settings(
                tmp_path,
                {"papers": ("papers.lancedb", [_fs(tmp_path, "papers", "drafts")])},
            ),
        )
        first = await app.reload()

        assert first.outcome == "failed"
        assert _changes(first) == {("papers", "failed")}
        assert app._running["papers"].removing
        assert _changes(await app.reload()) == {("papers", "failed")}

        second = await app.reload()

        assert second.outcome == "applied"
        assert [(c.database, c.change) for c in second.databases] == [
            ("papers", "removed"),
            ("papers", "added"),
        ]
        assert not app._running["papers"].removing


async def test_an_addition_failing_to_reconcile_is_reported(
    tmp_path, clients, monkeypatch
):
    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    app = _app(path)

    async def _fails(rag, sync, source_ids):
        raise RuntimeError("store unreachable")

    async with _serving(app):
        monkeypatch.setattr("haiku.rag.ingester.app.reconcile", _fails)
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "papers": ("papers.lancedb", [papers]),
                    "wiki": ("wiki.lancedb", [_fs(tmp_path, "wiki")]),
                },
            ),
        )
        report = await app.reload()

        assert report.outcome == "failed"
        assert report.databases[0].error == "store unreachable"
        assert list(app._running) == ["papers"]


def _case_insensitive(directory) -> bool:
    (directory / "Probe").mkdir()
    return (directory / "probe").exists()


async def test_two_new_names_for_one_directory_are_refused_once_opened(
    tmp_path, monkeypatch
):
    if not _case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")
    clients = defaultdict(_mock_client)

    @asynccontextmanager
    async def _creating(scope):
        scope.databases[0].location.mkdir(exist_ok=True)
        yield clients[scope.names[0]]

    stub = MagicMock()
    stub._covering = lambda scope, *a, **k: _creating(scope)
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    app = _app(path)

    async with _serving(app):
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "upper": ("Shared.lancedb", [_fs(tmp_path, "upper")]),
                    "lower": ("shared.lancedb", [_fs(tmp_path, "lower")]),
                },
            ),
        )
        report = await app.reload()

        assert report.outcome == "failed"
        assert _changes(report) == {("lower", "failed")}
        assert "'upper' and 'lower'" in (report.databases[0].error or "")
        assert app._running == {}


def _unwinding_hooks(monkeypatch) -> list[str]:
    """Configure a hook whose lifespan records the exception each client
    closes on, or "clean"."""
    from haiku.rag.hooks import Hook

    seen: list[str] = []

    class _Recording(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            try:
                yield
            except Exception as error:
                seen.append(type(error).__name__)
                raise
            seen.append("clean")

    class _EntryPoint:
        name = "recording"

        def load(self):
            return _Recording

    monkeypatch.setattr("haiku.rag.hooks.entry_points", lambda group: [_EntryPoint()])
    return seen


def _hooked(tmp_path, databases: dict[str, tuple[str, list[dict]]]) -> IngesterApp:
    settings = _settings(tmp_path, databases)
    settings["hooks"] = ["recording"]
    path = tmp_path / "haiku.rag.yaml"
    _write(path, settings)
    return _app(path)


async def test_hooks_see_the_exception_that_ends_ingestion(tmp_path, monkeypatch):
    seen = _unwinding_hooks(monkeypatch)
    app = _hooked(tmp_path, {"papers": ("papers", [_fs(tmp_path, "papers")])})

    with pytest.raises(ValueError, match="boom"):
        async with app._resources():
            raise ValueError("boom")

    assert seen == ["ValueError"]


async def test_hooks_see_the_exception_that_fails_startup(tmp_path, monkeypatch):
    seen = _unwinding_hooks(monkeypatch)
    wiki = {**_fs(tmp_path, "wiki"), "metadata_provider": "missing"}
    app = _hooked(
        tmp_path,
        {
            "papers": ("papers", [_fs(tmp_path, "papers")]),
            "wiki": ("wiki", [wiki]),
        },
    )

    with pytest.raises(ValueError, match="missing"):
        async with app._resources():
            pass

    assert seen == ["ValueError", "ValueError"]


async def test_hooks_see_the_exception_that_fails_a_reload(tmp_path, monkeypatch):
    seen = _unwinding_hooks(monkeypatch)
    app = _hooked(tmp_path, {})

    async def _fails(rag, sync, source_ids):
        raise RuntimeError("store unreachable")

    async with _serving(app):
        monkeypatch.setattr("haiku.rag.ingester.app.reconcile", _fails)
        _write(
            tmp_path / "haiku.rag.yaml",
            {
                **_settings(
                    tmp_path, {"wiki": ("wiki.lancedb", [_fs(tmp_path, "wiki")])}
                ),
                "hooks": ["recording"],
            },
        )
        report = await app.reload()

        assert report.outcome == "failed"
        assert seen == ["RuntimeError"]


@pytest.mark.parametrize("retried_by", ["reload", "shutdown"])
async def test_a_failed_close_is_reported_and_retried(
    tmp_path, monkeypatch, retried_by
):
    from tests.ingester.test_lifecycle import _FailsToCloseOnce

    clients = defaultdict(_mock_client)
    refused = {"renamed"}
    opened = {"good": _FailsToCloseOnce(clients["good"])}

    @asynccontextmanager
    async def _plain(scope):
        yield clients[scope.names[0]]

    def _covering(scope, *a, **k):
        name = scope.names[0]
        if name in refused:
            return _refusing(refused, clients)._covering(scope)
        return opened.get(name) or _plain(scope)

    stub = MagicMock()
    stub._covering = _covering
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", stub)
    path = tmp_path / "haiku.rag.yaml"
    legacy = _fs(tmp_path, "legacy")
    _write(path, _settings(tmp_path, {"legacy": ("legacy.lancedb", [legacy])}))
    app = _app(path)

    async with _serving(app):
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "good": ("good.lancedb", [_fs(tmp_path, "good")]),
                    "renamed": ("legacy.lancedb", [legacy]),
                },
            ),
        )
        report = await app.reload()
        assert opened["good"].closes == 1
        if retried_by == "reload":
            _write(path, _settings(tmp_path, {}))
            await app.reload()
            assert opened["good"].closes == 2

    assert report.outcome == "partial"
    assert [(c.database, c.change) for c in report.databases] == [
        ("legacy", "removed"),
        ("renamed", "failed"),
        ("good", "failed"),
    ]
    assert report.databases[2].error == "close failed"
    assert opened["good"].closes == 2


class _Closing:
    """Opens `client`, failing its first `fail` closes and holding the next
    one until `release` is set when `hold` is."""

    def __init__(self, client, *, fail: int = 0, hold: bool = False):
        self.client = client
        self.fail = fail
        self.closes = 0
        self.holding = asyncio.Event()
        self.release = asyncio.Event()
        if not hold:
            self.release.set()

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, *exc_info):
        self.closes += 1
        if self.closes <= self.fail:
            raise RuntimeError("close failed")
        self.holding.set()
        await self.release.wait()
        return False


def _opening(opened: dict, clients: dict, refused: set[str]):
    """A `_covering` double: `opened` names the openers of some databases,
    `refused` the databases that fail to open."""

    @asynccontextmanager
    async def _plain(scope):
        yield clients[scope.names[0]]

    def _covering(scope, *a, **k):
        name = scope.names[0]
        if name in refused:
            return _refusing(refused, clients)._covering(scope)
        return opened.get(name) or _plain(scope)

    stub = MagicMock()
    stub._covering = _covering
    return stub


def _leave_good_unclosed(tmp_path, monkeypatch, good: _Closing):
    """An app whose last reload left `good` prepared and unclosed."""
    clients = defaultdict(_mock_client)
    monkeypatch.setattr(
        "haiku.rag.client.HaikuRAG",
        _opening({"good": good}, clients, {"broken"}),
    )
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    app = _app(path)
    return app, path


async def test_a_reload_stops_when_a_leftover_close_fails_again(tmp_path, monkeypatch):
    good = _Closing(_mock_client(), fail=2)
    app, path = _leave_good_unclosed(tmp_path, monkeypatch, good)

    async with _serving(app):
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "good": ("good.lancedb", [_fs(tmp_path, "good")]),
                    "broken": ("broken.lancedb", [_fs(tmp_path, "broken")]),
                },
            ),
        )
        await app.reload()
        _write(
            path,
            _settings(tmp_path, {"other": ("other.lancedb", [_fs(tmp_path, "other")])}),
        )
        report = await app.reload()

        assert report.outcome == "failed"
        assert _changes(report) == {("good", "failed")}
        assert app._running == {}
        assert "other" not in app._config.ingester.databases
    assert good.closes == 3


async def test_an_addition_failing_to_apply_and_to_close_is_closed_at_shutdown(
    tmp_path, monkeypatch
):
    good = _Closing(_mock_client(), fail=1)
    app, path = _leave_good_unclosed(tmp_path, monkeypatch, good)

    async def _fails(rag, sync, source_ids):
        raise RuntimeError("store unreachable")

    async with _serving(app):
        monkeypatch.setattr("haiku.rag.ingester.app.reconcile", _fails)
        _write(
            path,
            _settings(tmp_path, {"good": ("good.lancedb", [_fs(tmp_path, "good")])}),
        )
        report = await app.reload()

        assert [(c.database, c.error) for c in report.databases] == [
            ("good", "store unreachable"),
            ("good", "close failed"),
        ]
        assert good.closes == 1
    assert good.closes == 2


async def test_a_cancelled_close_retry_keeps_the_leftover(tmp_path, monkeypatch):
    good = _Closing(_mock_client(), fail=1, hold=True)
    app, path = _leave_good_unclosed(tmp_path, monkeypatch, good)

    async with _serving(app):
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "good": ("good.lancedb", [_fs(tmp_path, "good")]),
                    "broken": ("broken.lancedb", [_fs(tmp_path, "broken")]),
                },
            ),
        )
        await app.reload()
        retry = app._start_reload()
        await asyncio.wait_for(good.holding.wait(), timeout=5.0)
        retry.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await retry

        assert [database.ref.name for database in app._prepared] == ["good"]
        good.release.set()


async def test_concurrent_reloads_apply_one_after_the_other(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    papers, wiki = _fs(tmp_path, "papers"), _fs(tmp_path, "wiki")
    _write(
        path,
        _settings(
            tmp_path,
            {"papers": ("papers.lancedb", [papers]), "wiki": ("wiki.lancedb", [wiki])},
        ),
    )
    gate = asyncio.Event()
    _gate(clients["papers"], gate)
    app = _app(path)

    async with _serving(app):
        await _wait_until(lambda: "papers" in _pool(app)._executions.values())
        _write(path, _settings(tmp_path, {"wiki": ("wiki.lancedb", [wiki])}))
        first = asyncio.create_task(app.reload())
        await _wait_until(lambda: "papers" not in _pool(app)._served)
        second = asyncio.create_task(app.reload())
        await asyncio.sleep(0.1)
        assert not second.done()

        gate.set()
        reports = await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)

    assert [_changes(report) for report in reports] == [{("papers", "removed")}, set()]


async def test_a_client_that_stops_waiting_does_not_cancel_the_reload(
    tmp_path, clients
):
    path = tmp_path / "haiku.rag.yaml"
    papers, wiki = _fs(tmp_path, "papers"), _fs(tmp_path, "wiki")
    _write(
        path,
        _settings(
            tmp_path,
            {"papers": ("papers.lancedb", [papers]), "wiki": ("wiki.lancedb", [wiki])},
        ),
    )
    gate = asyncio.Event()
    _gate(clients["papers"], gate)
    app = _app(path)

    async with _serving(app):
        await _wait_until(lambda: "papers" in _pool(app)._executions.values())
        _write(path, _settings(tmp_path, {"wiki": ("wiki.lancedb", [wiki])}))
        caller = asyncio.create_task(app.reload())
        await _wait_until(lambda: "papers" not in _pool(app)._served)
        caller.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await caller

        gate.set()
        await _wait_until(lambda: "papers" not in app._running)


async def test_removing_the_last_database_then_adding_one(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    app = _app(path)

    async with _serving(app):
        _write(path, _settings(tmp_path, {}))
        assert _changes(await app.reload()) == {("papers", "removed")}
        assert app._running == {}

        _write(
            path,
            _settings(tmp_path, {"wiki": ("wiki.lancedb", [_fs(tmp_path, "wiki")])}),
        )
        assert _changes(await app.reload()) == {("wiki", "added")}
        await _wait_until(lambda: _ingested(clients["wiki"]) == ["wiki.md"])


async def test_a_reload_without_a_configuration_file_is_refused(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    config = AppConfig.model_validate(load_yaml_config(path))
    app = IngesterApp(config=config)

    async with _serving(app):
        with pytest.raises(ReloadRefusedError, match="configuration file"):
            await app.reload()


def test_a_selected_database_cannot_be_reloaded(tmp_path):
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {"papers": ("papers", [_fs(tmp_path, "p")])}))

    with pytest.raises(ValueError, match="database_name"):
        _app(path, database_name="papers")


async def test_an_invalid_configuration_file_changes_nothing(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    app = _app(path)

    async with _serving(app):
        settings = _settings(tmp_path, {"papers": ("papers.lancedb", [papers])})
        settings["ingester"]["databases"]["twin"] = {"sources": [papers]}
        settings["lancedb"]["databases"]["twin"] = str(tmp_path / "twin.lancedb")
        _write(path, settings)

        with pytest.raises(InvalidConfigError, match="papers"):
            await app.reload()
        assert list(app._running) == ["papers"]


# --- starting on an empty set ---


async def test_serve_starts_on_an_empty_set_and_ingests_after_a_reload(
    tmp_path, clients
):
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    app = _app(path)

    with capture_logs(_APP_LOGGER, logging.WARNING) as records:
        async with _serving(app):
            _write(
                path,
                _settings(
                    tmp_path, {"papers": ("papers.lancedb", [_fs(tmp_path, "papers")])}
                ),
            )
            assert _changes(await app.reload()) == {("papers", "added")}
            await _wait_until(lambda: _ingested(clients["papers"]) == ["papers.md"])

    assert any("no database" in record.getMessage() for record in records)


async def test_run_batch_refuses_an_empty_set(tmp_path):
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))

    with pytest.raises(NothingToIngestError, match="ingester.databases"):
        await _app(path).run_batch()


# --- unrelated databases keep working ---


async def test_other_databases_keep_claiming_during_a_removals_drain(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    papers, wiki = _fs(tmp_path, "papers"), _fs(tmp_path, "wiki")
    _write(
        path,
        _settings(
            tmp_path,
            {"papers": ("papers.lancedb", [papers]), "wiki": ("wiki.lancedb", [wiki])},
        ),
    )
    gate = asyncio.Event()
    _gate(clients["papers"], gate)
    app = _app(path)

    async with _serving(app):
        await _wait_until(
            lambda: (
                _ingested(clients["wiki"]) == ["wiki.md"]
                and list(_pool(app)._executions.values()) == ["papers"]
            )
        )
        _write(path, _settings(tmp_path, {"wiki": ("wiki.lancedb", [wiki])}))
        reload = asyncio.create_task(app.reload())
        await _wait_until(lambda: "papers" not in _pool(app)._served)

        (tmp_path / "wiki" / "more.md").write_text("more")
        wiki_poller = next(p for p in _pollers(app).pollers if p.source_id == "wiki")
        assert await wiki_poller._sweep_once()
        await _wait_until(lambda: _ingested(clients["wiki"]) == ["wiki.md", "more.md"])
        assert not reload.done()

        gate.set()
        await asyncio.wait_for(reload, timeout=5.0)


async def test_other_databases_keep_claiming_during_a_refresh_sweep(
    tmp_path, clients, monkeypatch
):
    path = tmp_path / "haiku.rag.yaml"
    papers, wiki = _fs(tmp_path, "papers"), _fs(tmp_path, "wiki")
    _write(
        path,
        _settings(
            tmp_path,
            {"papers": ("papers.lancedb", [papers]), "wiki": ("wiki.lancedb", [wiki])},
        ),
    )
    app = _app(path)

    async with _serving(app), _api(app) as api:
        await _wait_until(
            lambda: (
                _ingested(clients["papers"]) == ["papers.md"]
                and _ingested(clients["wiki"]) == ["wiki.md"]
                and not _pool(app)._inflight
            )
        )
        poller = next(p for p in _pollers(app).pollers if p.source_id == "papers")
        listing, gate = asyncio.Event(), asyncio.Event()

        async def _discover(since=None, *, known_uris=None):
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
        refresh = asyncio.create_task(api.post("/sources/papers/refresh"))
        await asyncio.wait_for(listing.wait(), timeout=5.0)

        (tmp_path / "wiki" / "more.md").write_text("more")
        wiki_poller = next(p for p in _pollers(app).pollers if p.source_id == "wiki")
        assert await wiki_poller._sweep_once()
        await _wait_until(lambda: _ingested(clients["wiki"]) == ["wiki.md", "more.md"])

        gate.set()
        assert (await asyncio.wait_for(refresh, timeout=5.0)).json()["refreshed"]


# --- POST /reload ---


async def test_post_reload_reports_what_it_applied(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    app = _app(path)

    async with _serving(app), _api(app) as api:
        _write(
            path,
            _settings(
                tmp_path,
                {
                    "papers": ("papers.lancedb", [papers]),
                    "wiki": ("wiki.lancedb", [_fs(tmp_path, "wiki")]),
                },
            ),
        )
        resp = await api.post("/reload")

    assert resp.status_code == 200
    assert resp.json() == {
        "outcome": "applied",
        "databases": [{"database": "wiki", "change": "added", "error": None}],
    }


async def test_post_reload_answers_500_when_nothing_could_be_applied(
    tmp_path, monkeypatch
):
    clients = defaultdict(_mock_client)
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", _refusing({"broken"}, clients))
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    app = _app(path)

    async with _serving(app), _api(app) as api:
        _write(
            path,
            _settings(
                tmp_path, {"broken": ("broken.lancedb", [_fs(tmp_path, "broken")])}
            ),
        )
        resp = await api.post("/reload")

    assert resp.status_code == 500
    assert resp.json()["outcome"] == "failed"
    assert resp.json()["databases"][0]["database"] == "broken"


async def test_post_reload_answers_409_for_a_change_outside_the_databases(
    tmp_path, clients
):
    path = tmp_path / "haiku.rag.yaml"
    settings = _settings(tmp_path, {})
    _write(path, settings)
    app = _app(path)

    async with _serving(app), _api(app) as api:
        settings["ingester"]["workers"]["worker_count"] = 3
        _write(path, settings)
        resp = await api.post("/reload")

    assert resp.status_code == 409
    assert "ingester.workers" in resp.json()["detail"]


async def test_post_reload_answers_422_for_an_unreadable_file(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    app = _app(path)

    async with _serving(app), _api(app) as api:
        path.write_text("ingester: [", encoding="utf-8")
        resp = await api.post("/reload")

    assert resp.status_code == 422


async def test_post_reload_needs_the_token(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    app = _app(path)

    async with _serving(app), _api(app, auth_token="secret") as api:
        refused = await api.post("/reload")
        allowed = await api.post("/reload", headers={"Authorization": "Bearer secret"})

    assert refused.status_code == 401
    assert allowed.status_code == 200


async def test_post_reload_answers_503_without_a_running_app(jobs, sync):
    from haiku.rag.ingester.api.server import APIState, build_app

    state = APIState(config=AppConfig(), job_repo=jobs, sync_repo=sync)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=build_app(state)), base_url="http://testserver"
    ) as api:
        resp = await api.post("/reload")

    assert resp.status_code == 503


# --- SIGHUP and shutdown ---


@posix_only
async def test_sighup_applies_the_file(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    app = _app(path)

    async with _serving(app):
        _write(
            path,
            _settings(
                tmp_path, {"papers": ("papers.lancedb", [_fs(tmp_path, "papers")])}
            ),
        )
        os.kill(os.getpid(), signal.SIGHUP)
        await _wait_until(lambda: _ingested(clients["papers"]) == ["papers.md"])


@posix_only
async def test_a_failed_reload_under_sighup_is_logged(tmp_path, monkeypatch):
    clients = defaultdict(_mock_client)
    monkeypatch.setattr("haiku.rag.client.HaikuRAG", _refusing({"broken"}, clients))
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    app = _app(path)

    with capture_logs(_APP_LOGGER, logging.WARNING) as records:
        async with _serving(app):
            _write(
                path,
                _settings(
                    tmp_path, {"broken": ("broken.lancedb", [_fs(tmp_path, "broken")])}
                ),
            )
            os.kill(os.getpid(), signal.SIGHUP)
            await _wait_until(
                lambda: any("broken" in record.getMessage() for record in records)
            )

    messages = [record.getMessage() for record in records]
    assert any(
        "failed" in message and "database 'broken'" in message for message in messages
    )


@posix_only
async def test_a_refused_reload_under_sighup_is_logged(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    settings = _settings(tmp_path, {})
    _write(path, settings)
    app = _app(path)

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        async with _serving(app):
            settings["environment"] = "development"
            _write(path, settings)
            os.kill(os.getpid(), signal.SIGHUP)
            await _wait_until(lambda: bool(records))

    assert "Reload on SIGHUP failed: environment changed" in records[0].getMessage()


@posix_only
async def test_a_shutdown_cancels_a_reload_under_sighup(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    _gate(clients["papers"], asyncio.Event())
    app = _app(path)

    with capture_logs(_APP_LOGGER, logging.ERROR) as records:
        async with _serving(app):
            await _wait_until(lambda: "papers" in _pool(app)._executions.values())
            _write(path, _settings(tmp_path, {}))
            os.kill(os.getpid(), signal.SIGHUP)
            await _wait_until(lambda: not _pool(app)._served)

    assert app._running["papers"].removing
    assert not app._reloads
    assert records == []


@posix_only
async def test_sighups_during_a_reload_queue_one_more(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    _write(path, _settings(tmp_path, {}))
    app = _app(path)

    async with _serving(app):
        async with app._reload_lock:
            for _ in range(3):
                os.kill(os.getpid(), signal.SIGHUP)
                await asyncio.sleep(0.02)
            assert len(app._reloads) == 1
        await _wait_until(lambda: not app._reloads)


async def test_a_reload_requested_during_shutdown_is_refused(
    tmp_path, clients, monkeypatch
):
    from haiku.rag.ingester.exceptions import ReloadCancelledError

    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    app = _app(path)
    removing, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def _slow_removal(name):
        removing.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            raise

    monkeypatch.setattr(app, "remove_database", _slow_removal)
    serving = asyncio.create_task(app.serve(api=False))
    await _wait_until(lambda: app._pool is not None and app._pool.live_workers > 0)
    _write(path, _settings(tmp_path, {}))
    first = asyncio.create_task(app.reload())
    await asyncio.wait_for(removing.wait(), timeout=5.0)

    serving.cancel()
    await asyncio.wait_for(cleaning.wait(), timeout=5.0)
    with pytest.raises(ReloadCancelledError):
        await asyncio.wait_for(app.reload(), timeout=1.0)

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(serving, timeout=5.0)
    with pytest.raises(ReloadCancelledError):
        await first


@posix_only
async def test_sigterm_during_a_reload_waiting_on_a_drain_shuts_down(tmp_path, clients):
    path = tmp_path / "haiku.rag.yaml"
    papers = _fs(tmp_path, "papers")
    _write(path, _settings(tmp_path, {"papers": ("papers.lancedb", [papers])}))
    _gate(clients["papers"], asyncio.Event())
    app = _app(path)

    serving = asyncio.create_task(app.serve(api=False))
    await _wait_until(
        lambda: app._pool is not None and "papers" in app._pool._executions.values()
    )
    async with _api(app) as api:
        _write(path, _settings(tmp_path, {}))
        draining = asyncio.create_task(api.post("/reload"))
        await _wait_until(lambda: not _pool(app)._served)
        waiting = asyncio.create_task(api.post("/reload"))
        await asyncio.sleep(0.05)

        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(serving, timeout=5.0)
        responses = await asyncio.wait_for(
            asyncio.gather(draining, waiting), timeout=5.0
        )

    assert [response.status_code for response in responses] == [503, 503]

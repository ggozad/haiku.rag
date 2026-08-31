"""Hooks on a client that covers several databases."""

from contextlib import asynccontextmanager

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.hooks import Hook

from .helpers import _config, _seed


class SpyHook(Hook):
    def __init__(self):
        self.before: list[str] = []
        self.after: list[tuple] = []

    async def before_search(self, client, request):
        self.before.append(request.query)
        return request

    async def after_search(self, client, request, results):
        self.after.append((request.query, tuple(r.source for r in results)))
        return results


def _register_spies(monkeypatch, spies):
    """A config-activated hook, so facades would build their own copy."""

    class _EntryPoint:
        name = "spy"

        def load(self):
            def factory():
                spy = SpyHook()
                spies.append(spy)
                return spy

            return factory

    monkeypatch.setattr("haiku.rag.hooks.entry_points", lambda group: [_EntryPoint()])


@pytest.mark.asyncio
async def test_federated_search_fires_hooks_once(tmp_path, monkeypatch):
    """One search, one before_search and one after_search, however many
    databases it fans out to; the per-database facades run none."""
    spies: list[SpyHook] = []
    _register_spies(monkeypatch, spies)
    config = _config(tmp_path, ["a", "b"])
    config.hooks = ["spy"]
    await _seed(config, "a", ["alpha one"])
    await _seed(config, "b", ["beta one"])

    async with HaikuRAG(config=config) as client:
        results = await client.search("one", search_type="fts")

    assert {r.source for r in results} == {"a", "b"}
    before = [q for spy in spies for q in spy.before]
    after = [entry for spy in spies for entry in spy.after]
    assert before == ["one"]
    assert len(after) == 1
    assert set(after[0][1]) == {"a", "b"}


@pytest.mark.asyncio
async def test_single_database_selection_fires_hooks_once(tmp_path, monkeypatch):
    """The one-database shortcut delegates to a facade's search; the facade
    must not fire the hooks a second time."""
    spies: list[SpyHook] = []
    _register_spies(monkeypatch, spies)
    config = _config(tmp_path, ["a", "b"])
    config.hooks = ["spy"]
    await _seed(config, "a", ["alpha one"])
    await _seed(config, "b", ["beta one"])

    async with HaikuRAG(config=config) as client:
        results = await client.search("one", search_type="fts", sources=["a"])

    assert [r.source for r in results] == ["a"]
    before = [q for spy in spies for q in spy.before]
    assert before == ["one"]


@pytest.mark.asyncio
async def test_facades_borrow_without_hooks(tmp_path, monkeypatch):
    spies: list[SpyHook] = []
    _register_spies(monkeypatch, spies)
    config = _config(tmp_path, ["a", "b"])
    config.hooks = ["spy"]
    await _seed(config, "a", ["alpha one"])

    async with HaikuRAG(config=config) as client:
        [facade] = await client.clients_for(["a"])
        assert facade._hooks == []
        assert client._hooks != []


@pytest.mark.asyncio
async def test_lifespans_run_for_owners_not_facades(tmp_path):
    """The federating client owns its session and runs lifespans; the facades
    it lends out never do."""
    log: list[str] = []

    class _LifespanHook(Hook):
        @asynccontextmanager
        async def lifespan(self, client):
            log.append("enter")
            try:
                yield
            finally:
                log.append("exit")

    config = _config(tmp_path, ["a", "b"])
    await _seed(config, "a", ["alpha one"])

    client = HaikuRAG(config=config)
    client._hooks = [_LifespanHook()]
    async with client:
        assert log == ["enter"]
        [facade] = await client.clients_for(["a"])
        async with facade:
            assert log == ["enter"]
    assert log == ["enter", "exit"]

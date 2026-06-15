import hashlib
from datetime import UTC, datetime

import pytest

from haiku.rag.config import PluginSourceConfig
from haiku.rag.ingester.pollers.factory import build_source
from haiku.rag.ingester.pollers.periodic import PeriodicPoller
from haiku.rag.ingester.queue.models import JobOp
from haiku.rag.ingester.sources import plugins as plugins_module
from haiku.rag.ingester.sources import resolve_configured_source
from haiku.rag.ingester.sources.base import (
    FetchResult,
    Source,
    SourceEvent,
    SourceEventKind,
)
from haiku.rag.ingester.sources.plugins import (
    ENTRY_POINT_GROUP,
    load_source_factories,
)


class _MemorySource:
    """A real Source over an in-memory {uri: text} dict. Doubles as its own
    factory: build_source calls it with the plugin kwargs."""

    def __init__(
        self,
        *,
        source_id: str,
        options: dict,
        supported_extensions: list[str] | None,
        max_file_size: int | None,
    ):
        self.source_id = source_id
        self.options = options
        self.supported_extensions = supported_extensions
        self.max_file_size = max_file_size
        self._docs: dict[str, str] = options["docs"]

    def supports(self, uri: str) -> bool:
        return uri in self._docs

    async def head(self, uri: str) -> str | None:
        return "v1"

    async def aclose(self) -> None:
        return None

    async def fetch(self, uri: str) -> FetchResult:
        body = self._docs[uri].encode()
        return FetchResult(
            uri=uri,
            body=body,
            content_type="text/markdown",
            content_hash=hashlib.md5(body).hexdigest(),
            revision="v1",
        )

    async def discover(self, since=None, *, known_uris=None):
        for uri in self._docs:
            yield SourceEvent(
                source_id=self.source_id,
                uri=uri,
                kind=SourceEventKind.UPSERT,
                revision="v1",
                discovered_at=datetime.now(UTC),
            )


class _FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory
        self.loaded = False

    def load(self):
        self.loaded = True
        return self._factory


def _register(monkeypatch, *eps):
    captured: dict = {}

    def fake_entry_points(*, group):
        captured["group"] = group
        return list(eps)

    monkeypatch.setattr(plugins_module, "entry_points", fake_entry_points)
    return captured


def test_load_keys_entry_points_by_name_without_loading(monkeypatch):
    ep = _FakeEntryPoint("memory", _MemorySource)
    captured = _register(monkeypatch, ep)

    discovered = load_source_factories()

    assert captured["group"] == ENTRY_POINT_GROUP
    assert discovered == {"memory": ep}
    assert ep.loaded is False


def test_load_is_empty_when_none_registered(monkeypatch):
    _register(monkeypatch)
    assert load_source_factories() == {}


def _config(**overrides):
    return PluginSourceConfig(
        type="plugin",
        id="mem",
        plugin="memory",
        options={"docs": {"mem://a.md": "hello"}},
        **overrides,
    )


def test_build_source_loads_referenced_plugin_with_kwargs(monkeypatch):
    ep = _FakeEntryPoint("memory", _MemorySource)
    _register(monkeypatch, ep)

    source = build_source(_config(max_file_size=1024), supported_extensions=[".md"])

    assert ep.loaded is True
    assert isinstance(source, _MemorySource)
    assert isinstance(source, Source)
    assert source.source_id == "mem"
    assert source.options == {"docs": {"mem://a.md": "hello"}}
    assert source.supported_extensions == [".md"]
    assert source.max_file_size == 1024


def test_build_source_raises_on_unknown_plugin(monkeypatch):
    _register(monkeypatch, _FakeEntryPoint("memory", _MemorySource))

    cfg = _config().model_copy(update={"plugin": "missing"})
    with pytest.raises(ValueError, match="unknown source plugin 'missing'"):
        build_source(cfg)


def test_build_source_raises_when_plugin_returns_non_source(monkeypatch):
    _register(monkeypatch, _FakeEntryPoint("memory", lambda **kw: object()))

    with pytest.raises(TypeError, match="does not satisfy the Source protocol"):
        build_source(_config())


def test_build_source_does_not_load_unreferenced_plugins(monkeypatch):
    def _explode(**kw):
        raise ImportError("optional dependency missing")

    used = _FakeEntryPoint("memory", _MemorySource)
    unused = _FakeEntryPoint("broken", _explode)
    _register(monkeypatch, used, unused)

    build_source(_config())

    assert unused.loaded is False


@pytest.mark.asyncio
async def test_plugin_source_drives_poller_and_fetch(monkeypatch, jobs, sync):
    _register(monkeypatch, _FakeEntryPoint("memory", _MemorySource))
    cfg = _config()
    source = build_source(cfg)

    poller = PeriodicPoller(source=source, config=cfg, job_repo=jobs, sync_repo=sync)
    assert await poller._sweep_once() is True

    queued = await jobs.list_jobs(source_id="mem")
    assert len(queued) == 1
    assert queued[0].op is JobOp.UPSERT
    assert queued[0].uri == "mem://a.md"

    fetcher = resolve_configured_source("mem://a.md", "mem", [source])
    result = await fetcher.fetch("mem://a.md")
    assert result.body == b"hello"
    assert result.content_type == "text/markdown"

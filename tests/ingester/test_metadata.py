import pytest

from haiku.rag.ingester import metadata as metadata_module
from haiku.rag.ingester.metadata import (
    ENTRY_POINT_GROUP,
    MetadataProvider,
    build_providers,
    load_metadata_providers,
)
from haiku.rag.ingester.sources.base import FetchResult


class _Provider:
    async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
        return {"source": source_id}


class _FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory
        self.loaded = False

    def load(self):
        self.loaded = True
        return self._factory


def test_load_keys_entry_points_by_name_without_loading(monkeypatch):
    ep = _FakeEntryPoint("example-provider", _Provider)
    captured: dict = {}

    def fake_entry_points(*, group):
        captured["group"] = group
        return [ep]

    monkeypatch.setattr(metadata_module, "entry_points", fake_entry_points)

    discovered = load_metadata_providers()

    assert captured["group"] == ENTRY_POINT_GROUP
    assert discovered == {"example-provider": ep}
    assert ep.loaded is False


def test_load_is_empty_when_none_registered(monkeypatch):
    monkeypatch.setattr(metadata_module, "entry_points", lambda *, group: [])
    assert load_metadata_providers() == {}


@pytest.mark.asyncio
async def test_callable_object_satisfies_protocol():
    class Provider:
        async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
            return {"classification": "secret"}

    provider = Provider()
    result = FetchResult(
        uri="u",
        body=b"x",
        content_type="text/plain",
        content_hash="9dd4e461268c8034f5c8564e155c67a6",
    )
    assert isinstance(provider, MetadataProvider)
    assert await provider("src", "u", result) == {"classification": "secret"}


def test_build_providers_instantiates_named_factories():
    providers = build_providers(
        [("docs", "example-provider"), ("wiki", None)],
        {"example-provider": _FakeEntryPoint("example-provider", _Provider)},
    )

    assert set(providers) == {"docs"}
    assert isinstance(providers["docs"], _Provider)


def test_build_providers_skips_sources_without_a_provider():
    discovered = {"example-provider": _FakeEntryPoint("example-provider", _Provider)}
    assert build_providers([("docs", None)], discovered) == {}


def test_build_providers_raises_on_unknown_provider_name():
    discovered = {"example-provider": _FakeEntryPoint("example-provider", _Provider)}
    with pytest.raises(ValueError, match="unknown metadata provider 'missing'"):
        build_providers([("docs", "missing")], discovered)


def test_build_providers_does_not_load_unreferenced_entry_points():
    def _explode():
        raise ImportError("optional dependency missing")

    discovered = {
        "used": _FakeEntryPoint("used", _Provider),
        "unused": _FakeEntryPoint("unused", _explode),
    }

    providers = build_providers([("docs", "used")], discovered)

    assert isinstance(providers["docs"], _Provider)
    assert discovered["unused"].loaded is False

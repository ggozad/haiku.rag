import pytest

from haiku.rag.ingester import metadata as metadata_module
from haiku.rag.ingester.metadata import (
    ENTRY_POINT_GROUP,
    MetadataProvider,
    load_metadata_providers,
)


class _FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


def test_load_keys_factories_by_entry_point_name(monkeypatch):
    def factory():
        return None

    captured: dict = {}

    def fake_entry_points(*, group):
        captured["group"] = group
        return [_FakeEntryPoint("example-provider", factory)]

    monkeypatch.setattr(metadata_module, "entry_points", fake_entry_points)

    providers = load_metadata_providers()

    assert captured["group"] == ENTRY_POINT_GROUP
    assert providers == {"example-provider": factory}


def test_load_is_empty_when_none_registered(monkeypatch):
    monkeypatch.setattr(metadata_module, "entry_points", lambda *, group: [])
    assert load_metadata_providers() == {}


@pytest.mark.asyncio
async def test_callable_object_satisfies_protocol():
    class Provider:
        async def __call__(self, source_id: str, uri: str) -> dict:
            return {"classification": "secret"}

    provider = Provider()
    assert isinstance(provider, MetadataProvider)
    assert await provider("src", "u") == {"classification": "secret"}

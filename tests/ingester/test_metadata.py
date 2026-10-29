import pytest

from haiku.rag.config import MetadataProviderConfig
from haiku.rag.ingester import metadata as metadata_module
from haiku.rag.ingester.metadata import (
    ENTRY_POINT_GROUP,
    MetadataProvider,
    build_providers,
    load_metadata_providers,
)
from haiku.rag.sources.base import FetchResult
from tests.ingester.conftest import FakeEntryPoint


class _Provider:
    async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
        return {"source": source_id}


def test_load_keys_entry_points_by_name_without_loading(monkeypatch):
    ep = FakeEntryPoint("example-provider", _Provider)
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


def test_build_providers_instantiates_named_factories():
    providers = build_providers(
        [("docs", "example-provider"), ("wiki", None)],
        {"example-provider": FakeEntryPoint("example-provider", _Provider)},
    )

    assert set(providers) == {"docs"}
    assert isinstance(providers["docs"], _Provider)
    assert isinstance(providers["docs"], MetadataProvider)


def test_build_providers_raises_on_unknown_provider_name():
    discovered = {"example-provider": FakeEntryPoint("example-provider", _Provider)}
    with pytest.raises(ValueError, match="unknown metadata provider 'missing'"):
        build_providers([("docs", "missing")], discovered)


def test_build_providers_does_not_load_unreferenced_entry_points():
    def _explode():
        raise ImportError("optional dependency missing")

    discovered = {
        "used": FakeEntryPoint("used", _Provider),
        "unused": FakeEntryPoint("unused", _explode),
    }

    providers = build_providers([("docs", "used")], discovered)

    assert isinstance(providers["docs"], _Provider)
    assert discovered["unused"].loaded is False


class _OptionsProvider:
    def __init__(self, **options):
        self.options = options

    async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
        return dict(self.options)


async def test_build_providers_passes_options_to_the_factory():
    discovered = {"tuned": FakeEntryPoint("tuned", _OptionsProvider)}
    providers = build_providers(
        [
            ("docs", MetadataProviderConfig(name="tuned", options={"tier": "a"})),
            ("wiki", "tuned"),
        ],
        discovered,
    )

    result = FetchResult(
        uri="u",
        body=b"x",
        content_type="text/plain",
        content_hash="9dd4e461268c8034f5c8564e155c67a6",
    )
    assert await providers["docs"]("docs", "u", result) == {"tier": "a"}
    assert await providers["wiki"]("wiki", "u", result) == {}


def test_build_providers_names_source_and_provider_when_the_factory_fails():
    def _strict(**options):
        raise TypeError(f"unexpected options {sorted(options)}")

    discovered = {"strict": FakeEntryPoint("strict", _strict)}
    with pytest.raises(
        ValueError,
        match=r"Source 'docs' metadata provider 'strict': unexpected options \['tier'\]",
    ) as raised:
        build_providers(
            [("docs", MetadataProviderConfig(name="strict", options={"tier": "a"}))],
            discovered,
        )
    assert isinstance(raised.value.__cause__, TypeError)


def test_build_providers_names_source_and_provider_when_loading_fails():
    class _BrokenEntryPoint:
        def load(self):
            raise ImportError("optional dependency missing")

    with pytest.raises(
        ValueError,
        match="Source 'docs' metadata provider 'broken': optional dependency missing",
    ):
        build_providers([("docs", "broken")], {"broken": _BrokenEntryPoint()})

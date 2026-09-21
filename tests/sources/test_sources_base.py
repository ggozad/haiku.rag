import subprocess
import sys
from datetime import UTC, datetime

from haiku.rag.sources.base import (
    FetchResult,
    Source,
    SourceEvent,
    SourceEventKind,
)


def test_source_event_kind_values():
    assert SourceEventKind.UPSERT.value == "upsert"
    assert SourceEventKind.DELETE.value == "delete"
    assert SourceEventKind.UNCHANGED.value == "unchanged"


def test_source_event_round_trip():
    event = SourceEvent(
        source_id="fs:/tmp/docs",
        uri="file:///tmp/docs/a.md",
        kind=SourceEventKind.UPSERT,
        revision="123456",
        discovered_at=datetime(2026, 5, 20, 12, 0, 0, tzinfo=UTC),
    )
    raw = event.model_dump_json()
    again = SourceEvent.model_validate_json(raw)
    assert again == event


def test_fetch_result_round_trip():
    result = FetchResult(
        uri="file:///tmp/docs/a.md",
        body=b"hello",
        content_type="text/markdown",
        content_hash="abcd1234",
        revision="123456",
        extra_metadata={"source": "fs"},
    )
    raw = result.model_dump_json()
    again = FetchResult.model_validate_json(raw)
    assert again == result


def test_fetch_result_defaults_extra_metadata_to_empty():
    result = FetchResult(
        uri="file:///tmp/docs/a.md",
        body=b"x",
        content_type="text/markdown",
        content_hash="x",
        revision=None,
    )
    assert result.extra_metadata == {}


def test_source_protocol_runtime_checkable():
    class Dummy:
        source_id = "dummy"

        def supports(self, uri: str) -> bool:
            return True

        async def aclose(self) -> None:
            pass

        async def head(self, uri: str):
            return None

        async def fetch(self, uri: str):
            raise NotImplementedError

        def discover(self, since=None):
            raise NotImplementedError

    assert isinstance(Dummy(), Source)


def test_importing_registry_does_not_load_heavy_dependencies():
    """Same invariant as test_cli.py's test_importing_cli_does_not_load_heavy_dependencies:
    registry only needs UnsupportedSourceError, not the client package it lives under."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import haiku.rag.sources.registry, sys; "
            "loaded = {'lancedb', 'pyarrow', 'pydantic_ai'} & sys.modules.keys(); "
            "assert not loaded, loaded",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

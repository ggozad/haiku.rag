"""Regression: every source's revision must round-trip from FetchResult →
document.metadata → sync_state, so the next sweep can recognise the file
as unchanged. Catches the FS-specific bug where revision was lost in the
pipeline and every periodic sweep re-enqueued every file forever.
"""

from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.ingester.sources.base import SourceEventKind
from haiku.rag.ingester.sources.fs import FSSource


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent / "cassettes" / "test_revision_round_trip")


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_fs_ingest_writes_source_revision_to_metadata(temp_db_path, tmp_path):
    file_path = tmp_path / "doc.md"
    file_path.write_text("hello")
    expected_revision = str(file_path.stat().st_mtime_ns)

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document_from_source(file_path)

    assert doc.metadata["source_revision"] == expected_revision


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_fs_second_sweep_emits_unchanged_after_ingest(temp_db_path, tmp_path):
    """The full round-trip: ingest a file, build a sync_state-shaped snapshot
    from document.metadata, hand it to FSSource.discover() — must see
    UNCHANGED, not UPSERT. This is exactly what the periodic poller does."""
    file_path = tmp_path / "doc.md"
    file_path.write_text("hello")

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document_from_source(file_path)

    assert doc.uri is not None
    snapshot = {doc.uri: doc.metadata["source_revision"]}

    src = FSSource(root=tmp_path)
    kinds: list[SourceEventKind] = []
    async for event in src.discover(since=snapshot):
        kinds.append(event.kind)

    assert kinds == [SourceEventKind.UNCHANGED]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_fs_second_sweep_emits_upsert_when_file_changes(temp_db_path, tmp_path):
    """Counterpart to the unchanged test: a file modified after ingest still
    triggers UPSERT. Ensures the round-trip doesn't accidentally over-skip."""
    file_path = tmp_path / "doc.md"
    file_path.write_text("hello")

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document_from_source(file_path)

    # Bump mtime by writing different content; snapshot still has the old revision.
    assert doc.uri is not None
    snapshot = {doc.uri: doc.metadata["source_revision"]}
    file_path.write_text("hello world")
    # st_mtime_ns has nanosecond resolution; a write_text always advances it
    # on any sane filesystem, but assert anyway to make the intent explicit.
    assert str(file_path.stat().st_mtime_ns) != doc.metadata["source_revision"]

    src = FSSource(root=tmp_path)
    kinds: list[SourceEventKind] = []
    async for event in src.discover(since=snapshot):
        kinds.append(event.kind)

    assert kinds == [SourceEventKind.UPSERT]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_fs_head_short_circuit_skips_fetch_for_unchanged_revision(
    temp_db_path, tmp_path, monkeypatch
):
    """The one-shot (add-src) path: if the existing doc has source_revision
    and HEAD reports the same value, fetch must not run."""
    file_path = tmp_path / "doc.md"
    file_path.write_text("hello")

    async with HaikuRAG(temp_db_path, create=True) as client:
        first = await client.create_document_from_source(file_path)

        fetch_calls: list[str] = []
        original_fetch = FSSource.fetch

        async def _track_fetch(self: FSSource, uri: str):  # type: ignore[no-untyped-def]
            fetch_calls.append(uri)
            return await original_fetch(self, uri)

        monkeypatch.setattr(FSSource, "fetch", _track_fetch)

        second = await client.create_document_from_source(file_path)

    assert second.id == first.id
    assert fetch_calls == []

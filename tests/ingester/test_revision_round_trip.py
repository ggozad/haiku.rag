"""Regression: every source's revision must round-trip from FetchResult →
document.metadata → sync_state, so the next sweep can recognise the file
as unchanged. Catches the FS-specific bug where revision was lost in the
pipeline and every periodic sweep re-enqueued every file forever.
"""

import hashlib
from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.ingester.sources.base import FetchResult, SourceEventKind
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


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_provider_backed_unchanged_revision_keeps_head_short_circuit(
    temp_db_path, tmp_path, monkeypatch
):
    """Provider-backed sources keep the cheap HEAD path when the revision is
    unchanged. Existing provider metadata persists until the content changes."""
    file_path = tmp_path / "doc.md"
    file_path.write_text("hello")
    content_hash = hashlib.md5(b"hello", usedforsecurity=False).hexdigest()
    revision = str(file_path.stat().st_mtime_ns)
    stored_uri = file_path.absolute().as_uri()

    seen: dict = {}

    class Provider:
        async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
            seen["called"] = True
            raise AssertionError("provider must not run on unchanged HEAD")

    async with HaikuRAG(temp_db_path, create=True) as client:
        first = await client.create_document(
            "hello",
            uri=stored_uri,
            metadata={
                "md5": content_hash,
                "source_revision": revision,
                "content_type": "text/markdown",
                "classification": "secret",
            },
        )

        fetch_calls: list[str] = []
        original_fetch = FSSource.fetch

        async def _track_fetch(self: FSSource, uri: str):  # type: ignore[no-untyped-def]
            fetch_calls.append(uri)
            return await original_fetch(self, uri)

        monkeypatch.setattr(FSSource, "fetch", _track_fetch)

        second = await client.create_document_from_source(
            file_path, metadata_provider=Provider()
        )

    assert second.id == first.id
    assert fetch_calls == []
    assert seen == {}
    assert second.metadata["classification"] == "secret"
    assert second.metadata["md5"] == first.metadata["md5"]
    assert second.metadata["source_revision"] == first.metadata["source_revision"]
    assert second.metadata["content_type"] == first.metadata["content_type"]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_metadata_provider_applies_to_fresh_create(temp_db_path, tmp_path):
    """Fresh ingests pass FetchResult to the provider, merge provider metadata,
    and still keep source-owned metadata authoritative."""
    file_path = tmp_path / "doc.md"
    file_path.write_text("hello")
    content_hash = hashlib.md5(b"hello", usedforsecurity=False).hexdigest()
    revision = str(file_path.stat().st_mtime_ns)

    seen: dict = {}

    class Provider:
        async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
            seen["source_id"] = source_id
            seen["uri"] = uri
            seen["body"] = result.body
            seen["disk_path"] = result.disk_path
            seen["content_type"] = result.content_type
            return {
                "classification": "secret",
                "md5": "spoof",
                "source_revision": "spoof",
                "content_type": "text/spoof",
            }

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document_from_source(
            file_path, metadata_provider=Provider()
        )

    assert seen["body"] == b"hello"
    assert seen["disk_path"] == file_path
    assert seen["uri"] == str(file_path)
    assert seen["source_id"].startswith("fs:")
    assert doc.metadata["classification"] == "secret"
    assert doc.metadata["md5"] == content_hash
    assert doc.metadata["source_revision"] == revision
    assert doc.metadata["content_type"] == seen["content_type"]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_provider_mutating_fetch_result_cannot_corrupt_source_metadata(
    temp_db_path, tmp_path
):
    """A provider only contributes metadata via its returned dict. Mutating the
    FetchResult it receives must not reach the md5 short-circuit or source
    metadata, so source-owned keys stay authoritative."""
    file_path = tmp_path / "doc.md"
    file_path.write_text("hello")
    content_hash = hashlib.md5(b"hello", usedforsecurity=False).hexdigest()

    class Provider:
        async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
            result.content_hash = "spoof"
            result.extra_metadata["md5"] = "spoof"
            result.extra_metadata["injected"] = "x"
            return {"classification": "secret"}

    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document_from_source(
            file_path, metadata_provider=Provider()
        )

    assert doc.metadata["classification"] == "secret"
    assert doc.metadata["md5"] == content_hash
    assert "injected" not in doc.metadata


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_directory_ingest_threads_configured_source_to_provider(
    temp_db_path, tmp_path
):
    """Directory ingestion with a configured source passes that source's id and
    fetch context to each child, so the provider sees the configured source id
    rather than an ad-hoc fs: identity."""
    (tmp_path / "doc.md").write_text("hello")

    seen_source_ids: list[str] = []

    class Provider:
        async def __call__(self, source_id: str, uri: str, result: FetchResult) -> dict:
            seen_source_ids.append(source_id)
            return {"collection": source_id}

    source = FSSource(root=tmp_path, source_id="docs")

    async with HaikuRAG(temp_db_path, create=True) as client:
        docs = await client.create_document_from_source(
            tmp_path,
            sources=[source],
            source_id="docs",
            metadata_provider=Provider(),
        )

    assert isinstance(docs, list)
    assert seen_source_ids == ["docs"]
    assert docs[0].metadata["collection"] == "docs"

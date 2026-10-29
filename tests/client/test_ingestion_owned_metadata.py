import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG, documents
from haiku.rag.sources.fs import FSSource
from haiku.rag.store.models.document import Document

FRESHNESS = ("content_type", "md5", "source_revision")
FORGED = {"content_type": "x/forged", "md5": "forged", "source_revision": "forged"}


@pytest.fixture
def source_file() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "tracked.txt"
        path.write_text("Content ingested from a source file.")
        yield path


@pytest.fixture
def calls(monkeypatch) -> dict[str, int]:
    """Counts source fetches and conversions."""
    counts = {"fetch": 0, "convert": 0}
    fetch, convert = FSSource.fetch, documents.convert

    async def counting_fetch(self, uri):
        counts["fetch"] += 1
        return await fetch(self, uri)

    async def counting_convert(*args, **kwargs):
        counts["convert"] += 1
        return await convert(*args, **kwargs)

    monkeypatch.setattr(FSSource, "fetch", counting_fetch)
    monkeypatch.setattr(documents, "convert", counting_convert)
    return counts


async def _ingest(client: HaikuRAG, path: Path, metadata: dict | None = None):
    doc = await client.create_document_from_source(path, metadata=metadata)
    assert isinstance(doc, Document)
    return doc


def _freshness(doc: Document) -> dict:
    return {k: doc.metadata[k] for k in FRESHNESS}


@pytest.mark.vcr()
async def test_metadata_update_keeps_the_keys_ingestion_skips_on(
    temp_db_path, source_file, calls
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        ingested = await _ingest(client, source_file)
        assert ingested.id is not None

        updated = await client.update_document(
            ingested.id, metadata={"identifier": "ACC-1", **FORGED}
        )

        assert updated.metadata == {**ingested.metadata, "identifier": "ACC-1"}

        calls.update(fetch=0, convert=0)
        again = await _ingest(client, source_file)

        assert calls == {"fetch": 0, "convert": 0}
        assert again.metadata["identifier"] == "ACC-1"


@pytest.mark.vcr()
async def test_new_revision_with_the_same_bytes_skips_after_a_metadata_update(
    temp_db_path, source_file, calls
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        ingested = await _ingest(client, source_file)
        assert ingested.id is not None
        await client.update_document(ingested.id, metadata={"identifier": "ACC-1"})

        stat = source_file.stat()
        os.utime(source_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        calls.update(fetch=0, convert=0)
        again = await _ingest(client, source_file)

        assert calls == {"fetch": 1, "convert": 0}
        assert again.metadata["identifier"] == "ACC-1"
        assert again.metadata["source_revision"] != ingested.metadata["source_revision"]


@pytest.mark.vcr()
async def test_content_and_metadata_update_keeps_the_ingestion_keys(
    temp_db_path, source_file
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        ingested = await _ingest(client, source_file)
        assert ingested.id is not None

        updated = await client.update_document(
            ingested.id, content="A local edit.", metadata={"identifier": "ACC-1"}
        )

        assert updated.metadata == {**ingested.metadata, "identifier": "ACC-1"}


@pytest.mark.vcr()
async def test_source_ingest_drops_caller_freshness_keys(temp_db_path, source_file):
    async with HaikuRAG(temp_db_path, create=True) as client:
        ingested = await _ingest(client, source_file, {"team": "docs", **FORGED})
        unchanged = await _ingest(client, source_file, {"team": "docs", **FORGED})

        assert "forged" not in _freshness(ingested).values()
        assert _freshness(unchanged) == _freshness(ingested)
        assert unchanged.metadata["team"] == "docs"


@pytest.mark.vcr()
async def test_source_ingest_refetches_after_create_document_over_its_uri(
    temp_db_path, source_file, calls
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        ingested = await _ingest(client, source_file)
        overwritten = await client.create_document(
            "Text a caller wrote over the source's document.", uri=ingested.uri
        )
        assert overwritten.id == ingested.id
        assert not set(FRESHNESS) & overwritten.metadata.keys()

        calls.update(fetch=0, convert=0)
        again = await _ingest(client, source_file)

        assert calls == {"fetch": 1, "convert": 1}
        assert again.content == ingested.content

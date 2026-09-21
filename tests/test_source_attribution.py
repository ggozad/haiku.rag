import asyncio
import logging
import os
import tempfile
from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG, documents
from haiku.rag.client.documents import DocumentImport
from haiku.rag.sources.fs import FSSource
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository

from .conftest import capture_logs

DOCUMENTS_LOGGER = logging.getLogger("haiku.rag.client.documents")


def fs_source(root: Path, source_id: str = "fs:test") -> FSSource:
    return FSSource(root=root, source_id=source_id)


def _docling_document(text: str) -> DoclingDocument:
    doc = DoclingDocument(name="attribution")
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc


@pytest.mark.vcr()
async def test_full_ingest_records_source_id(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "attributed.txt"
            path.write_text("Content ingested through a configured source.")
            source = fs_source(root)

            doc = await client.create_document_from_source(
                path, sources=[source], source_id=source.source_id
            )

            assert isinstance(doc, Document)
            assert doc.metadata["source_id"] == source.source_id


@pytest.mark.vcr()
async def test_adhoc_ingest_records_no_source_id(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unattributed.txt"
            path.write_text("Content added by hand.")

            doc = await client.create_document_from_source(path)

            assert isinstance(doc, Document)
            assert "source_id" not in doc.metadata


@pytest.mark.vcr()
async def test_revision_short_circuit_records_source_id(temp_db_path):
    """An unchanged file is attributed on a HEAD alone."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "adopted.txt"
            path.write_text("Content added by hand, later swept.")
            first = await client.create_document_from_source(path)
            assert isinstance(first, Document)
            assert "source_id" not in first.metadata

            source = fs_source(root)
            second = await client.create_document_from_source(
                path, sources=[source], source_id=source.source_id
            )

            assert isinstance(second, Document)
            assert second.id == first.id
            assert second.metadata["source_id"] == source.source_id
            assert second.metadata["md5"] == first.metadata["md5"]


@pytest.mark.vcr()
async def test_md5_short_circuit_records_source_id(temp_db_path):
    """Revision rolled, bytes unchanged: attribution is written anyway."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "touched.txt"
            path.write_text("Bytes that do not change.")
            first = await client.create_document_from_source(path)
            assert isinstance(first, Document)

            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            source = fs_source(root)
            second = await client.create_document_from_source(
                path, sources=[source], source_id=source.source_id
            )

            assert isinstance(second, Document)
            assert second.id == first.id
            assert second.metadata["source_id"] == source.source_id
            assert (
                second.metadata["source_revision"] != first.metadata["source_revision"]
            )


@pytest.mark.vcr()
async def test_provider_cannot_overwrite_source_id(temp_db_path):
    async def provider(source_id: str, uri: str, result) -> dict:
        del source_id, uri, result  # interface-required, unused by this stub
        return {"source_id": "forged", "team": "docs"}

    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "provided.txt"
            path.write_text("Content whose provider lies about its source.")
            source = fs_source(root)

            doc = await client.create_document_from_source(
                path,
                sources=[source],
                source_id=source.source_id,
                metadata_provider=provider,
            )

            assert isinstance(doc, Document)
            assert doc.metadata["source_id"] == source.source_id
            assert doc.metadata["team"] == "docs"


@pytest.mark.vcr()
async def test_adhoc_reingest_preserves_source_id(temp_db_path):
    """Rewriting an owned document by hand does not detach it from its source."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "owned.txt"
            path.write_text("First revision of an owned document.")
            source = fs_source(root)
            first = await client.create_document_from_source(
                path, sources=[source], source_id=source.source_id
            )
            assert isinstance(first, Document)

            path.write_text("Second revision, written by hand this time.")
            second = await client.create_document_from_source(path)

            assert isinstance(second, Document)
            assert second.id == first.id
            assert second.metadata["source_id"] == source.source_id


@pytest.mark.vcr()
async def test_caller_metadata_cannot_set_source_id(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "claimed.txt"
            path.write_text("Content whose caller claims a source.")

            doc = await client.create_document_from_source(
                path, metadata={"source_id": "fs:forged", "team": "docs"}
            )

            assert isinstance(doc, Document)
            assert "source_id" not in doc.metadata
            assert doc.metadata["team"] == "docs"


@pytest.mark.vcr()
async def test_create_document_cannot_set_source_id(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(
            "Content a caller claims a source for.",
            uri="test://claimed",
            metadata={"source_id": "fs:forged", "team": "docs"},
        )

        assert "source_id" not in doc.metadata
        assert doc.metadata["team"] == "docs"


async def test_import_cannot_set_source_id(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        vector_dim = client.store.embedder.vector_dim
        docling_document = _docling_document("Imported content.")

        imported = await client.import_document(
            docling_document,
            [Chunk(content="Imported content.", embedding=[0.1] * vector_dim)],
            uri="test://imported",
            metadata={"source_id": "fs:forged", "team": "docs"},
        )
        batch = await client.import_documents(
            [
                DocumentImport(
                    docling_document=_docling_document("Batch content."),
                    chunks=[
                        Chunk(content="Batch content.", embedding=[0.1] * vector_dim)
                    ],
                    uri="test://batched",
                    metadata={"source_id": "fs:forged"},
                )
            ]
        )

        assert "source_id" not in imported.metadata
        assert imported.metadata["team"] == "docs"
        assert "source_id" not in batch[0].metadata


@pytest.mark.vcr()
async def test_update_document_cannot_drop_or_set_source_id(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "updated.txt"
            path.write_text("Content an owner ingested.")
            source = fs_source(root)
            doc = await client.create_document_from_source(
                path, sources=[source], source_id=source.source_id
            )
            assert isinstance(doc, Document)
            assert doc.id is not None

            await client.update_document(
                doc.id, metadata={"team": "docs", "source_id": "fs:forged"}
            )

            refreshed = await client.get_document_by_id(doc.id)
            assert refreshed is not None
            assert refreshed.metadata["source_id"] == source.source_id
            assert refreshed.metadata["team"] == "docs"


@pytest.mark.vcr()
async def test_concurrent_adhoc_ingest_keeps_source_id(temp_db_path):
    """The locked re-check keeps the source of the row it replaces."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "raced.txt"
            path.write_text("Content two ingestions write at once.")
            source = fs_source(root)

            results = await asyncio.gather(
                client.create_document_from_source(
                    path, sources=[source], source_id=source.source_id
                ),
                client.create_document_from_source(path),
            )

            assert results[0].id == results[1].id
            assert await client.count_documents() == 1
            surviving = await client.get_document_by_uri(path.as_uri())
            assert surviving is not None
            assert surviving.metadata["source_id"] == source.source_id


@pytest.mark.vcr()
async def test_failed_ingestion_reports_no_ownership_change(temp_db_path, monkeypatch):
    async def failing_fetch(uri: str):
        raise OSError(f"unreadable: {uri}")

    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "unreadable.txt"
            path.write_text("Content the second source cannot fetch.")
            outer = fs_source(root, "fs:outer")
            inner = fs_source(root, "fs:inner")
            first = await client.create_document_from_source(
                path, sources=[outer], source_id=outer.source_id
            )
            assert isinstance(first, Document)

            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            monkeypatch.setattr(inner, "fetch", failing_fetch)

            with capture_logs(DOCUMENTS_LOGGER, logging.WARNING) as records:
                with pytest.raises(OSError, match="unreadable"):
                    await client.create_document_from_source(
                        path, sources=[inner], source_id=inner.source_id
                    )

            assert records == []
            unchanged = await client.get_document_by_uri(path.as_uri())
            assert unchanged is not None
            assert unchanged.metadata["source_id"] == outer.source_id


@pytest.mark.vcr()
async def test_concurrent_first_ingest_by_two_sources_warns(temp_db_path):
    """A transfer resolved only under the lock is still reported."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "raced-first.txt"
            path.write_text("Content two sources create at once.")
            outer = fs_source(root, "fs:outer")
            inner = fs_source(root, "fs:inner")

            with capture_logs(DOCUMENTS_LOGGER, logging.WARNING) as records:
                results = await asyncio.gather(
                    client.create_document_from_source(
                        path, sources=[outer], source_id=outer.source_id
                    ),
                    client.create_document_from_source(
                        path, sources=[inner], source_id=inner.source_id
                    ),
                )

            assert results[0].id == results[1].id
            assert await client.count_documents() == 1
            messages = [record.getMessage() for record in records]
            assert [m for m in messages if "fs:outer" in m and "fs:inner" in m]
            surviving = await client.get_document_by_uri(path.as_uri())
            assert surviving is not None
            assert surviving.metadata["source_id"] in {
                outer.source_id,
                inner.source_id,
            }


@pytest.mark.vcr()
async def test_rolled_back_transfer_reports_no_ownership_change(
    temp_db_path, monkeypatch
):
    """A transfer whose write fails reports nothing."""

    async def failing_replace(self, document_id, chunks):
        del self, document_id, chunks  # interface-required, unused by this stub
        raise RuntimeError("chunk write failed")

    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "rolled-back.txt"
            path.write_text("First revision under the outer source.")
            outer = fs_source(root, "fs:outer")
            inner = fs_source(root, "fs:inner")
            first = await client.create_document_from_source(
                path, sources=[outer], source_id=outer.source_id
            )
            assert isinstance(first, Document)

            path.write_text("Second revision, written under the inner source.")
            monkeypatch.setattr(
                ChunkRepository, "replace_for_document", failing_replace
            )

            with capture_logs(DOCUMENTS_LOGGER, logging.WARNING) as records:
                with pytest.raises(RuntimeError, match="chunk write failed"):
                    await client.create_document_from_source(
                        path, sources=[inner], source_id=inner.source_id
                    )

            assert records == []
            unchanged = await client.get_document_by_uri(path.as_uri())
            assert unchanged is not None
            assert unchanged.metadata["source_id"] == outer.source_id


@pytest.mark.vcr()
async def test_second_source_takes_ownership_and_warns(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "contested.txt"
            path.write_text("Content two sources both cover.")
            outer = fs_source(root, "fs:outer")
            inner = fs_source(root, "fs:inner")
            first = await client.create_document_from_source(
                path, sources=[outer], source_id=outer.source_id
            )
            assert isinstance(first, Document)

            with capture_logs(DOCUMENTS_LOGGER, logging.WARNING) as records:
                second = await client.create_document_from_source(
                    path, sources=[inner], source_id=inner.source_id
                )

            assert isinstance(second, Document)
            assert second.id == first.id
            assert second.metadata["source_id"] == inner.source_id
            messages = [record.getMessage() for record in records]
            assert any("fs:outer" in m and "fs:inner" in m for m in messages)


async def test_set_document_source_attributes_an_unattributed_document(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        vector_dim = client.store.embedder.vector_dim
        doc = await client.import_document(
            _docling_document("Content the queue knows a source for."),
            [Chunk(content="Content.", embedding=[0.1] * vector_dim)],
            uri="test://legacy",
        )
        assert doc.id is not None
        assert "source_id" not in doc.metadata

        (updated,) = await client.set_document_source([doc.id], "fs:handbook")

        assert updated.metadata["source_id"] == "fs:handbook"
        refetched = await client.get_document_by_id(doc.id)
        assert refetched is not None
        assert refetched.metadata["source_id"] == "fs:handbook"


async def test_set_document_source_batches_id_lookups(temp_db_path, monkeypatch):
    """A whole-corpus migration resolves ids in bounded queries, not one each."""
    monkeypatch.setattr(documents, "ID_LOOKUP_BATCH", 2)
    async with HaikuRAG(temp_db_path, create=True) as client:
        vector_dim = client.store.embedder.vector_dim
        ids = []
        for n in range(5):
            doc = await client.import_document(
                _docling_document(f"Content {n}."),
                [Chunk(content=f"Content {n}.", embedding=[0.1] * vector_dim)],
                uri=f"test://batched-{n}",
            )
            assert doc.id is not None
            ids.append(doc.id)

        queries = []
        original = DocumentRepository.list_all

        async def counting_list_all(self, *args, **kwargs):
            queries.append(kwargs.get("filter"))
            return await original(self, *args, **kwargs)

        monkeypatch.setattr(DocumentRepository, "list_all", counting_list_all)
        updated = await client.set_document_source(ids, "fs:handbook")

        assert [doc.metadata["source_id"] for doc in updated] == ["fs:handbook"] * 5
        assert {doc.id for doc in updated} == set(ids)
        assert len(queries) == 3


async def test_set_document_source_rejects_unknown_id(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(ValueError, match="not found"):
            await client.set_document_source(["no-such-document"], "fs:handbook")

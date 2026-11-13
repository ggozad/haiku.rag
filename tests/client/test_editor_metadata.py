import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from haiku.rag import embeddings
from haiku.rag.client import HaikuRAG, RebuildMode, documents
from haiku.rag.client import rebuild as rebuild_module
from haiku.rag.client.documents import (
    DocumentImport,
    _reconcile_pdf_attachments,
    parent_uri_filter,
)
from haiku.rag.client.exceptions import SourceOwnedError
from haiku.rag.config import get_config
from haiku.rag.sources.fs import FSSource
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document
from tests.conftest import build_pdf, writing
from tests.documents import docling_doc
from tests.locks import ObservedLock, assert_waiting_for_lock

EDITED = {"reviewer": "ana", "tags": ["draft", "legal"]}


@pytest.fixture
def source_file() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "tracked.txt"
        path.write_text("Content ingested from a source file.")
        yield path


async def _import(client: HaikuRAG, uri: str, **kwargs) -> Document:
    dim = get_config().embeddings.model.vector_dim
    doc = await client.import_document(
        docling_doc("d", "body"),
        [Chunk(content="body", embedding=[0.1] * dim, order=0)],
        uri=uri,
        **kwargs,
    )
    assert doc.id is not None
    return doc


async def _ingest(client: HaikuRAG, path: Path, **kwargs) -> Document:
    doc = await client.create_document_from_source(path, **kwargs)
    assert isinstance(doc, Document) and doc.id is not None
    return doc


async def test_edit_sets_title_and_editor_metadata_and_keeps_other_keys(
    temp_db_path,
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a", title="Old", metadata={"team": "x"})
        await client.set_document_source([doc.id], "src")

        edited = await client.edit_document(doc.id, title="New", user_metadata=EDITED)
        stored = await client.get_document_by_id(doc.id)

    assert edited.title == "New"
    assert stored is not None
    assert stored.title == "New"
    assert stored.metadata == {"team": "x", "source_id": "src", "user": EDITED}


async def test_edit_leaves_an_omitted_field_unchanged(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a", title="Old")

        await client.edit_document(doc.id, user_metadata=EDITED)
        await client.edit_document(doc.id, title="New")
        stored = await client.get_document_by_id(doc.id)

    assert stored is not None
    assert stored.title == "New"
    assert stored.metadata["user"] == EDITED


async def test_edit_replaces_the_whole_editor_metadata(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a")

        await client.edit_document(doc.id, user_metadata=EDITED)
        await client.edit_document(doc.id, user_metadata={})
        stored = await client.get_document_by_id(doc.id)

    assert stored is not None
    assert stored.metadata["user"] == {}


async def test_edit_of_a_missing_document_raises(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        with pytest.raises(ValueError, match="not found"):
            await client.edit_document("missing", title="New")


async def test_edit_keeps_metadata_written_while_it_waited(temp_db_path, monkeypatch):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a")

        lock = ObservedLock()
        monkeypatch.setattr(client.store, "_write_lock", lock)
        async with lock:
            task = asyncio.create_task(
                client.edit_document(doc.id, user_metadata=EDITED)
            )
            await assert_waiting_for_lock(task, lock)
            doc.metadata = {"md5": "abc"}
            await writing(client).document_repository.update_meta(doc)
        await task
        stored = await client.get_document_by_id(doc.id)

    assert stored is not None
    assert stored.metadata == {"md5": "abc", "user": EDITED}


async def test_import_drops_caller_editor_metadata(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a", metadata={"user": EDITED, "team": "x"})

    assert doc.metadata == {"team": "x"}


async def test_batch_import_drops_caller_editor_metadata(temp_db_path):
    dim = get_config().embeddings.model.vector_dim
    async with HaikuRAG(temp_db_path, create=True) as client:
        [doc] = await client.import_documents(
            [
                DocumentImport(
                    docling_doc("d", "body"),
                    [Chunk(content="body", embedding=[0.1] * dim, order=0)],
                    uri="mem://a",
                    metadata={"user": EDITED, "team": "x"},
                )
            ]
        )

    assert doc.metadata == {"team": "x"}


async def test_import_over_a_stored_uri_keeps_editor_metadata(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a", title="Old")
        await client.edit_document(doc.id, title="Edited", user_metadata=EDITED)

        again = await _import(
            client, "mem://a", title="Imported", metadata={"user": {"forged": 1}}
        )

    assert again.id == doc.id
    assert again.title == "Imported"
    assert again.metadata == {"user": EDITED}


@pytest.mark.vcr()
async def test_create_document_over_a_stored_uri_keeps_editor_metadata(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a")
        await client.edit_document(doc.id, title="Edited", user_metadata=EDITED)

        again = await client.create_document(
            "Text written over the document.",
            uri="mem://a",
            title="Created",
            metadata={"user": {"forged": 1}},
        )

    assert again.id == doc.id
    assert again.title == "Created"
    assert again.metadata == {"user": EDITED}


async def test_metadata_update_keeps_editor_metadata(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a")
        await client.edit_document(doc.id, user_metadata=EDITED)

        updated = await client.update_document(
            doc.id, metadata={"team": "x", "user": {"forged": 1}}
        )

    assert updated.metadata == {"team": "x", "user": EDITED}


@pytest.mark.vcr()
async def test_source_reingest_keeps_edited_title_and_editor_metadata(
    temp_db_path, source_file
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _ingest(client, source_file)
        await client.edit_document(doc.id, title="Edited", user_metadata=EDITED)

        source_file.write_text("Changed content from the source file.")
        again = await _ingest(client, source_file, metadata={"user": {"forged": 1}})

    assert again.id == doc.id
    assert again.content.startswith("Changed content")
    assert again.title == "Edited"
    assert again.metadata["user"] == EDITED


@pytest.mark.vcr()
async def test_source_reingest_keeps_an_edit_made_during_conversion(
    temp_db_path, source_file, monkeypatch
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _ingest(client, source_file)
        convert = documents.convert

        async def editing_convert(*args, **kwargs):
            await client.edit_document(doc.id, title="Edited", user_metadata=EDITED)
            return await convert(*args, **kwargs)

        monkeypatch.setattr(documents, "convert", editing_convert)
        source_file.write_text("Changed content from the source file.")
        again = await _ingest(client, source_file)

    assert again.title == "Edited"
    assert again.metadata["user"] == EDITED


@pytest.mark.vcr()
async def test_source_reingest_of_a_document_deleted_during_conversion_writes_nothing(
    temp_db_path, source_file, monkeypatch
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _ingest(client, source_file)
        convert = documents.convert

        async def deleting_convert(*args, **kwargs):
            await client.delete_document(doc.id)
            return await convert(*args, **kwargs)

        monkeypatch.setattr(documents, "convert", deleting_convert)
        source_file.write_text("Changed content from the source file.")
        with pytest.raises(ValueError, match="not found"):
            await _ingest(client, source_file)

        assert await client.count_documents() == 0
        assert doc.id is not None
        assert await writing(client).chunk_repository.get_by_document_id(doc.id) == []


@pytest.mark.vcr()
async def test_source_ingest_drops_provider_and_source_editor_metadata(
    temp_db_path, source_file, monkeypatch
):
    fetch = FSSource.fetch

    async def fetch_with_user(self, uri):
        result = await fetch(self, uri)
        return result.model_copy(update={"extra_metadata": {"user": "from source"}})

    async def provider(source_id, uri, result):
        return {"user": {"from": "provider"}, "team": "x"}

    monkeypatch.setattr(FSSource, "fetch", fetch_with_user)
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _ingest(client, source_file, metadata_provider=provider)
        await client.edit_document(doc.id, user_metadata=EDITED)
        source_file.write_text("Changed content from the source file.")
        again = await _ingest(client, source_file, metadata_provider=provider)

    assert "user" not in doc.metadata
    assert doc.metadata["team"] == "x"
    assert again.metadata["user"] == EDITED


@pytest.mark.vcr()
async def test_full_rebuild_keeps_an_edit_made_during_conversion(
    temp_db_path, source_file, monkeypatch
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _ingest(client, source_file)
        await client.edit_document(doc.id, user_metadata={"before": "rebuild"})
        convert = documents.convert

        async def editing_convert(*args, **kwargs):
            await client.edit_document(doc.id, title="Edited", user_metadata=EDITED)
            return await convert(*args, **kwargs)

        monkeypatch.setattr(documents, "convert", editing_convert)
        rebuilt = [i async for i in client.rebuild_database(mode=RebuildMode.FULL)]
        stored = await client.get_document_by_id(doc.id)

    assert rebuilt == [doc.id]
    assert stored is not None
    assert stored.title == "Edited"
    assert stored.metadata["user"] == EDITED


@pytest.mark.vcr()
@pytest.mark.parametrize("mode", [RebuildMode.FULL, RebuildMode.RECHUNK])
async def test_rebuild_from_stored_content_keeps_an_edit_made_during_embedding(
    temp_db_path, monkeypatch, mode
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a", title="Old")
        embed_chunks = embeddings.embed_chunks

        async def editing_embed_chunks(*args, **kwargs):
            await client.edit_document(doc.id, title="Edited", user_metadata=EDITED)
            return await embed_chunks(*args, **kwargs)

        monkeypatch.setattr(embeddings, "embed_chunks", editing_embed_chunks)
        rebuilt = [i async for i in client.rebuild_database(mode=mode)]
        stored = await client.get_document_by_id(doc.id)

    assert rebuilt == [doc.id]
    assert stored is not None
    assert stored.title == "Edited"
    assert stored.metadata["user"] == EDITED


async def test_title_only_rebuild_keeps_an_edit_made_while_it_titled(
    temp_db_path, monkeypatch
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a")
        untouched = await _import(client, "mem://b")
        assert doc.title is None and untouched.title is None

        async def editing_generate_title(config, document):
            if document.id == doc.id:
                await client.edit_document(doc.id, title="Edited", user_metadata=EDITED)
            return "Generated"

        monkeypatch.setattr(rebuild_module, "generate_title", editing_generate_title)
        [i async for i in client.rebuild_database(mode=RebuildMode.TITLE_ONLY)]
        stored = await client.get_document_by_id(doc.id)
        other = await client.get_document_by_id(untouched.id)

    assert stored is not None and other is not None
    assert stored.title == "Edited"
    assert stored.metadata["user"] == EDITED
    assert other.title == "Generated"


@pytest.mark.vcr()
async def test_forced_attachment_reingest_keeps_editor_metadata(temp_db_path):
    pdf_bytes = build_pdf([("notes.txt", b"plain text payload")])
    parent_uri = "file:///fixtures/parent.pdf"
    async with HaikuRAG(temp_db_path, create=True) as client:
        session = writing(client)
        parent = await session.document_repository.create(
            Document(
                content="", uri=parent_uri, metadata={"content_type": "application/pdf"}
            )
        )
        await _reconcile_pdf_attachments(session, None, parent, pdf_bytes, depth=0)
        [child] = await client.list_documents(filter=parent_uri_filter(parent_uri))
        assert child.id is not None
        await client.edit_document(child.id, user_metadata=EDITED)

        await _reconcile_pdf_attachments(
            session, None, parent, pdf_bytes, depth=0, force=True
        )
        stored = await client.get_document_by_id(child.id)

    assert stored is not None
    assert stored.metadata["user"] == EDITED


async def test_delete_of_an_unowned_document_only_refuses_a_source_owned_one(
    temp_db_path,
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        owned = await _import(client, "mem://owned")
        unowned = await _import(client, "mem://unowned")
        await client.set_document_source([owned.id], "src")

        with pytest.raises(SourceOwnedError, match="src"):
            await client.delete_document(owned.id, unowned_only=True)
        assert await client.get_document_by_id(owned.id) is not None

        assert await client.delete_document(unowned.id, unowned_only=True) is True
        assert await client.delete_document(owned.id, source_id="src") is True


async def test_delete_of_an_unowned_document_refuses_one_owned_while_it_waited(
    temp_db_path, monkeypatch
):
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await _import(client, "mem://a")

        lock = ObservedLock()
        monkeypatch.setattr(client.store, "_write_lock", lock)
        async with lock:
            task = asyncio.create_task(
                client.delete_document(doc.id, unowned_only=True)
            )
            await assert_waiting_for_lock(task, lock)
            doc.metadata = {"source_id": "src"}
            await writing(client).document_repository.update_meta(doc)
        with pytest.raises(SourceOwnedError):
            await task
        assert await client.get_document_by_id(doc.id) is not None

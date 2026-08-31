"""Tests for the v0.25.0 docling-compression migration.

The migration replaces the `docling_document_json` text column with a
compressed `docling_document` blob. Rerunning it over an already-compressed
table must leave those blobs untouched.
"""

import json

import pytest

from haiku.rag.store.compression import compress_json, decompress_json
from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades.v0_25_0 import _apply_compress_docling_document
from tests.store.legacy_documents import (
    DocumentRecordV3,
    DocumentRecordV4,
    documents_schema,
    seed_documents,
)

STAGING = "documents_v4_staging"
DOC_JSON = json.dumps({"schema_name": "DoclingDocument", "name": "test"})


async def _seed_v3(store: Store, records: list[DocumentRecordV3]) -> None:
    await seed_documents(store, documents_schema(DocumentRecordV3), records)


async def _seed_v4(store: Store, records: list[DocumentRecordV4]) -> None:
    await seed_documents(store, documents_schema(DocumentRecordV4), records)


async def _read_migrated(store: Store, doc_id: str) -> dict:
    rows = await store.documents_table.query().where(f"id = '{doc_id}'").to_list()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.asyncio
async def test_json_text_column_becomes_a_compressed_blob(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v3(
            store,
            [
                DocumentRecordV3(
                    id="doc-1",
                    content="hello",
                    uri="test://doc-1",
                    title="Doc 1",
                    metadata='{"k": "v"}',
                    docling_document_json=DOC_JSON,
                    docling_version="1.10.0",
                    created_at="2026-01-01",
                    updated_at="2026-01-02",
                ),
                DocumentRecordV3(id="doc-2", content="no docling"),
            ],
        )

        await _apply_compress_docling_document(store)

        with_docling = await _read_migrated(store, "doc-1")
        without_docling = await _read_migrated(store, "doc-2")

    assert decompress_json(with_docling["docling_document"]) == DOC_JSON
    assert with_docling["uri"] == "test://doc-1"
    assert with_docling["title"] == "Doc 1"
    assert with_docling["metadata"] == '{"k": "v"}'
    assert with_docling["docling_version"] == "1.10.0"
    assert with_docling["created_at"] == "2026-01-01"
    assert with_docling["updated_at"] == "2026-01-02"
    assert without_docling["docling_document"] is None


@pytest.mark.asyncio
async def test_already_compressed_blobs_are_left_byte_identical(temp_db_path):
    """Rerunning over a migrated table must not re-compress what it finds."""
    blob = compress_json(DOC_JSON)
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(
            store,
            [DocumentRecordV4(id="doc-1", content="hello", docling_document=blob)],
        )

        await _apply_compress_docling_document(store)

        row = await _read_migrated(store, "doc-1")

    assert row["docling_document"] == blob


@pytest.mark.asyncio
async def test_uncompressed_blob_is_compressed(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(
            store,
            [
                DocumentRecordV4(
                    id="doc-1",
                    content="hello",
                    docling_document=DOC_JSON.encode("utf-8"),
                )
            ],
        )

        await _apply_compress_docling_document(store)

        row = await _read_migrated(store, "doc-1")

    assert decompress_json(row["docling_document"]) == DOC_JSON


@pytest.mark.asyncio
async def test_migrates_batches_larger_than_batch_size(temp_db_path):
    """BATCH_SIZE is 10; the staging round-trip must carry every document."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v3(
            store,
            [
                DocumentRecordV3(
                    id=f"doc-{n}",
                    content=f"body {n}",
                    docling_document_json=json.dumps({"name": f"doc-{n}"}),
                )
                for n in range(23)
            ],
        )

        await _apply_compress_docling_document(store)

        rows = await store.documents_table.query().to_list()
        assert STAGING not in (await store.db.list_tables()).tables

    assert {row["id"] for row in rows} == {f"doc-{n}" for n in range(23)}
    for row in rows:
        assert json.loads(decompress_json(row["docling_document"]))["name"] == row["id"]


@pytest.mark.asyncio
async def test_stale_staging_table_is_replaced(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v3(
            store,
            [
                DocumentRecordV3(
                    id="doc-1", content="hello", docling_document_json=DOC_JSON
                )
            ],
        )
        await store.db.create_table(STAGING, schema=documents_schema(DocumentRecordV4))

        await _apply_compress_docling_document(store)

        rows = await store.documents_table.query().to_list()
        assert STAGING not in (await store.db.list_tables()).tables

    assert [row["id"] for row in rows] == ["doc-1"]


@pytest.mark.asyncio
async def test_recovers_documents_from_staging_when_documents_table_is_empty(
    temp_db_path,
):
    """An interrupted run can leave the documents table emptied and every
    migrated row in staging; the rerun must adopt staging rather than drop it."""
    blob = compress_json(DOC_JSON)
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v3(store, [])
        staging = await store.db.create_table(
            STAGING, schema=documents_schema(DocumentRecordV4)
        )
        await staging.add(
            [DocumentRecordV4(id="doc-1", content="hello", docling_document=blob)]
        )

        await _apply_compress_docling_document(store)

        row = await _read_migrated(store, "doc-1")
        assert STAGING not in (await store.db.list_tables()).tables

    assert row["docling_document"] == blob


@pytest.mark.asyncio
async def test_unreadable_documents_table_falls_back_to_staging(temp_db_path):
    """A documents table without an `id` column cannot be enumerated."""
    blob = compress_json(DOC_JSON)
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await seed_documents(
            store,
            documents_schema(DocumentRecordV3).remove(0),
            [],
        )
        staging = await store.db.create_table(
            STAGING, schema=documents_schema(DocumentRecordV4)
        )
        await staging.add(
            [DocumentRecordV4(id="doc-1", content="hello", docling_document=blob)]
        )

        await _apply_compress_docling_document(store)

        row = await _read_migrated(store, "doc-1")
        assert STAGING not in (await store.db.list_tables()).tables

    assert row["docling_document"] == blob


@pytest.mark.asyncio
async def test_empty_database_is_rebuilt_on_the_new_schema(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v3(store, [])

        await _apply_compress_docling_document(store)

        names = {field.name for field in await store.documents_table.schema()}

    assert "docling_document" in names
    assert "docling_document_json" not in names


@pytest.mark.asyncio
async def test_empty_staging_table_is_not_mistaken_for_recovery(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v3(store, [])
        await store.db.create_table(STAGING, schema=documents_schema(DocumentRecordV4))

        await _apply_compress_docling_document(store)

        names = {field.name for field in await store.documents_table.schema()}

    assert "docling_document" in names
    assert "docling_document_json" not in names

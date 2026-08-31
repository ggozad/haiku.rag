"""Tests for the v0.38.0 page-splitting migration.

The migration reads the single ``docling_document`` blob written by v0.25.0,
splits page images into ``docling_pages``, and re-compresses both with zstd.
That blob reaches it gzip-compressed, zstd-compressed or uncompressed.
"""

import gzip
import json

import lancedb
import pytest

from haiku.rag.store.compression import compress_json, decompress_json
from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades.v0_38_0 import _apply_split_pages_zstd
from tests.store.legacy_documents import (
    DocumentRecordV4,
    LegacyDocumentRecord,
    documents_schema,
    seed_documents,
)

STAGING = "documents_v5_staging"


def _docling_doc(name: str = "test", with_pages: bool = True) -> dict:
    """A minimal DoclingDocument dict carrying one page image."""
    doc: dict = {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": name,
        "texts": [],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": [], "label": "unspecified"},
        "furniture": {
            "self_ref": "#/furniture",
            "children": [],
            "label": "unspecified",
        },
    }
    if with_pages:
        doc["pages"] = {"1": {"page_no": 1, "size": {"width": 10.0, "height": 20.0}}}
    return doc


async def _seed_v4(store: Store, records: list[DocumentRecordV4]) -> None:
    await seed_documents(store, documents_schema(DocumentRecordV4), records)


async def _read_migrated(store: Store, doc_id: str) -> dict:
    rows = await store.documents_table.query().where(f"id = '{doc_id}'").to_list()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encode",
    [
        pytest.param(
            lambda doc: gzip.compress(json.dumps(doc).encode("utf-8")), id="gzip"
        ),
        pytest.param(lambda doc: compress_json(json.dumps(doc)), id="zstd"),
        pytest.param(lambda doc: json.dumps(doc).encode("utf-8"), id="uncompressed"),
    ],
)
async def test_migrates_every_v0_25_0_blob_encoding(temp_db_path, encode):
    """Every encoding a v0.25.0 database can carry migrates to zstd."""
    doc = _docling_doc()
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(
            store,
            [
                DocumentRecordV4(
                    id="doc-1",
                    content="hello",
                    uri="test://doc-1",
                    title="Doc 1",
                    metadata='{"k": "v"}',
                    docling_document=encode(doc),
                    docling_version="1.10.0",
                    created_at="2026-01-01",
                    updated_at="2026-01-02",
                )
            ],
        )

        await _apply_split_pages_zstd(store)

        row = await _read_migrated(store, "doc-1")

    structure = json.loads(decompress_json(row["docling_document"]))
    assert structure["name"] == "test"
    assert "pages" not in structure
    assert json.loads(decompress_json(row["docling_pages"]))["1"]["page_no"] == 1
    assert row["uri"] == "test://doc-1"
    assert row["title"] == "Doc 1"
    assert row["metadata"] == '{"k": "v"}'
    assert row["docling_version"] == "1.10.0"
    assert row["created_at"] == "2026-01-01"
    assert row["updated_at"] == "2026-01-02"


@pytest.mark.asyncio
async def test_document_without_pages_gets_null_pages_column(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(
            store,
            [
                DocumentRecordV4(
                    id="doc-1",
                    content="hello",
                    docling_document=compress_json(
                        json.dumps(_docling_doc(with_pages=False))
                    ),
                ),
                DocumentRecordV4(id="doc-2", content="no blob"),
            ],
        )

        await _apply_split_pages_zstd(store)

        with_blob = await _read_migrated(store, "doc-1")
        without_blob = await _read_migrated(store, "doc-2")

    assert with_blob["docling_pages"] is None
    assert json.loads(decompress_json(with_blob["docling_document"]))["name"] == "test"
    assert without_blob["docling_document"] is None
    assert without_blob["docling_pages"] is None


@pytest.mark.asyncio
async def test_migrates_batches_larger_than_batch_size(temp_db_path):
    """BATCH_SIZE is 5; the staging round-trip must carry every document."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(
            store,
            [
                DocumentRecordV4(
                    id=f"doc-{n}",
                    content=f"body {n}",
                    docling_document=compress_json(
                        json.dumps(_docling_doc(name=f"doc-{n}"))
                    ),
                )
                for n in range(12)
            ],
        )

        await _apply_split_pages_zstd(store)

        rows = await store.documents_table.query().to_list()
        assert STAGING not in (await store.db.list_tables()).tables

    assert {row["id"] for row in rows} == {f"doc-{n}" for n in range(12)}
    for row in rows:
        structure = json.loads(decompress_json(row["docling_document"]))
        assert structure["name"] == row["id"]


@pytest.mark.asyncio
async def test_stale_staging_table_is_replaced(temp_db_path):
    """A staging table left by an interrupted run is dropped, not appended to."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(store, [])
        await store.db.create_table(
            STAGING, schema=documents_schema(LegacyDocumentRecord)
        )
        await store.documents_table.add(
            [
                DocumentRecordV4(
                    id="doc-1",
                    content="hello",
                    docling_document=compress_json(json.dumps(_docling_doc())),
                )
            ]
        )

        await _apply_split_pages_zstd(store)

        rows = await store.documents_table.query().to_list()
        assert STAGING not in (await store.db.list_tables()).tables

    assert [row["id"] for row in rows] == ["doc-1"]


@pytest.mark.asyncio
async def test_recovers_documents_from_staging_when_documents_table_is_empty(
    temp_db_path,
):
    """An interrupted run can leave the documents table emptied and every
    migrated row in staging; the rerun must adopt staging rather than drop it."""
    doc = _docling_doc()
    pages = compress_json(json.dumps(doc.pop("pages")))
    structure = compress_json(json.dumps(doc))

    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(store, [])
        staging = await store.db.create_table(
            STAGING, schema=documents_schema(LegacyDocumentRecord)
        )
        await staging.add(
            [
                {
                    "id": "doc-1",
                    "content": "hello",
                    "uri": "test://doc-1",
                    "title": "Doc 1",
                    "metadata": "{}",
                    "docling_document": structure,
                    "docling_pages": pages,
                    "docling_version": "1.10.0",
                    "created_at": "",
                    "updated_at": "",
                }
            ]
        )

        await _apply_split_pages_zstd(store)

        row = await _read_migrated(store, "doc-1")
        assert STAGING not in (await store.db.list_tables()).tables

    assert row["docling_document"] == structure
    assert row["docling_pages"] == pages


@pytest.mark.asyncio
async def test_unreadable_documents_table_falls_back_to_staging(
    temp_db_path, monkeypatch
):
    """An unreadable documents table falls back to adopting staging."""
    structure = compress_json(json.dumps(_docling_doc(with_pages=False)))
    reads: list[str] = []
    original = lancedb.AsyncTable.query

    def failing_query(self):
        reads.append(self.name)
        if self.name == "documents" and reads.count("documents") == 1:
            raise OSError("simulated read failure")
        return original(self)

    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(store, [])
        staging = await store.db.create_table(
            STAGING, schema=documents_schema(LegacyDocumentRecord)
        )
        await staging.add(
            [
                {
                    "id": "doc-1",
                    "content": "hello",
                    "uri": None,
                    "title": None,
                    "metadata": "{}",
                    "docling_document": structure,
                    "docling_pages": None,
                    "docling_version": None,
                    "created_at": "",
                    "updated_at": "",
                }
            ]
        )

        monkeypatch.setattr(lancedb.AsyncTable, "query", failing_query)
        await _apply_split_pages_zstd(store)
        monkeypatch.undo()

        row = await _read_migrated(store, "doc-1")
        assert STAGING not in (await store.db.list_tables()).tables

    assert row["docling_document"] == structure


@pytest.mark.asyncio
async def test_empty_database_is_rebuilt_on_the_new_schema(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(store, [])

        await _apply_split_pages_zstd(store)

        names = {field.name for field in await store.documents_table.schema()}

    assert "docling_pages" in names


@pytest.mark.asyncio
async def test_empty_staging_table_is_not_mistaken_for_recovery(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v4(store, [])
        await store.db.create_table(
            STAGING, schema=documents_schema(LegacyDocumentRecord)
        )

        await _apply_split_pages_zstd(store)

        names = {field.name for field in await store.documents_table.schema()}

    assert "docling_pages" in names

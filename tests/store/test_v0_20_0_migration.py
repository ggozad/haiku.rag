"""Tests for the v0.20.0 docling-column migration.

The migration rebuilds `documents` with `docling_document_json` and
`docling_version` added, carrying the existing rows across as NULL in both.
"""

import pyarrow as pa
import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades.v0_20_0 import _apply_add_docling_document_columns
from tests.store.legacy_documents import (
    DocumentRecordV2,
    documents_schema,
    seed_documents,
)


async def _seed_v2(store: Store, records: list[DocumentRecordV2]) -> None:
    await seed_documents(store, documents_schema(DocumentRecordV2), records)


@pytest.mark.asyncio
async def test_existing_rows_survive_with_empty_docling_columns(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v2(
            store,
            [
                DocumentRecordV2(
                    id="doc-1",
                    content="hello",
                    uri="test://doc-1",
                    title="Doc 1",
                    metadata='{"k": "v"}',
                    created_at="2026-01-01",
                    updated_at="2026-01-02",
                ),
                DocumentRecordV2(id="doc-2", content="second"),
            ],
        )

        await _apply_add_docling_document_columns(store)

        rows = sorted(
            await store.documents_table.query().to_list(), key=lambda r: r["id"]
        )

    assert [row["id"] for row in rows] == ["doc-1", "doc-2"]
    assert rows[0]["content"] == "hello"
    assert rows[0]["uri"] == "test://doc-1"
    assert rows[0]["title"] == "Doc 1"
    assert rows[0]["metadata"] == '{"k": "v"}'
    assert rows[0]["created_at"] == "2026-01-01"
    assert rows[0]["updated_at"] == "2026-01-02"
    for row in rows:
        assert row["docling_document_json"] is None
        assert row["docling_version"] is None


@pytest.mark.asyncio
async def test_null_metadata_becomes_an_empty_json_object(temp_db_path):
    """`metadata` is non-nullable from 0.20.0 on, so a NULL must be coerced."""
    nullable_metadata = pa.schema(
        [
            pa.field(f.name, f.type, nullable=f.nullable or f.name == "metadata")
            for f in documents_schema(DocumentRecordV2)
        ]
    )
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await seed_documents(store, nullable_metadata, [])
        await store.documents_table.add(
            [
                {
                    "id": "doc-1",
                    "content": "hello",
                    "uri": None,
                    "title": None,
                    "metadata": None,
                    "created_at": "",
                    "updated_at": "",
                }
            ]
        )

        await _apply_add_docling_document_columns(store)

        rows = await store.documents_table.query().to_list()

    assert rows[0]["metadata"] == "{}"


@pytest.mark.asyncio
async def test_empty_database_is_rebuilt_on_the_new_schema(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _seed_v2(store, [])

        await _apply_add_docling_document_columns(store)

        names = {field.name for field in await store.documents_table.schema()}
        rows = await store.documents_table.query().to_list()

    assert {"docling_document_json", "docling_version"} <= names
    assert rows == []


@pytest.mark.asyncio
async def test_missing_documents_table_is_created(temp_db_path):
    """Reruns after an interrupted migration find no documents table at all."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await store.db.drop_table("documents")

        await _apply_add_docling_document_columns(store)

        names = {field.name for field in await store.documents_table.schema()}

    assert {"docling_document_json", "docling_version"} <= names

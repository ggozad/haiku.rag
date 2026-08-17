import pytest
from lancedb.index import BTree

from haiku.rag.store.engine import Store
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.upgrades.v0_75_0 import _apply_index_hot_lookup_keys

# What a pre-0.75.0 database carried: no index on `documents` at all, FTS only on
# `chunks`, and `document_items` without the `label` Bitmap.
LEGACY_DROPPED = {
    "documents": ["id_idx"],
    "chunks": ["id_idx", "document_id_idx"],
    "document_items": ["label_idx"],
}


async def _indexed(table) -> dict[str, str]:
    return {
        column: index.index_type
        for index in await table.list_indices()
        for column in index.columns
    }


async def _make_legacy(store: Store) -> None:
    for table_name, indexes in LEGACY_DROPPED.items():
        table = store._tables()[table_name]
        for index in indexes:
            await table.drop_index(index)


@pytest.mark.asyncio
async def test_adds_the_missing_indexes(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _make_legacy(store)

        await _apply_index_hot_lookup_keys(store)

        assert await _indexed(store.documents_table) == {"id": "BTree"}
        assert await _indexed(store.chunks_table) == {
            "content_fts": "FTS",
            "id": "BTree",
            "document_id": "BTree",
        }
        assert await _indexed(store.document_items_table) == {
            "document_id": "BTree",
            "position": "BTree",
            "self_ref": "BTree",
            "label": "Bitmap",
        }


@pytest.mark.asyncio
async def test_keeps_every_row(temp_db_path):
    """Indexing must not touch data."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="Kept", uri="test://kept"))
        await _make_legacy(store)

        await _apply_index_hot_lookup_keys(store)

        docs = await repo.list_all(include_content=True)
        assert [d.content for d in docs] == ["Kept"]


@pytest.mark.asyncio
async def test_is_a_no_op_on_an_already_indexed_database(temp_db_path):
    """An already-migrated database must not be re-indexed. `create_index` with
    replace=True rebuilds, so a second run would re-sort every indexed column
    and write a new version of every table.
    """
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _apply_index_hot_lookup_keys(store)
        versions = {name: await t.version() for name, t in store._tables().items()}

        await _apply_index_hot_lookup_keys(store)

        assert {name: await t.version() for name, t in store._tables().items()} == (
            versions
        )


@pytest.mark.asyncio
async def test_replaces_a_wrong_typed_legacy_index(temp_db_path):
    """A database that indexed `label` with a BTree gets the Bitmap it should
    have, rather than being left alone because the column was covered."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await store.document_items_table.create_index(
            "label", config=BTree(), replace=True
        )

        await _apply_index_hot_lookup_keys(store)

        indexed = await _indexed(store.document_items_table)
        assert indexed["label"] == "Bitmap"


@pytest.mark.asyncio
async def test_leaves_undeclared_indexes_alone(temp_db_path):
    """A merged or externally built database may carry indexes haiku.rag never
    creates. The migration must not drop them."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await store.document_meta_table.create_index(
            "title", config=BTree(), replace=True
        )

        await _apply_index_hot_lookup_keys(store)

        assert "title" in await _indexed(store.document_meta_table)

import pyarrow as pa
import pytest
from lancedb.index import BTree

from haiku.rag.store.engine import Store, ensure_indexes
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository

EXPECTED_INDEXED_COLUMNS = {
    "documents": {"id"},
    "document_meta": {"id", "uri"},
    "chunks": {"content_fts", "id", "document_id"},
    "document_items": {"document_id", "position", "self_ref", "label"},
}


async def _indexed_columns(table) -> set[str]:
    return {column for index in await table.list_indices() for column in index.columns}


async def _index_type(table, column: str) -> str | None:
    for index in await table.list_indices():
        if column in index.columns:
            return index.index_type
    return None


@pytest.mark.asyncio
async def test_fresh_database_indexes_every_hot_lookup_key(temp_db_path):
    """A new database carries the full index set, not a subset."""
    async with Store(temp_db_path, create=True) as store:
        for name, table in store._tables().items():
            expected = EXPECTED_INDEXED_COLUMNS.get(name, set())
            assert await _indexed_columns(table) == expected, name


@pytest.mark.asyncio
async def test_ensure_indexes_skips_existing_instead_of_rebuilding(temp_db_path):
    """`create_index(replace=True)` rebuilds an identical index and writes a new
    table version, so a second pass must skip rather than replace."""
    async with Store(temp_db_path, create=True) as store:
        table = store.chunks_table
        version_before = await table.version()

        await ensure_indexes(table, "chunks")

        assert await table.version() == version_before
        assert await _indexed_columns(table) == EXPECTED_INDEXED_COLUMNS["chunks"]


@pytest.mark.asyncio
async def test_ensure_indexes_corrects_an_index_of_the_wrong_type(temp_db_path):
    """A column indexed with the wrong type must be re-indexed. `label` is the
    live case: a BTree over ~ten distinct values loses the low-cardinality
    equality lookup a Bitmap gives, and column coverage alone cannot see it.
    """
    async with Store(temp_db_path, create=True) as store:
        table = store.document_items_table
        await table.create_index("label", config=BTree(), replace=True)
        assert await _index_type(table, "label") == "BTree"

        await ensure_indexes(table, "document_items")

        assert await _index_type(table, "label") == "Bitmap"
        assert (
            await _indexed_columns(table) == EXPECTED_INDEXED_COLUMNS["document_items"]
        )


@pytest.mark.asyncio
async def test_delete_all_restores_the_full_index_set(temp_db_path):
    """delete_all drops and recreates tables; the recreated tables must come
    back with the same indexes a fresh database gets."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="A document"))

        await repo.delete_all()

        for name, table in store._tables().items():
            expected = EXPECTED_INDEXED_COLUMNS.get(name, set())
            assert await _indexed_columns(table) == expected, name


@pytest.mark.asyncio
async def test_delete_all_keeps_picture_data_as_large_binary(temp_db_path):
    """document_items must be recreated from the Arrow schema, which declares
    picture_data as large_binary. The 32-bit `binary` type overflows its offsets
    once a fragment holds enough embedded pictures.
    """
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="A document"))

        await repo.delete_all()

        schema = await store.document_items_table.schema()
        assert schema.field("picture_data").type == pa.large_binary()

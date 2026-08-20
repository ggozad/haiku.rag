import pyarrow as pa
import pytest
from lancedb.index import BTree

from haiku.rag.store.engine import Store
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.schema import ensure_indexes

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


async def _covering(table, column: str) -> list[tuple[str, str]]:
    """Every index over `column`, as (name, index_type)."""
    return [
        (index.name, index.index_type)
        for index in await table.list_indices()
        if column in index.columns
    ]


@pytest.mark.asyncio
async def test_fresh_database_indexes_every_hot_lookup_key(temp_db_path):
    """A new database carries the full index set."""
    async with Store(temp_db_path, create=True) as store:
        for name, table in store._tables().items():
            expected = EXPECTED_INDEXED_COLUMNS.get(name, set())
            assert await _indexed_columns(table) == expected, name


@pytest.mark.asyncio
async def test_ensure_indexes_skips_existing_instead_of_rebuilding(temp_db_path):
    """A second pass must not rebuild: replace=True writes a new version."""
    async with Store(temp_db_path, create=True) as store:
        table = store.chunks_table
        version_before = await table.version()

        await ensure_indexes(table, "chunks")

        assert await table.version() == version_before
        assert await _indexed_columns(table) == EXPECTED_INDEXED_COLUMNS["chunks"]


@pytest.mark.asyncio
async def test_ensure_indexes_corrects_an_index_of_the_wrong_type(temp_db_path):
    """A wrong-typed index does not satisfy the declared one."""
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
async def test_ensure_indexes_adds_the_declared_type_beside_a_custom_index(
    temp_db_path,
):
    """A custom-named index neither satisfies the check nor is destroyed."""
    async with Store(temp_db_path, create=True) as store:
        table = store.document_items_table
        await table.drop_index("label_idx")
        await table.create_index("label", config=BTree(), name="operator_label")

        await ensure_indexes(table, "document_items")

        covering = dict(await _covering(table, "label"))
        assert covering["operator_label"] == "BTree"
        assert "Bitmap" in covering.values()


@pytest.mark.asyncio
async def test_ensure_indexes_keeps_an_operator_index_on_a_declared_column(
    temp_db_path,
):
    """An index we did not declare survives, even on a declared column."""
    async with Store(temp_db_path, create=True) as store:
        table = store.document_items_table
        await table.create_index("label", config=BTree(), name="operator_label")

        await ensure_indexes(table, "document_items")

        covering = dict(await _covering(table, "label"))
        assert covering == {"label_idx": "Bitmap", "operator_label": "BTree"}


@pytest.mark.asyncio
async def test_delete_all_restores_the_full_index_set(temp_db_path):
    """Recreated tables come back with the full index set."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="A document"))

        await repo.delete_all()

        for name, table in store._tables().items():
            expected = EXPECTED_INDEXED_COLUMNS.get(name, set())
            assert await _indexed_columns(table) == expected, name


@pytest.mark.asyncio
async def test_delete_all_keeps_picture_data_as_large_binary(temp_db_path):
    """picture_data must survive delete_all as large_binary, not binary."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="A document"))

        await repo.delete_all()

        schema = await store.document_items_table.schema()
        assert schema.field("picture_data").type == pa.large_binary()

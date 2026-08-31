import pyarrow as pa
import pytest
from lancedb.index import FTS, BTree

from haiku.rag.store.engine import Store
from haiku.rag.store.models import Chunk, Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.schema import ensure_indexes

EXPECTED_INDEXED_COLUMNS = {
    "documents": {"id"},
    "document_meta": {"id", "uri"},
    "chunks": {"id", "document_id"},
    "document_items": {"document_id", "position", "self_ref", "label"},
}

# content_fts joins the set once the table holds rows to index.
EXPECTED_POPULATED_CHUNK_COLUMNS = {"content_fts", "id", "document_id"}


async def _fts_indexed_rows(table) -> int | None:
    """Rows the FTS index covers, or None when there is no FTS index."""
    for index in await table.list_indices():
        if index.index_type == "FTS":
            return (await table.index_stats(index.name)).num_indexed_rows
    return None


async def _add_chunk(
    store, content: str = "a chunk about gardens", document_id: str = "doc-1"
) -> None:
    await ChunkRepository(store).create(
        Chunk(
            document_id=document_id,
            content=content,
            embedding=[0.1] * store.embedder.vector_dim,
            order=0,
        )
    )


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
        await table.create_index(
            "label", config=BTree(), replace=True, name="label_idx"
        )
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


@pytest.mark.asyncio
async def test_empty_chunks_table_carries_no_fts_index(temp_db_path):
    """An FTS index over no rows indexes nothing and lance never catches it up,
    so it waits for rows rather than being built with the table."""
    async with Store(temp_db_path, create=True) as store:
        assert await _fts_indexed_rows(store.chunks_table) is None


@pytest.mark.asyncio
async def test_ensure_indexes_skips_fts_while_the_table_is_empty(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await ensure_indexes(store.chunks_table, "chunks")

        assert await _fts_indexed_rows(store.chunks_table) is None
        assert (
            await _indexed_columns(store.chunks_table)
            == (EXPECTED_INDEXED_COLUMNS["chunks"])
        )


@pytest.mark.asyncio
async def test_ensure_indexes_builds_fts_over_existing_rows(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await store.chunks_table.add(
            [
                store.ChunkRecord(
                    document_id="doc-1",
                    content="a chunk",
                    content_fts="a chunk",
                    metadata="{}",
                    order=0,
                    vector=[0.1] * store.embedder.vector_dim,
                )
            ]
        )

        await ensure_indexes(store.chunks_table, "chunks")

        assert await _fts_indexed_rows(store.chunks_table) == 1
        assert await _indexed_columns(store.chunks_table) == (
            EXPECTED_POPULATED_CHUNK_COLUMNS
        )


@pytest.mark.asyncio
async def test_creating_chunks_builds_a_covering_fts_index(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await _add_chunk(store)

        assert await _fts_indexed_rows(store.chunks_table) == 1


@pytest.mark.asyncio
async def test_replacing_chunks_builds_a_covering_fts_index(temp_db_path):
    """replace_for_document inserts where nothing matches, so it can be the
    first write into a fresh table."""
    async with Store(temp_db_path, create=True) as store:
        await ChunkRepository(store).replace_for_document(
            "doc-1",
            [
                Chunk(
                    document_id="doc-1",
                    content="a chunk about gardens",
                    embedding=[0.1] * store.embedder.vector_dim,
                    order=0,
                )
            ],
        )

        assert await _fts_indexed_rows(store.chunks_table) == 1


@pytest.mark.asyncio
async def test_a_second_write_leaves_the_covering_index_in_place(temp_db_path):
    """One indexed row is enough: later rows merge as a scanned tail, so the
    index is not rebuilt per write."""
    async with Store(temp_db_path, create=True) as store:
        await _add_chunk(store, "first")
        version_after_first = await store.chunks_table.version()

        await _add_chunk(store, "second")

        assert await _fts_indexed_rows(store.chunks_table) == 1
        assert await store.chunks_table.version() == version_after_first + 1


@pytest.mark.asyncio
async def test_delete_all_then_write_rebuilds_a_covering_fts_index(temp_db_path):
    """delete_all recreates the table empty, so the next write owns the index."""
    async with Store(temp_db_path, create=True) as store:
        await _add_chunk(store)
        await ChunkRepository(store).delete_all()
        assert await _fts_indexed_rows(store.chunks_table) is None

        await _add_chunk(store)

        assert await _fts_indexed_rows(store.chunks_table) == 1


@pytest.mark.asyncio
async def test_deleting_the_indexed_rows_rebuilds_the_fts_index(temp_db_path):
    """Deleting every row the index covers, while unindexed rows remain,
    reaches the zero-coverage scan path; the delete repairs it."""
    async with Store(temp_db_path, create=True) as store:
        await _add_chunk(store, "first", document_id="doc-a")
        await _add_chunk(store, "second", document_id="doc-b")
        assert await _fts_indexed_rows(store.chunks_table) == 1

        await ChunkRepository(store).delete_by_document_id("doc-a")

        assert await store.chunks_table.count_rows() == 1
        assert await _fts_indexed_rows(store.chunks_table) == 1


@pytest.mark.asyncio
async def test_replacing_the_indexed_rows_rebuilds_the_fts_index(temp_db_path):
    """Replacement rewrites rows, and rewritten rows are unindexed."""
    async with Store(temp_db_path, create=True) as store:
        await _add_chunk(store, "first", document_id="doc-a")
        await _add_chunk(store, "second", document_id="doc-b")

        await ChunkRepository(store).replace_for_document(
            "doc-a",
            [
                Chunk(
                    document_id="doc-a",
                    content="rewritten",
                    embedding=[0.1] * store.embedder.vector_dim,
                    order=0,
                )
            ],
        )

        assert await _fts_indexed_rows(store.chunks_table) == 2


@pytest.mark.asyncio
async def test_a_write_repairs_a_legacy_index_that_covers_no_rows(temp_db_path):
    """A database whose FTS index predates its rows is repaired by the first
    write that runs index maintenance."""
    async with Store(temp_db_path, create=True) as store:
        await store.chunks_table.create_index(
            "content_fts", config=FTS(with_position=True, remove_stop_words=False)
        )
        await store.chunks_table.add(
            [
                store.ChunkRecord(
                    document_id="doc-a",
                    content="a legacy chunk",
                    content_fts="a legacy chunk",
                    metadata="{}",
                    order=0,
                    vector=[0.1] * store.embedder.vector_dim,
                )
            ]
        )
        assert await _fts_indexed_rows(store.chunks_table) == 0

        await _add_chunk(store, "second", document_id="doc-b")

        assert await _fts_indexed_rows(store.chunks_table) == 2


@pytest.mark.asyncio
async def test_unavailable_index_stats_repair_matches_doctor(temp_db_path, monkeypatch):
    """index_stats may return None; doctor treats that as uncovered, so the
    write-path repair does too."""
    from lancedb.table import AsyncTable

    async with Store(temp_db_path, create=True) as store:
        await _add_chunk(store)

        original = AsyncTable.index_stats

        async def no_stats(self, name):
            if self.name == "chunks":
                return None
            return await original(self, name)

        monkeypatch.setattr(AsyncTable, "index_stats", no_stats)
        applied = await ensure_indexes(store.chunks_table, "chunks")

        assert applied == ["content_fts"]

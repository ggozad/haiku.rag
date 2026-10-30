from lancedb.index import BTree

from haiku.rag.store.engine import Store
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.upgrades.v0_75_0 import _apply_index_hot_lookup_keys

# The indexes a pre-0.75.0 database lacked.
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


async def test_adds_missing_indexes_idempotently_and_keeps_rows_and_indexes(
    temp_db_path,
):
    """The run adds the declared indexes and keeps the rows and every undeclared
    index. A rerun writes no version."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="Kept", uri="test://kept"))
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
        await store.document_meta_table.create_index(
            "title", config=BTree(), replace=True
        )
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
        assert "title" in await _indexed(store.document_meta_table)
        docs = await repo.list_all(include_content=True)
        assert [d.content for d in docs] == ["Kept"]

        versions = {name: await t.version() for name, t in store._tables().items()}
        await _apply_index_hot_lookup_keys(store)
        assert {name: await t.version() for name, t in store._tables().items()} == (
            versions
        )

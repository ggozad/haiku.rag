import pytest

from haiku.rag.store import Store
from haiku.rag.store.engine import get_database_stats


class TestGetDatabaseStats:
    def test_empty_database_stats(self, temp_db_path):
        """get_database_stats() on a fresh database reports zero rows and no vector index."""
        store = Store(temp_db_path, create=True)

        stats = get_database_stats(store.db)

        for name in ("documents", "chunks", "document_items", "settings"):
            assert stats[name]["exists"] is True
            assert stats[name]["num_rows"] >= 0
            assert stats[name]["total_bytes"] >= 0
            assert stats[name]["num_versions"] >= 1

        assert stats["documents"]["num_rows"] == 0
        assert stats["chunks"]["num_rows"] == 0
        assert stats["chunks"]["has_vector_index"] is False

        store.close()

    def test_missing_tables_report_absent(self, temp_db_path):
        """Tables that don't exist on the connection are reported as absent."""
        import lancedb
        from lancedb.pydantic import LanceModel
        from pydantic import Field

        class SettingsRecord(LanceModel):
            id: str = Field(default="settings")
            settings: str = Field(default="{}")

        db = lancedb.connect(temp_db_path)
        db.create_table("settings", schema=SettingsRecord)

        stats = get_database_stats(db)

        assert stats["settings"]["exists"] is True
        assert stats["documents"] == {"exists": False}
        assert stats["chunks"] == {"exists": False}
        assert stats["document_items"] == {"exists": False}

    @pytest.mark.asyncio
    async def test_stats_after_adding_document(self, temp_db_path):
        """get_database_stats() reflects document and chunk counts after inserts."""
        from haiku.rag.store.models import Chunk, Document
        from haiku.rag.store.repositories.chunk import ChunkRepository
        from haiku.rag.store.repositories.document import DocumentRepository

        store = Store(temp_db_path, create=True)
        doc_repo = DocumentRepository(store)
        chunk_repo = ChunkRepository(store)

        doc = await doc_repo.create(Document(content="hello world"))
        assert doc.id is not None

        await chunk_repo.create(
            Chunk(
                content="hello world",
                document_id=doc.id,
                embedding=[0.0] * store.embedder._vector_dim,
            )
        )

        stats = get_database_stats(store.db)
        assert stats["documents"]["num_rows"] == 1
        assert stats["chunks"]["num_rows"] == 1
        store.close()

    def test_stats_with_vector_index(self, temp_db_path):
        """get_database_stats() reports vector index details once an index exists."""
        from datetime import timedelta

        store = Store(temp_db_path, create=True)
        dim = store.embedder._vector_dim

        # Need >=256 rows for IVF_PQ training.
        rows = [
            {
                "id": f"chunk-{i}",
                "document_id": "doc-1",
                "content": f"content {i}",
                "content_fts": "",
                "metadata": "{}",
                "order": i,
                "vector": [float(i % 7) + 0.01 * j for j in range(dim)],
            }
            for i in range(256)
        ]
        store.chunks_table.add(rows)
        store.chunks_table.create_index(
            metric="cosine", index_type="IVF_PQ", replace=True
        )
        store.chunks_table.wait_for_index(["vector_idx"], timeout=timedelta(minutes=1))

        stats = get_database_stats(store.db)
        assert stats["chunks"]["has_vector_index"] is True
        assert stats["chunks"]["num_indexed_rows"] >= 0
        assert "num_unindexed_rows" in stats["chunks"]
        store.close()

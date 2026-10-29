from haiku.rag.store.engine import Store
from haiku.rag.store.info import get_database_stats


class TestStoredSettings:
    async def test_a_new_database_carries_the_version_it_was_created_with(
        self, temp_db_path
    ):
        """Creating writes the settings row and empty tables, and the store reports them."""
        from importlib import metadata

        async with Store(temp_db_path, create=True) as store:
            assert store.stored_settings["version"] == metadata.version(
                "haiku.rag-slim"
            )
            stats = await get_database_stats(store.db)

        for name in ("documents", "chunks", "document_items", "settings"):
            assert stats[name]["exists"] is True
            assert stats[name]["num_rows"] >= 0
            assert stats[name]["total_bytes"] >= 0
            assert stats[name]["num_versions"] >= 1
        assert stats["documents"]["num_rows"] == 0
        assert stats["chunks"]["num_rows"] == 0
        assert stats["chunks"]["has_vector_index"] is False

    async def test_an_existing_database_carries_its_stored_settings(self, temp_db_path):
        """Read once on open: reporting on a database reads them from
        here."""
        async with Store(temp_db_path, create=True) as store:
            written = store.stored_settings

        async with Store(temp_db_path) as store:
            assert store.stored_settings == written
            assert store.stored_settings["embeddings"]["model"]["vector_dim"] > 0

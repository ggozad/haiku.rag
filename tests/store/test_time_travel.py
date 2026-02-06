import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from haiku.rag.store import ReadOnlyError, Store
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository


class TestStoreTimeTravel:
    def test_store_with_before_is_read_only(self, temp_db_path):
        """Store with before parameter is automatically read-only."""
        # Create a store first
        store = Store(temp_db_path, create=True)
        store.close()

        # Open with before - should be read-only
        before = datetime.now(UTC) + timedelta(hours=1)
        store = Store(temp_db_path, before=before)
        assert store.is_read_only is True
        store.close()

    def test_store_before_raises_on_write(self, temp_db_path):
        """Store with before parameter raises on write operations."""
        store = Store(temp_db_path, create=True)
        store.close()

        before = datetime.now(UTC) + timedelta(hours=1)
        store = Store(temp_db_path, before=before)

        with pytest.raises(ReadOnlyError):
            store._assert_writable()
        store.close()

    @pytest.mark.asyncio
    async def test_store_before_checks_out_historical_state(self, temp_db_path):
        """Store with before parameter checks out tables to historical state."""
        # Create store and add a document
        store = Store(temp_db_path, create=True)
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))

        # Get the version timestamp after first document
        versions_after_first = store.list_table_versions("documents")
        # Find the latest version timestamp
        latest_version = max(versions_after_first, key=lambda v: v["version"])
        time_after_first = latest_version["timestamp"]

        # Wait a bit to ensure the next write gets a distinct timestamp
        await asyncio.sleep(0.5)

        # Add second document
        await repo.create(Document(content="Second document"))

        # Verify we have more versions now
        versions_after_second = store.list_table_versions("documents")
        assert len(versions_after_second) > len(versions_after_first)

        store.close()

        # Open at historical state (using the timestamp from after first write)
        store = Store(temp_db_path, before=time_after_first)
        repo = DocumentRepository(store)

        # Should only see first document
        docs = await repo.list_all(include_content=True)
        assert len(docs) == 1
        assert docs[0].content == "First document"
        store.close()

        # Open at current state
        store = Store(temp_db_path)
        repo = DocumentRepository(store)

        # Should see both documents
        docs = await repo.list_all()
        assert len(docs) == 2
        store.close()

    def test_store_before_no_version_raises(self, temp_db_path):
        """Store with before datetime before any version raises ValueError."""
        store = Store(temp_db_path, create=True)
        store.close()

        # Try to open before the database was created
        before = datetime(2000, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError) as exc_info:
            Store(temp_db_path, before=before)
        assert "No data exists before" in str(exc_info.value)

    def test_current_table_versions_returns_versions(self, temp_db_path):
        """current_table_versions returns dict of table versions."""
        store = Store(temp_db_path, create=True)
        versions = store.current_table_versions()

        assert "documents" in versions
        assert "chunks" in versions
        assert "settings" in versions
        assert all(isinstance(v, int) for v in versions.values())
        store.close()

    def test_list_table_versions_returns_history(self, temp_db_path):
        """list_table_versions returns version history for a table."""
        store = Store(temp_db_path, create=True)
        versions = store.list_table_versions("documents")

        assert len(versions) >= 1
        for v in versions:
            assert "version" in v
            assert "timestamp" in v
        store.close()

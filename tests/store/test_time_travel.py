import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from haiku.rag.store import ReadOnlyError, Store
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository


class TestStoreTimeTravel:
    @pytest.mark.asyncio
    async def test_store_with_before_is_read_only(self, temp_db_path):
        """Store with before parameter is automatically read-only."""
        async with Store(temp_db_path, create=True):
            pass

        before = datetime.now(UTC) + timedelta(hours=1)
        async with Store(temp_db_path, before=before) as store:
            assert store.is_read_only is True

    @pytest.mark.asyncio
    async def test_store_before_raises_on_write(self, temp_db_path):
        """Store with before parameter raises on write operations."""
        async with Store(temp_db_path, create=True):
            pass

        before = datetime.now(UTC) + timedelta(hours=1)
        async with Store(temp_db_path, before=before) as store:
            with pytest.raises(ReadOnlyError):
                store._assert_writable()

    @pytest.mark.asyncio
    async def test_store_before_checks_out_historical_state(self, temp_db_path):
        """Store with before parameter checks out tables to historical state."""
        async with Store(temp_db_path, create=True) as store:
            repo = DocumentRepository(store)
            await repo.create(Document(content="First document"))

            versions_after_first = await store.list_table_versions("documents")
            latest_version = max(versions_after_first, key=lambda v: v["version"])
            time_after_first = latest_version["timestamp"]

            await asyncio.sleep(0.5)

            await repo.create(Document(content="Second document"))

            versions_after_second = await store.list_table_versions("documents")
            assert len(versions_after_second) > len(versions_after_first)

        async with Store(temp_db_path, before=time_after_first) as store:
            repo = DocumentRepository(store)

            docs = await repo.list_all(include_content=True)
            assert len(docs) == 1
            assert docs[0].content == "First document"

        async with Store(temp_db_path) as store:
            repo = DocumentRepository(store)

            docs = await repo.list_all()
            assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_store_before_no_version_raises(self, temp_db_path):
        """Store with before datetime before any version raises ValueError."""
        async with Store(temp_db_path, create=True):
            pass

        before = datetime(2000, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError) as exc_info:
            async with Store(temp_db_path, before=before):
                pass
        assert "No data exists before" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_current_table_versions_returns_versions(self, temp_db_path):
        """current_table_versions returns dict of table versions."""
        async with Store(temp_db_path, create=True) as store:
            versions = await store.current_table_versions()

            assert "documents" in versions
            assert "chunks" in versions
            assert "settings" in versions
            assert all(isinstance(v, int) for v in versions.values())

    @pytest.mark.asyncio
    async def test_list_table_versions_returns_history(self, temp_db_path):
        """list_table_versions returns version history for a table."""
        async with Store(temp_db_path, create=True) as store:
            versions = await store.list_table_versions("documents")

            assert len(versions) >= 1
            for v in versions:
                assert "version" in v
                assert "timestamp" in v

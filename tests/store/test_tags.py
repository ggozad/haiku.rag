import asyncio

import pytest
from lancedb.table import AsyncTags

from haiku.rag.store import ReadOnlyError, Store
from haiku.rag.store.engine import REQUIRED_TABLES
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository


@pytest.mark.asyncio
async def test_create_and_list_tags(temp_db_path):
    """create_tag tags every table at its current version; list_tags reports
    the tag as complete with the exact versions."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))

        versions = await store.current_table_versions()
        await store.create_tag("release-1")

        tags = await store.list_tags()
        assert set(tags) == {"release-1"}
        info = tags["release-1"]
        assert info.complete is True
        assert info.missing_tables == []
        assert info.tables == versions


@pytest.mark.asyncio
async def test_create_tag_rejects_existing(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("release-1")

        with pytest.raises(ValueError, match="already exists"):
            await store.create_tag("release-1")

        tags = await store.list_tags()
        assert tags["release-1"].complete is True


@pytest.mark.asyncio
async def test_create_tag_rejects_partial_existing(temp_db_path):
    """A tag present on only some tables blocks creation before anything is
    written; the error tells the user to delete it first."""
    async with Store(temp_db_path, create=True) as store:
        version = await store.chunks_table.version()
        await store.chunks_table.tags.create("stale", version)

        with pytest.raises(ValueError, match="delete"):
            await store.create_tag("stale")

        tags = await store.list_tags()
        assert tags["stale"].complete is False
        assert set(tags["stale"].tables) == {"chunks"}
        assert set(tags["stale"].missing_tables) == set(REQUIRED_TABLES) - {"chunks"}


@pytest.mark.asyncio
async def test_create_tag_rolls_back_own_tags_on_failure(temp_db_path, monkeypatch):
    """A midway failure removes the tags this call created and leaves
    pre-existing tags untouched."""
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("keep")

        real_create = AsyncTags.create
        calls = {"n": 0}

        async def flaky(self, name: str, version: int) -> None:
            calls["n"] += 1
            if calls["n"] == 4:
                raise RuntimeError("boom")
            await real_create(self, name, version)

        monkeypatch.setattr(AsyncTags, "create", flaky)

        with pytest.raises(RuntimeError, match="boom"):
            await store.create_tag("broken")

        monkeypatch.undo()

        tags = await store.list_tags()
        assert "broken" not in tags
        assert tags["keep"].complete is True


@pytest.mark.asyncio
async def test_create_tag_reports_failed_cleanup(temp_db_path, monkeypatch):
    """When midway-failure cleanup also fails, the error reports both the
    original failure and the remaining partial-tag risk."""
    async with Store(temp_db_path, create=True) as store:
        real_create = AsyncTags.create
        calls = {"n": 0}

        async def flaky_create(self, name: str, version: int) -> None:
            calls["n"] += 1
            if calls["n"] == 4:
                raise RuntimeError("create boom")
            await real_create(self, name, version)

        async def failing_delete(self, name: str) -> None:
            raise RuntimeError("delete boom")

        monkeypatch.setattr(AsyncTags, "create", flaky_create)
        monkeypatch.setattr(AsyncTags, "delete", failing_delete)

        with pytest.raises(RuntimeError) as exc_info:
            await store.create_tag("broken")

        msg = str(exc_info.value)
        assert "create boom" in msg
        assert "partial" in msg
        assert exc_info.value.__cause__ is not None

        monkeypatch.undo()
        tags = await store.list_tags()
        assert tags["broken"].complete is False


@pytest.mark.asyncio
async def test_delete_tag_reports_failed_tables(temp_db_path, monkeypatch):
    """delete_tag never claims success when remnants remain: it names the
    tables where deletion failed."""
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("release-1")

        real_delete = AsyncTags.delete
        calls = {"n": 0}

        async def flaky_delete(self, name: str) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("delete boom")
            await real_delete(self, name)

        monkeypatch.setattr(AsyncTags, "delete", flaky_delete)

        with pytest.raises(RuntimeError) as exc_info:
            await store.delete_tag("release-1")

        assert "document_meta" in str(exc_info.value)

        monkeypatch.undo()
        tags = await store.list_tags()
        assert set(tags["release-1"].tables) == {"document_meta"}

        await store.delete_tag("release-1")
        assert await store.list_tags() == {}


@pytest.mark.asyncio
async def test_create_tag_waits_for_write_lock(temp_db_path):
    """create_tag serializes with client writes so a write cannot land
    between the version snapshot and the per-table tag creation."""
    async with Store(temp_db_path, create=True) as store:
        async with store._write_lock:
            task = asyncio.create_task(store.create_tag("release-1"))
            await asyncio.sleep(0.1)
            assert not task.done()
        await task

        tags = await store.list_tags()
        assert tags["release-1"].complete is True


@pytest.mark.asyncio
async def test_delete_tag_waits_for_write_lock(temp_db_path):
    """delete_tag serializes with create_tag and client writes so it cannot
    remove tags out from under a concurrent create_tag."""
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("release-1")

        async with store._write_lock:
            task = asyncio.create_task(store.delete_tag("release-1"))
            await asyncio.sleep(0.1)
            assert not task.done()
        await task

        assert await store.list_tags() == {}


@pytest.mark.asyncio
async def test_delete_tag(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("release-1")
        await store.delete_tag("release-1")

        assert await store.list_tags() == {}


@pytest.mark.asyncio
async def test_delete_tag_heals_partial(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        version = await store.chunks_table.version()
        await store.chunks_table.tags.create("stale", version)

        await store.delete_tag("stale")

        assert await store.list_tags() == {}


@pytest.mark.asyncio
async def test_delete_tag_missing_raises(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        with pytest.raises(ValueError, match="does not exist"):
            await store.delete_tag("nope")


@pytest.mark.asyncio
async def test_vacuum_cleans_untagged_versions_and_keeps_tagged(temp_db_path):
    """Vacuum must both preserve tagged versions (lance hard-errors when a
    tagged version falls inside the cleanup window, which vacuum would
    swallow) and still clean untagged versions older than the oldest tag's
    safety margin."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        versions_before = [
            v["version"] for v in await store.list_table_versions("documents")
        ]

        # Age the pre-tag versions past the retention safety margin.
        await asyncio.sleep(1.5)

        await repo.create(Document(content="Second document"))
        await store.create_tag("release-1")
        tagged_version = (await store.list_tags())["release-1"].tables["documents"]

        await store.vacuum(retention_seconds=0)

        remaining = [v["version"] for v in await store.list_table_versions("documents")]
        assert tagged_version in remaining
        assert min(versions_before) not in remaining

        await store.documents_table.checkout("release-1")
        rows = await store.documents_table.count_rows()
        await store.documents_table.checkout_latest()
        assert rows == 2


@pytest.mark.asyncio
async def test_tag_operations_rejected_during_rebuild(temp_db_path):
    """While a rebuild holds the rebuild lock, tag operations fail fast
    instead of snapshotting a half-rebuilt database."""
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("keep")

        async with store._rebuild_lock:
            with pytest.raises(ValueError, match="[Rr]ebuild in progress"):
                await store.create_tag("release-1")
            with pytest.raises(ValueError, match="[Rr]ebuild in progress"):
                await store.delete_tag("keep")

        await store.create_tag("release-1")
        await store.delete_tag("keep")
        assert set(await store.list_tags()) == {"release-1"}


@pytest.mark.asyncio
async def test_vacuum_waits_for_write_lock(temp_db_path):
    """Vacuum serializes with writers and tag operations so a tag cannot be
    created between _tag_safe_retention's read and the optimize call."""
    async with Store(temp_db_path, create=True) as store:
        async with store._write_lock:
            task = asyncio.create_task(store.vacuum(retention_seconds=0))
            await asyncio.sleep(0.1)
            assert not task.done()
        await task


@pytest.mark.asyncio
async def test_tag_writes_raise_when_read_only(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("release-1")

    async with Store(temp_db_path, read_only=True) as store:
        with pytest.raises(ReadOnlyError):
            await store.create_tag("release-2")
        with pytest.raises(ReadOnlyError):
            await store.delete_tag("release-1")

        tags = await store.list_tags()
        assert tags["release-1"].complete is True


@pytest.mark.asyncio
async def test_current_table_versions_returns_versions(temp_db_path):
    """current_table_versions returns dict of table versions."""
    async with Store(temp_db_path, create=True) as store:
        versions = await store.current_table_versions()

        assert "documents" in versions
        assert "chunks" in versions
        assert "settings" in versions
        assert all(isinstance(v, int) for v in versions.values())


@pytest.mark.asyncio
async def test_list_table_versions_returns_history(temp_db_path):
    """list_table_versions returns version history for a table."""
    async with Store(temp_db_path, create=True) as store:
        versions = await store.list_table_versions("documents")

        assert len(versions) >= 1
        for v in versions:
            assert "version" in v
            assert "timestamp" in v

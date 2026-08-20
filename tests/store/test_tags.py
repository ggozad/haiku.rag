import asyncio

import pytest
from lancedb.table import AsyncTags

from haiku.rag.store import ReadOnlyError
from haiku.rag.store.engine import Store
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.schema import REQUIRED_TABLES


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
async def test_vacuum_reraises_runtime_error(temp_db_path, monkeypatch):
    """Vacuum suppresses OSError only; lance errors (RuntimeError) surface
    instead of silently skipping cleanup."""
    from lancedb.table import AsyncTable

    async with Store(temp_db_path, create=True) as store:

        async def failing_optimize(self, **kwargs):
            raise RuntimeError("lance error: boom")

        monkeypatch.setattr(AsyncTable, "optimize", failing_optimize)
        with pytest.raises(RuntimeError, match="boom"):
            await store.vacuum(retention_seconds=0)

        async def failing_optimize_os(self, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(AsyncTable, "optimize", failing_optimize_os)
        await store.vacuum(retention_seconds=0)


@pytest.mark.asyncio
async def test_vacuum_multiple_tags_uses_oldest_cutoff(temp_db_path):
    """With several tags the retention clamp must key off the oldest one;
    clamping to a newer tag would put the older tagged version inside the
    cleanup window and lance would hard-error."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        await store.create_tag("old")

        await asyncio.sleep(1.5)

        await repo.create(Document(content="Second document"))
        await store.create_tag("new")

        await store.vacuum(retention_seconds=0)

        tags = await store.list_tags()
        remaining = [v["version"] for v in await store.list_table_versions("documents")]
        assert tags["old"].tables["documents"] in remaining
        assert tags["new"].tables["documents"] in remaining


@pytest.mark.asyncio
async def test_vacuum_partial_tag_protects_its_tables(temp_db_path):
    """A partial tag still protects the versions of the tables it exists on,
    while untagged tables clean up normally."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))

        chunks_version = await store.chunks_table.version()
        await store.chunks_table.tags.create("stale", chunks_version)
        docs_versions_before = [
            v["version"] for v in await store.list_table_versions("documents")
        ]

        await asyncio.sleep(1.5)

        await repo.create(Document(content="Second document"))
        await store.vacuum(retention_seconds=0)

        chunk_versions = [
            v["version"] for v in await store.list_table_versions("chunks")
        ]
        assert chunks_version in chunk_versions

        docs_versions_after = [
            v["version"] for v in await store.list_table_versions("documents")
        ]
        assert min(docs_versions_before) not in docs_versions_after


@pytest.mark.asyncio
async def test_deleting_oldest_tag_advances_cleanup(temp_db_path):
    """Versions pinned by a tag become cleanable once the tag is deleted;
    the cleanup cutoff advances to the next retained tag without removing
    its version."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        await store.create_tag("old")
        old_version = (await store.list_tags())["old"].tables["documents"]

        await asyncio.sleep(1.5)

        await repo.create(Document(content="Second document"))
        await store.create_tag("new")
        new_version = (await store.list_tags())["new"].tables["documents"]

        await store.vacuum(retention_seconds=0)
        remaining = [v["version"] for v in await store.list_table_versions("documents")]
        assert old_version in remaining
        assert new_version in remaining

        await store.delete_tag("old")
        await asyncio.sleep(1.5)
        await store.vacuum(retention_seconds=0)

        remaining = [v["version"] for v in await store.list_table_versions("documents")]
        assert old_version not in remaining
        assert new_version in remaining


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


@pytest.mark.asyncio
async def test_delete_tag_reports_listing_failures(temp_db_path, monkeypatch):
    """A tags.list() failure mid-delete is reported with the table named and
    a recovery hint, instead of escaping raw after earlier deletions."""
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("release-1")

        real_list = AsyncTags.list
        calls = {"n": 0}

        async def flaky_list(self):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("list boom")
            return await real_list(self)

        monkeypatch.setattr(AsyncTags, "list", flaky_list)

        with pytest.raises(RuntimeError) as exc_info:
            await store.delete_tag("release-1")

        msg = str(exc_info.value)
        assert "document_meta" in msg
        assert "retry delete_tag" in msg

        monkeypatch.undo()
        tags = await store.list_tags()
        assert set(tags["release-1"].tables) == {"document_meta"}

        await store.delete_tag("release-1")
        assert await store.list_tags() == {}


@pytest.mark.asyncio
async def test_create_tag_cancellation_cleans_up(temp_db_path, monkeypatch):
    """Cancellation during per-table tag creation must not leave a partial
    tag behind: cleanup runs before the cancellation propagates."""
    async with Store(temp_db_path, create=True) as store:
        real_create = AsyncTags.create
        calls = {"n": 0}

        async def cancelled_create(self, name: str, version: int) -> None:
            calls["n"] += 1
            if calls["n"] == 4:
                raise asyncio.CancelledError()
            await real_create(self, name, version)

        monkeypatch.setattr(AsyncTags, "create", cancelled_create)

        with pytest.raises(asyncio.CancelledError):
            await store.create_tag("broken")

        monkeypatch.undo()
        assert await store.list_tags() == {}


@pytest.mark.asyncio
async def test_create_tag_cleanup_survives_cancellation(temp_db_path, monkeypatch):
    """Cancelling create_tag while it cleans up a failed creation does not
    interrupt the cleanup: no partial tag remains and the cancellation is
    delivered afterwards."""
    async with Store(temp_db_path, create=True) as store:
        real_create = AsyncTags.create
        real_delete = AsyncTags.delete
        create_calls = {"n": 0}
        cleanup_started = asyncio.Event()
        release = asyncio.Event()

        async def flaky_create(self, name: str, version: int) -> None:
            create_calls["n"] += 1
            if create_calls["n"] == 4:
                raise RuntimeError("create boom")
            await real_create(self, name, version)

        async def slow_delete(self, name: str) -> None:
            cleanup_started.set()
            await release.wait()
            await real_delete(self, name)

        monkeypatch.setattr(AsyncTags, "create", flaky_create)
        monkeypatch.setattr(AsyncTags, "delete", slow_delete)

        task = asyncio.create_task(store.create_tag("broken"))
        await cleanup_started.wait()
        task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        monkeypatch.undo()
        assert await store.list_tags() == {}


@pytest.mark.asyncio
async def test_create_tag_cancellation_after_commit_cleans_committed_tag(
    temp_db_path, monkeypatch
):
    """Cancellation arriving after lance committed a table's tag but before
    the attempt recorded it must still clean that table: cleanup sweeps all
    tables, relying on the preflight guarantee that the name was unused."""
    async with Store(temp_db_path, create=True) as store:
        real_create = AsyncTags.create
        calls = {"n": 0}

        async def committing_cancelled_create(self, name: str, version: int) -> None:
            calls["n"] += 1
            await real_create(self, name, version)
            if calls["n"] == 4:
                raise asyncio.CancelledError()

        monkeypatch.setattr(AsyncTags, "create", committing_cancelled_create)

        with pytest.raises(asyncio.CancelledError):
            await store.create_tag("broken")

        monkeypatch.undo()
        assert await store.list_tags() == {}

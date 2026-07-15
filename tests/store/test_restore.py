import re

import pytest
from lancedb.table import AsyncTable, AsyncTags

from haiku.rag.store import ReadOnlyError, Store
from haiku.rag.store.engine import RESTORE_TABLE_ORDER
from haiku.rag.store.models import Document
from haiku.rag.store.repositories.document import DocumentRepository

SAFETY_TAG_PATTERN = r"before-restore-\d{8}T\d{6}Z"


async def _doc_contents(store: Store) -> set[str]:
    docs = await DocumentRepository(store).list_all(include_content=True)
    return {d.content for d in docs}


@pytest.mark.asyncio
async def test_restore_tag_restores_all_tables(temp_db_path):
    """A complete tag restores every table; rows added after the tag are
    absent from the restored latest state, which stays writable."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        await store.create_tag("release-1")
        await repo.create(Document(content="Second document"))
        pre_restore_docs_version = await store.documents_table.version()

        safety_tag = await store.restore_tag("release-1")

        assert re.fullmatch(SAFETY_TAG_PATTERN, safety_tag)
        assert await _doc_contents(store) == {"First document"}

        # restore writes a NEW latest version; the table is not a read-only
        # checkout and stays writable.
        assert await store.documents_table.version() > pre_restore_docs_version
        await repo.create(Document(content="Third document"))
        assert await _doc_contents(store) == {"First document", "Third document"}


@pytest.mark.asyncio
async def test_restore_safety_tag_matches_pre_restore_state(temp_db_path):
    """The safety tag records the exact pre-restore version map, and
    restoring it returns the database to its prior logical state."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        await store.create_tag("release-1")
        await repo.create(Document(content="Second document"))
        snapshot = await store.current_table_versions()

        safety_tag = await store.restore_tag("release-1")

        tags = await store.list_tags()
        assert tags[safety_tag].complete is True
        assert tags[safety_tag].tables == snapshot

        await store.restore_tag(safety_tag)
        assert await _doc_contents(store) == {"First document", "Second document"}


@pytest.mark.asyncio
async def test_restore_missing_tag_makes_no_changes(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await DocumentRepository(store).create(Document(content="First document"))
        versions = await store.current_table_versions()

        with pytest.raises(ValueError, match="does not exist"):
            await store.restore_tag("nope")

        assert await store.current_table_versions() == versions
        assert await store.list_tags() == {}


@pytest.mark.asyncio
async def test_restore_partial_tag_makes_no_changes(temp_db_path):
    """A partial tag can never be restored; the error lists every missing
    table and no safety tag is created."""
    async with Store(temp_db_path, create=True) as store:
        version = await store.chunks_table.version()
        await store.chunks_table.tags.create("stale", version)
        versions = await store.current_table_versions()

        with pytest.raises(ValueError) as exc_info:
            await store.restore_tag("stale")

        msg = str(exc_info.value)
        for table_name in ("documents", "document_meta", "document_items", "settings"):
            assert table_name in msg

        assert await store.current_table_versions() == versions
        assert set(await store.list_tags()) == {"stale"}


@pytest.mark.asyncio
async def test_restore_safety_tag_name_collision(temp_db_path, monkeypatch):
    """A colliding safety-tag name gets a numeric suffix."""
    import haiku.rag.store.engine as engine_mod

    class FixedDatetime:
        @staticmethod
        def now(tz=None):
            from datetime import UTC, datetime

            return datetime(2026, 7, 15, 14, 30, 12, tzinfo=UTC)

    monkeypatch.setattr(engine_mod, "datetime", FixedDatetime)

    async with Store(temp_db_path, create=True) as store:
        await DocumentRepository(store).create(Document(content="First document"))
        await store.create_tag("release-1")
        await store.create_tag("before-restore-20260715T143012Z")

        safety_tag = await store.restore_tag("release-1")
        assert safety_tag == "before-restore-20260715T143012Z-2"


@pytest.mark.asyncio
async def test_restore_safety_tag_failure_leaves_state_untouched(
    temp_db_path, monkeypatch
):
    """If the safety tag cannot be created, restore never begins."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        await store.create_tag("release-1")
        await repo.create(Document(content="Second document"))
        versions = await store.current_table_versions()

        async def failing_create(self, name: str, version: int) -> None:
            raise RuntimeError("tag boom")

        monkeypatch.setattr(AsyncTags, "create", failing_create)

        with pytest.raises(RuntimeError) as exc_info:
            await store.restore_tag("release-1")

        msg = str(exc_info.value)
        assert "did not begin" in msg
        assert "No table was changed" in msg
        assert "tag boom" in msg
        assert exc_info.value.__cause__ is not None

        monkeypatch.undo()
        assert await store.current_table_versions() == versions
        assert await _doc_contents(store) == {"First document", "Second document"}
        assert set(await store.list_tags()) == {"release-1"}


@pytest.mark.asyncio
async def test_restore_midway_failure_rolls_back(temp_db_path, monkeypatch):
    """A restore failure after some tables were restored rolls every table
    back to the pre-restore snapshot; the error names the failed table and
    the safety tag."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        await store.create_tag("release-1")
        await repo.create(Document(content="Second document"))

        real_restore = AsyncTable.restore
        calls = {"n": 0}

        async def flaky_restore(self, version=None):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("restore boom")
            return await real_restore(self, version)

        monkeypatch.setattr(AsyncTable, "restore", flaky_restore)

        with pytest.raises(RuntimeError) as exc_info:
            await store.restore_tag("release-1")

        msg = str(exc_info.value)
        assert RESTORE_TABLE_ORDER[2] in msg
        assert "rolled back" in msg
        assert "before-restore-" in msg

        monkeypatch.undo()
        assert await _doc_contents(store) == {"First document", "Second document"}
        assert any(t.startswith("before-restore-") for t in await store.list_tags())


@pytest.mark.asyncio
async def test_restore_rollback_failure_reports_inconsistency(
    temp_db_path, monkeypatch
):
    """When rollback also fails, the error lists the failed tables, names
    the safety tag, and states manual recovery is required."""
    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        await store.create_tag("release-1")
        await repo.create(Document(content="Second document"))

        real_restore = AsyncTable.restore
        calls = {"n": 0}

        async def flaky_restore(self, version=None):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise RuntimeError("restore boom")
            return await real_restore(self, version)

        monkeypatch.setattr(AsyncTable, "restore", flaky_restore)

        with pytest.raises(RuntimeError) as exc_info:
            await store.restore_tag("release-1")

        msg = str(exc_info.value)
        assert "inconsistent" in msg
        assert "manual recovery" in msg
        assert "before-restore-" in msg
        for table_name in RESTORE_TABLE_ORDER:
            assert table_name in msg


@pytest.mark.asyncio
async def test_restore_cancellation_rolls_back(temp_db_path, monkeypatch):
    """Cancellation mid-restore must not bypass rollback: the tables return
    to the pre-restore snapshot and the cancellation re-raises."""
    import asyncio

    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        await store.create_tag("release-1")
        await repo.create(Document(content="Second document"))

        real_restore = AsyncTable.restore
        calls = {"n": 0}

        async def cancelled_restore(self, version=None):
            calls["n"] += 1
            if calls["n"] == 3:
                raise asyncio.CancelledError()
            return await real_restore(self, version)

        monkeypatch.setattr(AsyncTable, "restore", cancelled_restore)

        with pytest.raises(asyncio.CancelledError):
            await store.restore_tag("release-1")

        monkeypatch.undo()
        assert await _doc_contents(store) == {"First document", "Second document"}
        assert any(t.startswith("before-restore-") for t in await store.list_tags())


@pytest.mark.asyncio
async def test_restore_cancellation_with_failed_rollback_reports(
    temp_db_path, monkeypatch
):
    """If rollback after a cancellation also fails, the manual-recovery
    error is raised instead of the bare cancellation."""
    import asyncio

    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        await store.create_tag("release-1")
        await repo.create(Document(content="Second document"))

        real_restore = AsyncTable.restore
        calls = {"n": 0}

        async def broken_restore(self, version=None):
            calls["n"] += 1
            if calls["n"] < 3:
                return await real_restore(self, version)
            if calls["n"] == 3:
                raise asyncio.CancelledError()
            raise RuntimeError("restore boom")

        monkeypatch.setattr(AsyncTable, "restore", broken_restore)

        with pytest.raises(RuntimeError) as exc_info:
            await store.restore_tag("release-1")

        msg = str(exc_info.value)
        assert "cancel" in msg.lower()
        assert "manual recovery" in msg
        assert "before-restore-" in msg


@pytest.mark.asyncio
async def test_restore_read_only_raises(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("release-1")

    async with Store(temp_db_path, read_only=True) as store:
        with pytest.raises(ReadOnlyError):
            await store.restore_tag("release-1")


@pytest.mark.asyncio
async def test_restore_rejected_during_rebuild(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await store.create_tag("release-1")

        async with store._rebuild_lock:
            with pytest.raises(ValueError, match="[Rr]ebuild in progress"):
                await store.restore_tag("release-1")


@pytest.mark.asyncio
async def test_restore_old_version_marker_requires_explicit_migration(temp_db_path):
    """Restore never migrates: restoring a tag whose settings carry an old
    version marker completes, the next normal open hits the migration gate,
    explicit migration works, and the safety tag remains usable after it."""
    from haiku.rag.store.exceptions import MigrationRequiredError

    async with Store(temp_db_path, create=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="First document"))
        current_version = await store.get_haiku_version()
        await store.set_haiku_version("0.63.0")
        await store.create_tag("old-marker")
        await store.set_haiku_version(current_version)
        await repo.create(Document(content="Second document"))

    async with Store(temp_db_path) as store:
        safety_tag = await store.restore_tag("old-marker")
        assert await store.get_haiku_version() == "0.63.0"
        assert await _doc_contents(store) == {"First document"}

    with pytest.raises(MigrationRequiredError):
        async with Store(temp_db_path):
            pass

    async with Store(temp_db_path, skip_migration_check=True) as store:
        await store.migrate()

    async with Store(temp_db_path) as store:
        assert await _doc_contents(store) == {"First document"}
        await store.restore_tag(safety_tag)
        assert await _doc_contents(store) == {"First document", "Second document"}

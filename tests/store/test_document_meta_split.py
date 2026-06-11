import lancedb
import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.exceptions import MigrationRequiredError
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository
from tests.store.legacy_documents import (
    LegacyDocumentRecord,
    seed_legacy_documents,
)


@pytest.mark.asyncio
async def test_create_rolls_back_meta_when_documents_write_fails(temp_db_path):
    """A failed documents write must not leave an orphan document_meta row that
    list_all/count would surface (they read document_meta). The rollback must be
    targeted — it deletes only the failed row, leaving other documents intact."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        repo = DocumentRepository(store)

        # A pre-existing document that must survive the failed create's rollback.
        good = await repo.create(Document(content="keep", uri="mem://keep"))
        assert good.id is not None

        original_add = store.documents_table.add

        async def boom(*_args, **_kwargs):
            raise RuntimeError("documents write failed")

        store.documents_table.add = boom
        with pytest.raises(RuntimeError, match="documents write failed"):
            await repo.create(Document(content="x", uri="mem://ghost"))
        store.documents_table.add = original_add

        # The ghost's meta row was deleted; the good document is untouched.
        assert await repo.count() == 1
        assert [d.id for d in await repo.list_all()] == [good.id]
        assert await store.document_meta_table.count_rows() == 1
        assert await repo.get_by_uri("mem://ghost") is None
        fetched = await repo.get_by_id(good.id)
        assert fetched is not None and fetched.uri == "mem://keep"


@pytest.mark.asyncio
async def test_update_missing_id_does_not_create_ghost(temp_db_path):
    """update()/update_meta() for an id with no documents row must not insert a
    document_meta row — otherwise it would show up in list_all/count while
    get_by_id returns None."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        repo = DocumentRepository(store)

        await repo.update(Document(id="missing", content="x", uri="u"))

        assert await repo.count() == 0
        assert await repo.list_all() == []
        assert await store.documents_table.count_rows() == 0
        assert await store.document_meta_table.count_rows() == 0
        assert await repo.get_by_id("missing") is None


@pytest.mark.asyncio
async def test_opening_legacy_db_raises_migration_without_mutating(temp_db_path):
    """Opening a pre-0.58 DB (no document_meta) must raise MigrationRequiredError
    up front — in both writable and read-only mode — and must not mutate the DB
    by creating the new table on open."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await seed_legacy_documents(
            store,
            [LegacyDocumentRecord(id="d", content="x", uri="u", metadata="{}")],
        )
        # A real pre-0.58 DB has no document_meta table.
        await store.db.drop_table("document_meta")
        await store.set_haiku_version("0.56.0")

    # Writable open: pending migration surfaces before any table creation.
    with pytest.raises(MigrationRequiredError):
        async with Store(temp_db_path):
            pass

    # Read-only open: must also be MigrationRequiredError (not ReadOnlyError).
    with pytest.raises(MigrationRequiredError):
        async with Store(temp_db_path, read_only=True):
            pass

    # The failed opens did not create document_meta.
    raw = await lancedb.connect_async(str(temp_db_path))
    assert "document_meta" not in (await raw.list_tables()).tables


@pytest.mark.asyncio
async def test_migrate_creates_and_populates_document_meta(temp_db_path):
    """The migrate path (skip_migration_check) still creates and fills
    document_meta for a legacy DB."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await seed_legacy_documents(
            store,
            [LegacyDocumentRecord(id="d", content="x", uri="u", metadata="{}")],
        )
        await store.db.drop_table("document_meta")
        await store.set_haiku_version("0.56.0")

    async with Store(temp_db_path, skip_migration_check=True) as store:
        await store.migrate()
        assert "document_meta" in (await store.db.list_tables()).tables
        repo = DocumentRepository(store)
        doc = await repo.get_by_id("d")
        assert doc is not None and doc.uri == "u"

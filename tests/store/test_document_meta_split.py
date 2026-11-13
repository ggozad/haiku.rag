import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.document import DocumentRepository


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


async def test_repository_empty_and_missing_id_paths(temp_db_path):
    """Cover the repository's early-return branches for empty input and
    missing ids, plus get_content."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        repo = DocumentRepository(store)

        assert await repo.create([]) == []
        assert await repo.get_content("nope") is None
        assert await repo.get_docling_data("nope") is None
        assert await repo.get_pages_data("nope") is None
        assert await repo.delete("nope") is False

        doc = await repo.create(Document(content="hello body", uri="u1"))
        assert doc.id is not None
        assert await repo.get_content(doc.id) == "hello body"


async def test_batch_create_rolls_back_meta_on_failure(temp_db_path, monkeypatch):
    """The list create path also rolls back its document_meta rows if the
    documents write fails."""
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        repo = DocumentRepository(store)

        async def boom(*_a, **_k):
            raise RuntimeError("documents add failed")

        monkeypatch.setattr(store.documents_table, "add", boom)
        with pytest.raises(RuntimeError, match="documents add failed"):
            await repo.create(
                [Document(content="x", uri="u1"), Document(content="y", uri="u2")]
            )
        monkeypatch.undo()

        assert await store.documents_table.count_rows() == 0
        assert await store.document_meta_table.count_rows() == 0


async def test_get_by_uri_with_orphan_meta_returns_none(temp_db_path):
    """Defensive: a document_meta row whose documents row is missing (an
    invariant violation) resolves to None, not a half-hydrated document."""
    from haiku.rag.store.schema import DocumentMetaRecord

    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        repo = DocumentRepository(store)
        await store.document_meta_table.add(
            [DocumentMetaRecord(id="ghost", uri="u-ghost", metadata="{}")]
        )
        assert await repo.get_by_uri("u-ghost") is None


async def test_update_meta_all_writes_one_version(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        repo = DocumentRepository(store)
        docs = [
            await repo.create(Document(content=f"c{n}", uri=f"mem://{n}"))
            for n in range(3)
        ]
        before = await store.document_meta_table.version()

        for doc in docs:
            doc.metadata = {"tag": "batch"}
        await repo.update_meta_all(docs)

        assert await store.document_meta_table.version() - before == 1
        assert all(d.metadata == {"tag": "batch"} for d in await repo.list_all())


async def test_update_meta_all_writes_nothing_for_no_documents(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        repo = DocumentRepository(store)
        await repo.create(Document(content="c", uri="mem://a"))
        before = await store.document_meta_table.version()

        assert await repo.update_meta_all([]) == []

        assert await store.document_meta_table.version() == before

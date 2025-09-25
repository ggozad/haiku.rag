import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.utils import text_to_docling_document


@pytest.mark.asyncio
async def test_version_rollback_on_create_failure(temp_db_path):
    store = Store(temp_db_path)
    repo = DocumentRepository(store)

    # Ensure chunk repository is instantiated and stub embeddings to avoid network
    dim = repo.chunk_repository.embedder._vector_dim

    async def fake_embed(x):  # type: ignore[no-redef]
        if isinstance(x, list):
            return [[0.0] * dim for _ in x]
        return [0.0] * dim

    repo.chunk_repository.embedder.embed = fake_embed  # type: ignore[assignment]

    # Patch create_chunks_for_document to succeed then fail, triggering rollback
    orig = repo.chunk_repository.create_chunks_for_document

    async def succeed_then_fail(document_id, dl_doc):  # noqa: ARG001
        await orig(document_id, dl_doc)
        raise RuntimeError("boom")

    repo.chunk_repository.create_chunks_for_document = succeed_then_fail  # type: ignore[assignment]

    # Attempt to create document with chunks; expect failure and rollback
    content = "Hello, rollback!"
    doc = Document(content=content)
    dl_doc = text_to_docling_document(content, name="test.md")

    with pytest.raises(RuntimeError):
        await repo._create_with_docling(doc, dl_doc)

    # State should be restored (no documents/chunks)
    docs = await repo.list_all()
    assert len(docs) == 0
    chunks_repo = ChunkRepository(store)
    all_chunks = await chunks_repo.list_all()
    assert len(all_chunks) == 0


@pytest.mark.asyncio
async def test_version_rollback_on_update_failure(temp_db_path):
    store = Store(temp_db_path)
    repo = DocumentRepository(store)

    # Stub embeddings to avoid network
    dim = repo.chunk_repository.embedder._vector_dim

    async def fake_embed(x):  # type: ignore[no-redef]
        if isinstance(x, list):
            return [[0.0] * dim for _ in x]
        return [0.0] * dim

    repo.chunk_repository.embedder.embed = fake_embed  # type: ignore[assignment]

    # Create a valid document first (with real chunking and stubbed embeddings)
    base_content = "Base content"
    base_doc = Document(content=base_content)
    base_dl = text_to_docling_document(base_content, name="base.md")
    created = await repo._create_with_docling(base_doc, base_dl)

    # Force new chunk creation to fail during update after writing
    orig = repo.chunk_repository.create_chunks_for_document

    async def succeed_then_fail(document_id, dl_doc):  # noqa: ARG001
        await orig(document_id, dl_doc)
        raise RuntimeError("update fail")

    repo.chunk_repository.create_chunks_for_document = succeed_then_fail  # type: ignore[assignment]

    # Attempt update
    updated_content = "Updated content"
    created.content = updated_content
    updated_dl = text_to_docling_document(updated_content, name="updated.md")

    with pytest.raises(RuntimeError):
        await repo._update_with_docling(created, updated_dl)

    # Content and chunks should remain the original
    persisted = await repo.get_by_id(created.id)  # type: ignore[arg-type]
    assert persisted is not None
    assert persisted.content == base_content
    chunks_repo = ChunkRepository(store)
    original_chunks = await chunks_repo.get_by_document_id(created.id)  # type: ignore[arg-type]
    assert len(original_chunks) > 0


def test_new_database_does_not_run_upgrades(monkeypatch, temp_db_path):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_pending_upgrades should not be called for new DB")

    monkeypatch.setattr(
        "haiku.rag.store.upgrades.run_pending_upgrades",
        fail_if_called,
    )

    Store(temp_db_path)


def test_existing_database_runs_upgrades(monkeypatch, temp_db_path):
    Store(temp_db_path)

    called = {"value": False}

    def mark_called(*_args, **_kwargs):
        called["value"] = True

    monkeypatch.setattr(
        "haiku.rag.store.upgrades.run_pending_upgrades",
        mark_called,
    )

    Store(temp_db_path)

    assert called["value"]

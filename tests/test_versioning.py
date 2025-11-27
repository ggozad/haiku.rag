import pytest

from haiku.rag.config import Config
from haiku.rag.converters import get_converter
from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository


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
    converter = get_converter(Config)
    dl_doc = await converter.convert_text(content, name="test.md")

    with pytest.raises(RuntimeError):
        await repo._create_and_chunk(doc, dl_doc)

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
    converter = get_converter(Config)
    base_dl = await converter.convert_text(base_content, name="base.md")
    created = await repo._create_and_chunk(base_doc, base_dl)

    # Force new chunk creation to fail during update after writing
    orig = repo.chunk_repository.create_chunks_for_document

    async def succeed_then_fail(document_id, dl_doc):  # noqa: ARG001
        await orig(document_id, dl_doc)
        raise RuntimeError("update fail")

    repo.chunk_repository.create_chunks_for_document = succeed_then_fail  # type: ignore[assignment]

    # Attempt update
    updated_content = "Updated content"
    created.content = updated_content
    updated_dl = await converter.convert_text(updated_content, name="updated.md")

    with pytest.raises(RuntimeError):
        await repo._update_and_rechunk(created, updated_dl)

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


@pytest.mark.asyncio
async def test_vacuum_with_retention_threshold(temp_db_path):
    store = Store(temp_db_path)
    repo = DocumentRepository(store)

    # Stub embeddings to avoid network
    dim = repo.chunk_repository.embedder._vector_dim

    async def fake_embed(x):  # type: ignore[no-redef]
        if isinstance(x, list):
            return [[0.0] * dim for _ in x]
        return [0.0] * dim

    repo.chunk_repository.embedder.embed = fake_embed  # type: ignore[assignment]

    # Create first document
    converter = get_converter(Config)
    doc1 = Document(content="First document")
    dl_doc1 = await converter.convert_text("First document", name="doc1.md")
    await repo._create_and_chunk(doc1, dl_doc1)

    # Create second document
    doc2 = Document(content="Second document")
    dl_doc2 = await converter.convert_text("Second document", name="doc2.md")
    await repo._create_and_chunk(doc2, dl_doc2)

    # Get initial version counts (should have multiple versions from creates)
    initial_doc_versions = len(list(store.documents_table.list_versions()))
    initial_chunk_versions = len(list(store.chunks_table.list_versions()))

    assert initial_doc_versions > 1, "Should have multiple document table versions"
    assert initial_chunk_versions > 1, "Should have multiple chunk table versions"

    # Vacuum with default threshold (60 seconds) - should keep recent versions
    # Note: vacuum may create new versions even when not cleaning up old ones
    await store.vacuum()

    after_default_doc_versions = len(list(store.documents_table.list_versions()))
    after_default_chunk_versions = len(list(store.chunks_table.list_versions()))

    # After vacuum with retention, version count should stay the same or increase
    # (optimize may create new versions) but not decrease
    assert after_default_doc_versions >= initial_doc_versions, (
        "Default vacuum should not remove recent versions"
    )
    assert after_default_chunk_versions >= initial_chunk_versions, (
        "Default vacuum should not remove recent versions"
    )

    # Vacuum with 0 threshold - should significantly reduce versions
    await store.vacuum(retention_seconds=0)

    after_zero_doc_versions = len(list(store.documents_table.list_versions()))
    after_zero_chunk_versions = len(list(store.chunks_table.list_versions()))

    # After aggressive vacuum, should have minimal versions (1-2)
    # Note: optimize operation may create a version after cleanup
    assert after_zero_doc_versions <= 2, (
        f"Should have minimal document versions after vacuum(0), got {after_zero_doc_versions}"
    )
    assert after_zero_chunk_versions <= 2, (
        f"Should have minimal chunk versions after vacuum(0), got {after_zero_chunk_versions}"
    )

    # And it should be significantly fewer than before
    assert after_zero_doc_versions < initial_doc_versions, (
        "Should have fewer versions after vacuum(0)"
    )
    assert after_zero_chunk_versions < initial_chunk_versions, (
        "Should have fewer versions after vacuum(0)"
    )


@pytest.mark.asyncio
async def test_vacuum_completes_before_context_exit(temp_db_path, monkeypatch):
    """Test that background vacuum completes when context manager exits."""
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import Config

    # Set aggressive vacuum retention for this test
    monkeypatch.setattr(Config.storage, "vacuum_retention_seconds", 0)

    async with HaikuRAG(db_path=temp_db_path) as client:
        # Create multiple documents - each creation triggers automatic vacuum with retention=0
        # This aggressively cleans up old versions between operations
        for i in range(3):
            await client.create_document(content=f"Test document {i}")

    # After context exit, automatic vacuum should have kept versions minimal
    store = Store(temp_db_path)
    final_versions = len(list(store.documents_table.list_versions()))

    # With retention_seconds=0, vacuum aggressively cleans up between operations
    # Should have very few versions remaining (1-2)
    assert final_versions <= 2, (
        f"Aggressive vacuum should keep minimal versions, got {final_versions}"
    )
    assert final_versions >= 1, "Should have at least one version remaining"
    store.close()

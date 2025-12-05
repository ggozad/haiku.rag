import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.store.engine import Store


@pytest.mark.asyncio
async def test_version_rollback_on_create_failure(temp_db_path):
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Patch chunk_repository.create to succeed then fail, triggering rollback
        orig_create = client.chunk_repository.create

        async def succeed_then_fail(chunks):
            await orig_create(chunks)
            raise RuntimeError("boom")

        client.chunk_repository.create = succeed_then_fail  # type: ignore[method-assign]

        # Attempt to create document; expect failure and rollback
        content = "Hello, rollback!"

        with pytest.raises(RuntimeError):
            await client.create_document(content=content)

        # State should be restored (no documents/chunks)
        docs = await client.list_documents()
        assert len(docs) == 0
        all_chunks = await client.chunk_repository.list_all()
        assert len(all_chunks) == 0


@pytest.mark.asyncio
async def test_version_rollback_on_update_failure(temp_db_path):
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Create a valid document first
        base_content = "Base content"
        created = await client.create_document(content=base_content)

        # Patch chunk_repository.create to succeed then fail during update
        orig_create = client.chunk_repository.create

        async def succeed_then_fail(chunks):
            await orig_create(chunks)
            raise RuntimeError("update fail")

        client.chunk_repository.create = succeed_then_fail  # type: ignore[method-assign]

        # Attempt update
        with pytest.raises(RuntimeError):
            await client.update_document(
                document_id=created.id,  # type: ignore[arg-type]
                content="Updated content",
            )

        # Content and chunks should remain the original
        persisted = await client.get_document_by_id(created.id)  # type: ignore[arg-type]
        assert persisted is not None
        assert persisted.content == base_content
        original_chunks = await client.chunk_repository.get_by_document_id(created.id)  # type: ignore[arg-type]
        assert len(original_chunks) > 0


def test_new_database_does_not_run_upgrades(monkeypatch, temp_db_path):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_pending_upgrades should not be called for new DB")

    monkeypatch.setattr(
        "haiku.rag.store.upgrades.run_pending_upgrades",
        fail_if_called,
    )

    Store(temp_db_path, create=True)


def test_existing_database_runs_upgrades(monkeypatch, temp_db_path):
    Store(temp_db_path, create=True)

    called = {"value": False}

    def mark_called(*_args, **_kwargs):
        called["value"] = True

    monkeypatch.setattr(
        "haiku.rag.store.upgrades.run_pending_upgrades",
        mark_called,
    )

    # Opening an existing database should trigger upgrades
    Store(temp_db_path)

    assert called["value"]


@pytest.mark.asyncio
async def test_vacuum_with_retention_threshold(temp_db_path):
    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Create first document
        await client.create_document(content="First document")

        # Create second document
        await client.create_document(content="Second document")

        store = client.store

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
    from haiku.rag.config import Config

    # Set aggressive vacuum retention for this test
    monkeypatch.setattr(Config.storage, "vacuum_retention_seconds", 0)

    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Create multiple documents - each creation triggers automatic vacuum with retention=0
        # This aggressively cleans up old versions between operations
        for i in range(3):
            await client.create_document(content=f"Test document {i}")

    # After context exit, automatic vacuum should have kept versions minimal
    store = Store(temp_db_path, create=True)
    final_versions = len(list(store.documents_table.list_versions()))

    # With retention_seconds=0, vacuum aggressively cleans up between operations
    # Should have very few versions remaining (1-2)
    assert final_versions <= 2, (
        f"Aggressive vacuum should keep minimal versions, got {final_versions}"
    )
    assert final_versions >= 1, "Should have at least one version remaining"
    store.close()

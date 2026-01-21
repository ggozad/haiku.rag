from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.store import ReadOnlyError, Store
from haiku.rag.store.models import Chunk, Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.settings import SettingsRepository


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent / "cassettes" / "test_read_only")


class TestReadOnlyError:
    def test_read_only_error_is_exception(self):
        """ReadOnlyError should be a subclass of Exception."""
        assert issubclass(ReadOnlyError, Exception)

    def test_read_only_error_can_be_raised(self):
        """ReadOnlyError can be raised and caught."""
        with pytest.raises(ReadOnlyError) as exc_info:
            raise ReadOnlyError("Cannot modify database in read-only mode")
        assert "read-only" in str(exc_info.value)


class TestStoreReadOnly:
    def test_store_default_is_not_read_only(self, temp_db_path):
        """Store defaults to not read-only."""
        store = Store(temp_db_path, create=True)
        assert store.is_read_only is False
        store.close()

    def test_store_can_be_created_read_only(self, temp_db_path):
        """Store can be created with read_only=True."""
        # First create a normal store to initialize the database
        store = Store(temp_db_path, create=True)
        store.close()

        # Now open in read-only mode
        store = Store(temp_db_path, read_only=True)
        assert store.is_read_only is True
        store.close()

    def test_assert_writable_raises_when_read_only(self, temp_db_path):
        """_assert_writable() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        store.close()

        store = Store(temp_db_path, read_only=True)
        with pytest.raises(ReadOnlyError):
            store._assert_writable()
        store.close()

    def test_assert_writable_passes_when_not_read_only(self, temp_db_path):
        """_assert_writable() does not raise when read_only=False."""
        store = Store(temp_db_path, create=True)
        store._assert_writable()  # Should not raise
        store.close()

    @pytest.mark.asyncio
    async def test_vacuum_raises_when_read_only(self, temp_db_path):
        """vacuum() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        store.close()

        store = Store(temp_db_path, read_only=True)
        with pytest.raises(ReadOnlyError):
            await store.vacuum()
        store.close()

    def test_set_haiku_version_raises_when_read_only(self, temp_db_path):
        """set_haiku_version() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        store.close()

        store = Store(temp_db_path, read_only=True)
        with pytest.raises(ReadOnlyError):
            store.set_haiku_version("1.0.0")
        store.close()

    def test_recreate_embeddings_table_raises_when_read_only(self, temp_db_path):
        """recreate_embeddings_table() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        store.close()

        store = Store(temp_db_path, read_only=True)
        with pytest.raises(ReadOnlyError):
            store.recreate_embeddings_table()
        store.close()

    def test_restore_table_versions_raises_when_read_only(self, temp_db_path):
        """restore_table_versions() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        versions = store.current_table_versions()
        store.close()

        store = Store(temp_db_path, read_only=True)
        with pytest.raises(ReadOnlyError):
            store.restore_table_versions(versions)
        store.close()


class TestDocumentRepositoryReadOnly:
    @pytest.mark.asyncio
    async def test_create_raises_when_read_only(self, temp_db_path):
        """DocumentRepository.create() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        store.close()

        store = Store(temp_db_path, read_only=True)
        repo = DocumentRepository(store)
        doc = Document(content="test content")

        with pytest.raises(ReadOnlyError):
            await repo.create(doc)
        store.close()

    @pytest.mark.asyncio
    async def test_update_raises_when_read_only(self, temp_db_path):
        """DocumentRepository.update() raises ReadOnlyError when read_only=True."""
        # First create a document
        store = Store(temp_db_path, create=True)
        repo = DocumentRepository(store)
        doc = Document(content="test content")
        created_doc = await repo.create(doc)
        store.close()

        # Try to update in read-only mode
        store = Store(temp_db_path, read_only=True)
        repo = DocumentRepository(store)
        created_doc.content = "updated content"

        with pytest.raises(ReadOnlyError):
            await repo.update(created_doc)
        store.close()

    @pytest.mark.asyncio
    async def test_delete_raises_when_read_only(self, temp_db_path):
        """DocumentRepository.delete() raises ReadOnlyError when read_only=True."""
        # First create a document
        store = Store(temp_db_path, create=True)
        repo = DocumentRepository(store)
        doc = Document(content="test content")
        created_doc = await repo.create(doc)
        assert created_doc.id is not None
        doc_id = created_doc.id
        store.close()

        # Try to delete in read-only mode
        store = Store(temp_db_path, read_only=True)
        repo = DocumentRepository(store)

        with pytest.raises(ReadOnlyError):
            await repo.delete(doc_id)
        store.close()

    @pytest.mark.asyncio
    async def test_delete_all_raises_when_read_only(self, temp_db_path):
        """DocumentRepository.delete_all() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        store.close()

        store = Store(temp_db_path, read_only=True)
        repo = DocumentRepository(store)

        with pytest.raises(ReadOnlyError):
            await repo.delete_all()
        store.close()


class TestChunkRepositoryReadOnly:
    @pytest.mark.asyncio
    async def test_create_raises_when_read_only(self, temp_db_path):
        """ChunkRepository.create() raises ReadOnlyError when read_only=True."""
        # First create a document to have a valid document_id
        store = Store(temp_db_path, create=True)
        doc_repo = DocumentRepository(store)
        doc = Document(content="test content")
        created_doc = await doc_repo.create(doc)
        store.close()

        store = Store(temp_db_path, read_only=True)
        repo = ChunkRepository(store)
        chunk = Chunk(
            content="test chunk",
            document_id=created_doc.id,
            embedding=[0.0] * store.embedder._vector_dim,
        )

        with pytest.raises(ReadOnlyError):
            await repo.create(chunk)
        store.close()

    @pytest.mark.asyncio
    async def test_delete_by_document_id_raises_when_read_only(self, temp_db_path):
        """ChunkRepository.delete_by_document_id() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        store.close()

        store = Store(temp_db_path, read_only=True)
        repo = ChunkRepository(store)

        with pytest.raises(ReadOnlyError):
            await repo.delete_by_document_id("some-id")
        store.close()

    @pytest.mark.asyncio
    async def test_delete_all_raises_when_read_only(self, temp_db_path):
        """ChunkRepository.delete_all() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        store.close()

        store = Store(temp_db_path, read_only=True)
        repo = ChunkRepository(store)

        with pytest.raises(ReadOnlyError):
            await repo.delete_all()
        store.close()


class TestSettingsRepositoryReadOnly:
    def test_save_current_settings_raises_when_read_only(self, temp_db_path):
        """SettingsRepository.save_current_settings() raises ReadOnlyError when read_only=True."""
        store = Store(temp_db_path, create=True)
        store.close()

        store = Store(temp_db_path, read_only=True)
        repo = SettingsRepository(store)

        with pytest.raises(ReadOnlyError):
            repo.save_current_settings()
        store.close()


class TestClientReadOnly:
    def test_client_default_is_not_read_only(self, temp_db_path):
        """Client defaults to not read-only."""
        client = HaikuRAG(temp_db_path, create=True)
        assert client.is_read_only is False
        client.close()

    def test_client_can_be_created_read_only(self, temp_db_path):
        """Client can be created with read_only=True."""
        client = HaikuRAG(temp_db_path, create=True)
        client.close()

        client = HaikuRAG(temp_db_path, read_only=True)
        assert client.is_read_only is True
        client.close()

    @pytest.mark.vcr()
    async def test_client_create_document_raises_when_read_only(self, temp_db_path):
        """Client.create_document() raises ReadOnlyError when read_only=True."""
        client = HaikuRAG(temp_db_path, create=True)
        client.close()

        async with HaikuRAG(temp_db_path, read_only=True) as client:
            with pytest.raises(ReadOnlyError):
                await client.create_document("test content")

    @pytest.mark.vcr()
    async def test_client_delete_document_raises_when_read_only(self, temp_db_path):
        """Client.delete_document() raises ReadOnlyError when read_only=True."""
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document("test content")
            assert doc.id is not None
            doc_id = doc.id

        async with HaikuRAG(temp_db_path, read_only=True) as client:
            with pytest.raises(ReadOnlyError):
                await client.delete_document(doc_id)

    @pytest.mark.vcr()
    async def test_client_search_works_when_read_only(self, temp_db_path):
        """Client.search() works in read-only mode."""
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document("test content about cats")

        async with HaikuRAG(temp_db_path, read_only=True) as client:
            results = await client.search("cats")
            assert len(results) > 0

    @pytest.mark.vcr()
    async def test_client_list_documents_works_when_read_only(self, temp_db_path):
        """Client.list_documents() works in read-only mode."""
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document("test content")

        async with HaikuRAG(temp_db_path, read_only=True) as client:
            docs = await client.list_documents()
            assert len(docs) == 1

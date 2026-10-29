import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.store import ReadOnlyError
from haiku.rag.store.engine import Store
from haiku.rag.store.models import Chunk, Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.settings import SettingsRepository
from tests.conftest import for_path


class TestStoreReadOnly:
    async def test_store_read_only_raises_on_empty_directory(self, tmp_path):
        """Opening an empty directory in read-only mode raises ReadOnlyError."""
        empty_dir = tmp_path / "empty_db"
        empty_dir.mkdir()

        with pytest.raises(
            ReadOnlyError, match="Cannot create tables in read-only mode"
        ):
            async with Store(
                empty_dir,
                read_only=True,
                skip_validation=True,
                skip_migration_check=True,
            ):
                pass


async def _enter_write_transaction(store: Store, doc: Document) -> None:
    async with store.write_transaction():
        pass  # pragma: no cover - entering already raised


async def _update_document(store: Store, doc: Document) -> None:
    doc.content = "updated content"
    await DocumentRepository(store).update(doc)


async def _create_chunk(store: Store, doc: Document) -> None:
    chunk = Chunk(
        content="test chunk",
        document_id=doc.id,
        embedding=[0.0] * store.embedder._vector_dim,
    )
    await ChunkRepository(store).create(chunk)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda store, doc: store.vacuum(), id="vacuum"),
        pytest.param(
            lambda store, doc: store.set_haiku_version("1.0.0"),
            id="set_haiku_version",
        ),
        pytest.param(
            lambda store, doc: store.recreate_embeddings_table(),
            id="recreate_embeddings_table",
        ),
        pytest.param(_enter_write_transaction, id="write_transaction"),
        pytest.param(
            lambda store, doc: DocumentRepository(store).create(
                Document(content="test content")
            ),
            id="document_create",
        ),
        pytest.param(_update_document, id="document_update"),
        pytest.param(
            lambda store, doc: DocumentRepository(store).delete(doc.id),
            id="document_delete",
        ),
        pytest.param(
            lambda store, doc: DocumentRepository(store).delete_all(),
            id="document_delete_all",
        ),
        pytest.param(_create_chunk, id="chunk_create"),
        pytest.param(
            lambda store, doc: ChunkRepository(store).delete_by_document_id("some-id"),
            id="chunk_delete_by_document_id",
        ),
        pytest.param(
            lambda store, doc: ChunkRepository(store).delete_all(),
            id="chunk_delete_all",
        ),
        pytest.param(
            lambda store, doc: SettingsRepository(store).save_current_settings(),
            id="settings_save_current_settings",
        ),
    ],
)
async def test_a_read_only_store_refuses_writes(temp_db_path, operation):
    async with Store(temp_db_path, create=True) as store:
        assert store.is_read_only is False
        doc = await DocumentRepository(store).create(Document(content="test content"))
        assert doc.id is not None

    async with Store(temp_db_path, read_only=True) as store:
        assert store.is_read_only is True
        with pytest.raises(ReadOnlyError):
            await operation(store, doc)


class TestClientReadOnly:
    @pytest.mark.vcr()
    async def test_client_create_document_raises_when_read_only(self, temp_db_path):
        """Client.create_document() raises ReadOnlyError when read_only=True."""
        async with HaikuRAG(temp_db_path, create=True):
            pass

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


class TestAppReadVerbsDoNotWrite:
    async def test_list_works_with_mismatched_embeddings_config(self, temp_db_path):
        """list skips the embeddings compatibility check and never writes."""
        from haiku.rag.app import HaikuRAGApp
        from haiku.rag.config import AppConfig

        async with Store(temp_db_path, create=True) as store:
            await DocumentRepository(store).create(Document(content="test content"))
            settings_before = await SettingsRepository(store).get_current_settings()

        drift = AppConfig()
        drift.embeddings.model.name = "different-model"
        drift.embeddings.model.vector_dim = 4096

        app = HaikuRAGApp(scope=for_path(temp_db_path, drift), config=drift)
        await app.list_documents()

        async with Store(temp_db_path, skip_validation=True, read_only=True) as store:
            settings_after = await SettingsRepository(store).get_current_settings()
        assert settings_after == settings_before

    async def test_get_works_with_mismatched_embeddings_config(self, temp_db_path):
        """get skips the embeddings compatibility check."""
        from haiku.rag.app import HaikuRAGApp
        from haiku.rag.config import AppConfig

        async with Store(temp_db_path, create=True) as store:
            doc = await DocumentRepository(store).create(
                Document(content="test content")
            )

        drift = AppConfig()
        drift.embeddings.model.vector_dim = 4096

        app = HaikuRAGApp(scope=for_path(temp_db_path, drift), config=drift)
        assert doc.id is not None
        await app.get_document(doc.id)

    async def test_visualize_works_with_mismatched_embeddings_config(
        self, temp_db_path
    ):
        """visualize skips the embeddings compatibility check."""
        from haiku.rag.app import HaikuRAGApp
        from haiku.rag.config import AppConfig

        async with Store(temp_db_path, create=True):
            pass

        drift = AppConfig()
        drift.embeddings.model.vector_dim = 4096

        app = HaikuRAGApp(scope=for_path(temp_db_path, drift), config=drift)
        await app.visualize_chunk("missing-chunk-id")

    async def test_write_verb_raises_on_drift_without_writing(self, temp_db_path):
        """A write CLI verb opens writable: drift raises before any write."""
        from haiku.rag.app import HaikuRAGApp
        from haiku.rag.config import AppConfig
        from haiku.rag.store.exceptions import ConfigMismatchError

        async with Store(temp_db_path, create=True) as store:
            stored_name_before = (
                await SettingsRepository(store).get_current_settings()
            )["embeddings"]["model"]["name"]

        drift = AppConfig()
        drift.embeddings.model.name = "different-model"

        app = HaikuRAGApp(scope=for_path(temp_db_path, drift), config=drift)
        with pytest.raises(ConfigMismatchError):
            await app.add_document_from_text("hello")

        async with Store(temp_db_path, skip_validation=True, read_only=True) as store:
            stored_name_after = (
                await SettingsRepository(store).get_current_settings()
            )["embeddings"]["model"]["name"]

        assert stored_name_after == stored_name_before

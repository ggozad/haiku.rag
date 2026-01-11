import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.store.engine import Store
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.mm_asset import MMAsset


def _mm_enabled_config():
    cfg = Config.model_copy(deep=True)
    cfg.multimodal.enabled = True
    # Ensure dim matches our dummy vectors in tests
    cfg.multimodal.model.vector_dim = 2048
    return cfg


def test_mm_assets_table_created_when_enabled(temp_db_path):
    cfg = _mm_enabled_config()
    store = Store(temp_db_path, create=True, config=cfg)
    assert store.mm_assets_table is not None
    assert "mm_assets" in store.db.table_names()


async def test_version_rollback_on_mm_index_failure(temp_db_path, monkeypatch):
    cfg = _mm_enabled_config()

    async with HaikuRAG(db_path=temp_db_path, create=True, config=cfg) as client:
        dim = int(client.store.embedder._vector_dim)

        async def fail_after_writing_one_asset(doc):
            # write one asset, then fail -> should rollback all tables
            assert doc.id is not None
            await client.mm_asset_repository.create(
                MMAsset(
                    document_id=doc.id,
                    doc_item_ref="#/pictures/0",
                    item_index=0,
                    page_no=1,
                    bbox={"left": 0.0, "top": 1.0, "right": 2.0, "bottom": 3.0},
                    caption="dummy",
                    embedding=[0.0] * 2048,
                )
            )
            raise RuntimeError("boom")

        monkeypatch.setattr(client, "_index_mm_assets_for_document", fail_after_writing_one_asset)

        # Instead of calling create_document (which would hit external embedders),
        # call the internal store path with pre-embedded chunks.
        from haiku.rag.store.models.document import Document

        with pytest.raises(RuntimeError, match="boom"):
            await client._store_document_with_chunks(
                Document(content="x"),
                [Chunk(content="c", metadata={}, order=0, embedding=[0.0] * dim)],
            )

    # After failure, database should be rolled back to empty state
    store = Store(temp_db_path, create=True, config=cfg)
    assert store.documents_table.count_rows() == 0
    assert store.chunks_table.count_rows() == 0
    assert store.mm_assets_table is not None
    assert store.mm_assets_table.count_rows() == 0


async def test_update_rollback_restores_mm_assets(temp_db_path, monkeypatch):
    cfg = _mm_enabled_config()

    async with HaikuRAG(db_path=temp_db_path, create=True, config=cfg) as client:
        dim = int(client.store.embedder._vector_dim)

        # Create baseline document without calling external embedders
        from haiku.rag.store.models.document import Document

        doc = await client._store_document_with_chunks(
            Document(content="base"),
            [Chunk(content="c", metadata={}, order=0, embedding=[0.0] * dim)],
        )
        assert doc.id is not None

        # Seed mm_assets with one row
        await client.mm_asset_repository.create(
            MMAsset(
                document_id=doc.id,
                doc_item_ref="#/pictures/seed",
                item_index=0,
                page_no=1,
                bbox={"left": 0.0, "top": 1.0, "right": 2.0, "bottom": 3.0},
                caption="seed",
                embedding=[1.0] * 2048,
            )
        )

        async def fail_index(_doc):
            raise RuntimeError("mm index fail")

        monkeypatch.setattr(client, "_index_mm_assets_for_document", fail_index)

        with pytest.raises(RuntimeError, match="mm index fail"):
            # Call internal update path to avoid external embedder usage
            await client._update_document_with_chunks(
                Document(
                    id=doc.id,
                    content="changed",
                    docling_document_json=doc.docling_document_json,
                    docling_version=doc.docling_version,
                ),
                [Chunk(content="c2", metadata={}, order=0, embedding=[0.0] * dim)],
            )

    # After rollback, original mm_assets row should still exist
    store = Store(temp_db_path, create=True, config=cfg)
    assert store.mm_assets_table is not None
    assert store.mm_assets_table.count_rows() == 1


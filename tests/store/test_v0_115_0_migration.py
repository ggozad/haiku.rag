import pytest

from haiku.rag.store.engine import Store
from haiku.rag.store.models.document_item import DocumentItem
from haiku.rag.store.repositories.document_item import DocumentItemRepository
from haiku.rag.store.upgrades import get_pending_upgrades
from haiku.rag.store.upgrades.v0_115_0 import _apply_chunk_media, _apply_item_times


async def _columns(store: Store) -> set[str]:
    return {f.name for f in await store.document_items_table.schema()}


async def test_adds_item_time_columns_to_existing_rows(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await DocumentItemRepository(store).create_items(
            "doc-1",
            [DocumentItem(document_id="doc-1", position=0, self_ref="#/texts/0")],
        )
        await store.document_items_table.drop_columns(["start_time", "end_time"])

        await _apply_item_times(store)

        assert {"start_time", "end_time"} <= await _columns(store)
        [item] = (
            await DocumentItemRepository(store).get_items_in_ranges({"doc-1": (0, 0)})
        )["doc-1"]
        assert (item.start_time, item.end_time) == (None, None)


async def test_leaves_a_current_table_unwritten(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        versions = await store.current_table_versions()

        await _apply_item_times(store)

        assert await store.current_table_versions() == versions


def test_is_registered():
    assert "0.115.0" in [u.version for u in get_pending_upgrades("0.114.0")]


async def _chunks_without_media(temp_db_path, version: str = "0.112.2") -> None:
    from haiku.rag.client import HaikuRAG
    from tests.store.test_chunk_media import _chunks

    async with HaikuRAG(temp_db_path, create=True) as rag:
        table = rag.store.chunks_table
        await rag.chunk_repository.create(
            _chunks(rag.store._config.embeddings.model.vector_dim)
        )
        for index in await table.list_indices():
            if index.columns == ["media"]:
                await table.drop_index(index.name)
        await table.drop_columns(["media", "start_time", "end_time"])
        await rag.store.set_haiku_version(version)


@pytest.mark.parametrize("version", ["0.112.2", "0.64.0"])
async def test_migrate_fills_chunk_media_from_the_metadata(temp_db_path, version):
    from tests.store.test_chunk_media import EXPECTED

    await _chunks_without_media(temp_db_path, version)

    async with Store(temp_db_path, skip_migration_check=True) as store:
        applied = await store.migrate()

        assert applied[-1] == "0.115.0: Add media, start_time and end_time to chunks"
        rows = (await store.chunks_table.query().to_arrow()).to_pylist()
        assert {
            r["content"]: (r["media"], r["start_time"], r["end_time"]) for r in rows
        } == EXPECTED
        assert ["media"] in [i.columns for i in await store.chunks_table.list_indices()]


async def test_chunk_media_leaves_a_current_table_unwritten(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        versions = await store.current_table_versions()

        await _apply_chunk_media(store)

        assert await store.current_table_versions() == versions


async def test_chunk_media_leaves_a_document_chunk_with_a_null_time_range(
    temp_db_path,
):
    from haiku.rag.client import HaikuRAG
    from haiku.rag.store.models.chunk import Chunk

    async with HaikuRAG(temp_db_path, create=True) as rag:
        table = rag.store.chunks_table
        dim = rag.store._config.embeddings.model.vector_dim
        await rag.chunk_repository.create(
            Chunk(
                document_id="doc-1",
                content="paper",
                metadata={"doc_item_refs": ["#/texts/0"], "time_range": None},
                embedding=[0.1] * dim,
            )
        )
        await table.drop_columns(["media", "start_time", "end_time"])

        await _apply_chunk_media(rag.store)

        [row] = (await table.query().to_arrow()).to_pylist()
        assert (row["media"], row["start_time"], row["end_time"]) == (None, None, None)


async def test_an_interrupted_chunk_media_fill_resumes(temp_db_path, monkeypatch):
    from haiku.rag.store.upgrades import v0_115_0
    from tests.store.test_chunk_media import EXPECTED

    await _chunks_without_media(temp_db_path)

    def fail(metadata):
        raise RuntimeError("interrupted")

    async with Store(temp_db_path, skip_migration_check=True) as store:
        with monkeypatch.context() as patched:
            patched.setattr(v0_115_0, "chunk_media", fail)
            with pytest.raises(RuntimeError, match="interrupted"):
                await _apply_chunk_media(store)

        await _apply_chunk_media(store)

        rows = (await store.chunks_table.query().to_arrow()).to_pylist()
        assert {
            r["content"]: (r["media"], r["start_time"], r["end_time"]) for r in rows
        } == EXPECTED

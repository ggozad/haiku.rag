from importlib import metadata

from haiku.rag.store.engine import Store
from haiku.rag.store.models import Chunk
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.upgrades import get_pending_upgrades
from haiku.rag.store.upgrades.v0_91_0 import _apply_fts_format_v2


async def _add_chunks(store: Store) -> None:
    await ChunkRepository(store).create(
        [
            Chunk(
                document_id="doc-1",
                content=content,
                embedding=[0.1] * store.embedder.vector_dim,
                order=order,
            )
            for order, content in enumerate(
                ["gardens in spring", "rivers and gardens", "mountain passes"]
            )
        ]
    )


async def _fts_index(store: Store):
    [index] = [
        i for i in await store.chunks_table.list_indices() if i.index_type == "FTS"
    ]
    return index


async def _search(store: Store, query: str) -> list[str]:
    rows = (
        await store.chunks_table.query()
        .nearest_to_text(query, columns="content_fts")
        .select(["id", "_score"])
        .to_arrow()
    )
    return rows.column("id").to_pylist()


async def _make_v1(store: Store) -> None:
    dataset = await store.chunks_table.to_lance()
    dataset.create_scalar_index(
        "content_fts",
        index_type="INVERTED",
        name="content_fts_idx",
        replace=True,
        with_position=True,
        remove_stop_words=False,
        format_version=1,
    )
    await store.chunks_table.checkout_latest()


async def test_rebuilds_a_v1_fts_index_as_v2(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _add_chunks(store)
        await _make_v1(store)
        assert (await _fts_index(store)).index_version == 1
        before = await _search(store, "gardens")
        rows = await store.chunks_table.count_rows()

        await _apply_fts_format_v2(store)

        index = await _fts_index(store)
        assert (index.name, index.index_version) == ("content_fts_idx", 2)
        assert await store.chunks_table.count_rows() == rows
        assert await _search(store, "gardens") == before


async def test_leaves_a_v2_database_unwritten(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _add_chunks(store)
        versions = await store.current_table_versions()

        await _apply_fts_format_v2(store)

        assert await store.current_table_versions() == versions


async def test_a_new_database_has_no_pending_migration(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        assert await store.get_haiku_version() == metadata.version("haiku.rag-slim")
        assert get_pending_upgrades(await store.get_haiku_version()) == []


async def test_migrating_a_v2_database_writes_only_the_version(temp_db_path):
    async with Store(temp_db_path, create=True) as store:
        await _add_chunks(store)
        await store.set_haiku_version("0.90.0")

    async with Store(temp_db_path, skip_migration_check=True) as store:
        assert [u.version for u in get_pending_upgrades("0.90.0")] == ["0.91.0"]
        versions = await store.current_table_versions()

        await store.migrate()

        after = await store.current_table_versions()
        assert {n: v for n, v in after.items() if n != "settings"} == {
            n: v for n, v in versions.items() if n != "settings"
        }
        assert after["settings"] > versions["settings"]

    async with Store(temp_db_path) as store:
        assert await store.get_haiku_version() == metadata.version("haiku.rag-slim")


async def test_an_undeclared_v1_fts_index_triggers_nothing(temp_db_path):
    async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
        await _add_chunks(store)
        dataset = await store.chunks_table.to_lance()
        dataset.create_scalar_index(
            "content",
            index_type="INVERTED",
            name="operator_content_fts",
            with_position=True,
            format_version=1,
        )
        await store.chunks_table.checkout_latest()
        versions = await store.current_table_versions()

        await _apply_fts_format_v2(store)

        assert await store.current_table_versions() == versions

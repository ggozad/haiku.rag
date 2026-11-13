from haiku.rag.store.engine import Store
from haiku.rag.store.models.document_item import DocumentItem
from haiku.rag.store.repositories.document_item import DocumentItemRepository
from haiku.rag.store.upgrades import get_pending_upgrades
from haiku.rag.store.upgrades.v0_112_0 import _apply_item_times


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
    assert "0.112.0" in [u.version for u in get_pending_upgrades("0.111.1")]

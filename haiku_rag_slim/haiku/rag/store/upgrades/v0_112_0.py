import pyarrow as pa

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade


async def _apply_item_times(store: Store) -> None:
    """Add the nullable `start_time` and `end_time` columns to `document_items`."""
    existing = {f.name for f in await store.document_items_table.schema()}
    missing = [
        pa.field(name, pa.float64())
        for name in ("start_time", "end_time")
        if name not in existing
    ]
    if missing:
        await store.document_items_table.add_columns(pa.schema(missing))


upgrade_item_times = Upgrade(
    version="0.112.0",
    apply=_apply_item_times,
    description="Add start_time and end_time to document items",
)

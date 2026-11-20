import pyarrow as pa

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade


async def _apply_original_columns(store: Store) -> None:
    """Add the columns holding a document's original file and its MD5."""
    present = {field.name for field in await store.documents_table.schema()}
    missing = [
        field
        for field in (
            pa.field("original", pa.large_binary()),
            pa.field("original_md5", pa.utf8()),
        )
        if field.name not in present
    ]
    if missing:
        await store.documents_table.add_columns(missing)


upgrade_original_columns = Upgrade(
    version="0.113.0",
    apply=_apply_original_columns,
    description="Add the original file columns to documents",
)

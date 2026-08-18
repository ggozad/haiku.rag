import logging

from haiku.rag.store.engine import Store, ensure_indexes
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)


async def _apply_index_hot_lookup_keys(store: Store) -> None:
    """Add the declared indexes to a database created before 0.75.0.

    Rewrites no rows. Each index build reads the column it indexes.
    """
    for table_name, table in store._tables().items():
        applied = await ensure_indexes(table, table_name)
        if applied:
            logger.info(f"Indexed {table_name}: {', '.join(sorted(applied))}")


upgrade_index_hot_lookup_keys = Upgrade(
    version="0.75.0",
    apply=_apply_index_hot_lookup_keys,
    description="Index documents.id, chunks.id, chunks.document_id and document_items.label",
)

import logging

from haiku.rag.store.engine import Store, ensure_indexes
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)


async def _apply_index_hot_lookup_keys(store: Store) -> None:
    """Bring an existing database up to the declared index set.

    Adds what earlier versions never created: BTree on `documents.id`,
    `chunks.id` and `chunks.document_id`, and a Bitmap on `document_items.label`.
    Without them those lookups scan the column, which on object storage is
    network I/O on paths that run per document.

    Does not materialize rows in Python or rewrite table data. The cost is the
    index builds themselves, which read the indexed columns to sort them.

    `ensure_indexes` skips a column already indexed with the declared type, so a
    database that already has the full set (a merged one, say) comes through
    untouched rather than re-sorting every indexed column. Columns it does not
    declare are left alone, so a vector index or an externally added index
    survives.
    """
    for table_name, table in store._tables().items():
        applied = await ensure_indexes(table, table_name)
        if applied:
            logger.info(
                f"Indexed {table_name} (created or corrected): "
                f"{', '.join(sorted(applied))}"
            )


upgrade_index_hot_lookup_keys = Upgrade(
    version="0.75.0",
    apply=_apply_index_hot_lookup_keys,
    description="Index documents.id, chunks.id, chunks.document_id and document_items.label",
)

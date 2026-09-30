import logging

from lancedb.index import FTS

from haiku.rag.store.engine import Store
from haiku.rag.store.schema import index_specs, rebuild_indexes
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)


async def _apply_fts_format_v2(store: Store) -> None:
    """Rebuild the indexes of a table whose declared FTS index is in the v1 format.

    Rewrites no rows, and writes no table version where every declared FTS
    index is already v2 or later. `Store.migrate` still advances the stored
    version.
    """
    for table_name, table in store._tables().items():
        declared = {c for c, cfg in index_specs(table_name) if isinstance(cfg, FTS)}
        if any(
            i.index_type == "FTS"
            and declared.intersection(i.columns)
            and i.index_version is not None
            and i.index_version < 2
            for i in await table.list_indices()
        ):
            applied = await rebuild_indexes(table, table_name)
            logger.info(f"Reindexed {table_name}: {', '.join(sorted(applied))}")


upgrade_fts_format_v2 = Upgrade(
    version="0.91.0",
    apply=_apply_fts_format_v2,
    description="Rebuild v1-format full-text indexes as v2",
)

import logging

import pyarrow as pa

from haiku.rag.store.engine import ConnectionMode, Store
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)


async def _apply_original_columns(store: Store) -> None:
    """Add the original file columns and delete the stored page images.

    Vacuums at zero retention, so no other process may use the database while
    it runs.
    """
    table = store.documents_table
    present = {field.name for field in await table.schema()}
    missing = [
        field
        for field in (
            pa.field("original", pa.large_binary()),
            pa.field("original_md5", pa.utf8()),
        )
        if field.name not in present
    ]
    if missing:
        await table.add_columns(missing)
    if "docling_pages" in present:
        await table.drop_columns(["docling_pages"])

    # A dropped column stays in its data files. Rewriting every row writes
    # files without it, and the vacuum deletes the old ones.
    rows = await table.query().select(["id", "docling_version"]).to_arrow()
    await (
        table.merge_insert("id")
        .when_matched_update_all()
        .use_index(False)
        .execute(rows)
    )
    vacuumed = await store.vacuum(retention_seconds=0)
    if not vacuumed and store._connection_mode != ConnectionMode.CLOUD:
        raise RuntimeError(
            "The vacuum was skipped, so the page images still take their space. "
            "Run 'haiku-rag migrate' again."
        )

    tags = sorted(await table.tags.list())
    if tags:
        logger.warning(
            "Tags taken before this migration keep their page images on disk: "
            f"{', '.join(tags)}. Delete them and run 'haiku-rag vacuum "
            "--retention-seconds 0' to free the space."
        )


upgrade_original_columns = Upgrade(
    version="0.113.0",
    apply=_apply_original_columns,
    description="Store originals, delete page images",
)

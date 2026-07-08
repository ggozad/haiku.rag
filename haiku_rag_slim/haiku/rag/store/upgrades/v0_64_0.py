import logging

from lancedb.index import BTree

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)


async def _apply_rename_document_meta_id(store: Store) -> None:
    """Rename the `document_meta` identity column `document_id` → `id`.

    In `document_meta` the column is the document's own identity (1:1 with the
    row), so `id` matches the `documents` table and the public `Document.id`.
    Callers filter/list/count against `document_meta`, so with the old name a
    filter on `id` raised "No field named id". Idempotent: skips if the column
    is already `id`.
    """
    schema = await store.document_meta_table.schema()
    if "id" in schema.names:
        logger.info("document_meta.id already present; nothing to rename")
        return

    await store.document_meta_table.alter_columns(
        {"path": "document_id", "rename": "id"}  # ty: ignore[invalid-argument-type]
    )
    await store.document_meta_table.create_index("id", config=BTree(), replace=True)


upgrade_rename_document_meta_id = Upgrade(
    version="0.64.0",
    apply=_apply_rename_document_meta_id,
    description="Rename the document_meta identity column document_id to id",
)

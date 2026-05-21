import json
import logging

import pyarrow as pa

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade
from haiku.rag.utils import escape_sql_string

logger = logging.getLogger(__name__)

# Columns v0.40.0 owns. Later migrations add more (``picture_data`` in v0.45.0,
# ``heading_level`` / ``tree_depth`` in v0.48.0); those are filled with
# type-appropriate defaults at insert time so the migration stays independent
# of future model changes.
_V0_40_0_OWNED_COLUMNS: frozenset[str] = frozenset(
    {"document_id", "position", "self_ref", "label", "text", "page_numbers"}
)


def _default_for(field: pa.Field):
    """Return a sensible placeholder for a column v0.40.0 doesn't write.

    Nullable columns get ``None``; non-nullable columns get the Arrow-typed
    zero (``0`` for ints, ``""`` for strings, ``b""`` for binary). This
    matches the Pydantic ``Field(default=...)`` values on the live model.
    """
    if field.nullable:
        return None
    if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
        return 0
    # Defensive branches for column types not yet introduced after v0.40.0.
    # Kept so a future non-nullable string / binary / bool addition doesn't
    # turn the migration into a TypeError on real DBs.
    if pa.types.is_string(field.type) or pa.types.is_large_string(
        field.type
    ):  # pragma: no cover
        return ""
    if pa.types.is_binary(field.type) or pa.types.is_large_binary(
        field.type
    ):  # pragma: no cover
        return b""
    if pa.types.is_boolean(field.type):  # pragma: no cover
        return False
    raise TypeError(  # pragma: no cover
        f"No default for non-nullable arrow type {field.type}"
    )


def _build_items_arrow_table(live_schema: pa.Schema, rows: list[dict]) -> pa.Table:
    """Materialise ``rows`` as a PyArrow table matching ``live_schema``.

    ``rows`` provides values for the v0.40.0-owned columns; any additional
    columns the live schema carries (added by later migrations) are filled
    with type-appropriate defaults so ``Table.add`` accepts the input even
    when those columns are non-nullable.
    """
    columns: dict[str, pa.Array] = {}
    for field in live_schema:
        if field.name in _V0_40_0_OWNED_COLUMNS:
            columns[field.name] = pa.array(
                [row[field.name] for row in rows], type=field.type
            )
        else:
            columns[field.name] = pa.array(
                [_default_for(field)] * len(rows), type=field.type
            )
    return pa.table(columns, schema=live_schema)


async def _apply_populate_document_items(store: Store) -> None:  # pragma: no cover
    """Populate document_items table from existing docling documents."""
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.store.compression import decompress_json
    from haiku.rag.store.models.document_item import extract_items

    # Get all document IDs that have docling data
    ids = (await store.documents_table.query().select(["id"]).to_arrow()).to_pylist()
    ids = [row["id"] for row in ids]

    if not ids:
        logger.info("No documents to migrate")
        return

    total = len(ids)
    logger.info("Populating document_items for %d documents", total)
    migrated = 0
    skipped = 0

    for idx, doc_id in enumerate(ids, 1):
        # Load only docling data
        safe_id = escape_sql_string(doc_id)
        rows = await (
            store.documents_table.query()
            .select(["id", "docling_document"])
            .where(f"id = '{safe_id}'")
            .limit(1)
            .to_list()
        )

        if not rows:
            skipped += 1
            continue

        row = rows[0]
        docling_blob = row.get("docling_document")
        if not docling_blob or not isinstance(docling_blob, bytes):
            skipped += 1
            continue

        try:
            json_str = decompress_json(docling_blob)
            docling_doc = DoclingDocument.model_validate_json(json_str)
            items = extract_items(doc_id, docling_doc)

            if items:
                live_schema = await store.document_items_table.schema()
                records = _build_items_arrow_table(
                    live_schema,
                    [
                        {
                            "document_id": item.document_id,
                            "position": item.position,
                            "self_ref": item.self_ref,
                            "label": item.label,
                            "text": item.text,
                            "page_numbers": json.dumps(item.page_numbers),
                        }
                        for item in items
                    ],
                )
                await store.document_items_table.add(records)

            migrated += 1
            if idx % 10 == 0 or idx == total:
                logger.info(
                    "Progress: %d/%d documents (%d migrated, %d skipped)",
                    idx,
                    total,
                    migrated,
                    skipped,
                )
        except Exception:
            logger.warning("Failed to extract items for document %s", doc_id)
            skipped += 1

    logger.info(
        "Migration complete: %d migrated, %d skipped out of %d",
        migrated,
        skipped,
        total,
    )


upgrade_populate_document_items = Upgrade(
    version="0.40.0",
    apply=_apply_populate_document_items,
    description="Populate document_items table for context expansion",
)

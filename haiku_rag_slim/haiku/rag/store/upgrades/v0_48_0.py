import logging

import pyarrow as pa
from lancedb.index import BTree

from haiku.rag.store.engine import DocumentItemRecord, Store
from haiku.rag.store.upgrades import Upgrade
from haiku.rag.utils import escape_sql_string

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 10


async def _ensure_columns(store: Store) -> None:
    """Add heading_level + tree_depth (int64) columns if missing. Idempotent."""
    arrow_schema = await store.document_items_table.schema()
    existing = {f.name for f in arrow_schema}
    new_fields = []
    if "heading_level" not in existing:
        new_fields.append(pa.field("heading_level", pa.int64()))  # pragma: no cover
    if "tree_depth" not in existing:
        new_fields.append(pa.field("tree_depth", pa.int64()))  # pragma: no cover
    if new_fields:
        # Pre-0.48.0 DBs only — fresh DBs declare both columns in _init_tables.
        await store.document_items_table.add_columns(
            pa.schema(new_fields)
        )  # pragma: no cover


async def _ensure_indexes(store: Store) -> None:
    """Ensure BTree scalar indexes exist on the hot document_items lookup columns.

    Fresh DBs created via ``_init_tables`` get these on first creation, but
    DBs that predate that code (or were downloaded as pre-built artifacts) were
    table-scanning every per-doc query — visible as ~100–300 ms even for small
    docs. Built before the heading_level backfill so the per-doc WHERE clauses
    in the backfill loop benefit from it.
    """
    for column in ("document_id", "position", "self_ref"):
        await store.document_items_table.create_index(
            column, config=BTree(), replace=True
        )


async def _apply_backfill_heading_hierarchy(store: Store) -> None:
    """Add heading_level + tree_depth columns and backfill from docling structure.

    For each document with a docling_document blob, decompress it, re-run
    extract_items, and update the existing items rows with the two new ints.
    Documents without docling (plain-text adds) are skipped; their rows keep
    the column-add default (NULL, materialised as 0 by the repository).

    Idempotent: re-running on migrated rows yields identical values.
    """
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.store.compression import decompress_json
    from haiku.rag.store.models.document_item import extract_items

    await _ensure_columns(store)
    await _ensure_indexes(store)

    ids = (await store.documents_table.query().select(["id"]).to_arrow()).to_pylist()
    ids = [row["id"] for row in ids]
    total = len(ids)

    logger.info("Backfilling heading_level + tree_depth across %d documents", total)
    backfilled = 0
    skipped = 0

    for idx, doc_id in enumerate(ids, 1):
        safe_id = escape_sql_string(doc_id)
        rows = await (
            store.documents_table.query()
            .select(["id", "docling_document"])
            .where(f"id = '{safe_id}'")
            .limit(1)
            .to_list()
        )
        blob = rows[0].get("docling_document")
        if not blob or not isinstance(blob, bytes):
            skipped += 1
            continue

        try:
            docling_doc = DoclingDocument.model_validate_json(decompress_json(blob))
            fresh_items = extract_items(doc_id, docling_doc)
        except Exception:  # pragma: no cover
            logger.warning(
                "Failed to re-extract items for %s; skipping", doc_id, exc_info=True
            )
            skipped += 1
            continue

        if not fresh_items:  # pragma: no cover
            skipped += 1
            continue

        existing_rows = await (
            store.document_items_table.query()
            .where(f"document_id = '{safe_id}'")
            .to_list()
        )
        existing_by_ref = {r["self_ref"]: r for r in existing_rows}

        # If the stored rows and a fresh extract disagree on count, the
        # docling parser version has drifted. Skip and let `rebuild` reconcile.
        if len(existing_rows) != len(fresh_items):  # pragma: no cover
            logger.warning(
                "Item count drift for %s (stored=%d, fresh=%d); skipping",
                doc_id,
                len(existing_rows),
                len(fresh_items),
            )
            skipped += 1
            continue

        records: list[DocumentItemRecord] = []
        for item in fresh_items:
            row = existing_by_ref.get(item.self_ref)
            if row is None:  # pragma: no cover
                continue
            records.append(
                DocumentItemRecord(
                    document_id=doc_id,
                    position=row["position"],
                    self_ref=item.self_ref,
                    label=row.get("label", ""),
                    text=row.get("text", ""),
                    page_numbers=row.get("page_numbers", "[]"),
                    picture_data=row.get("picture_data"),
                    heading_level=item.heading_level,
                    tree_depth=item.tree_depth,
                )
            )

        if records:
            await (
                store.document_items_table.merge_insert(["document_id", "self_ref"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(records)
            )
            backfilled += 1

        if idx % PROGRESS_INTERVAL == 0 or idx == total:
            logger.info(
                "Progress: %d/%d (%d backfilled, %d skipped)",
                idx,
                total,
                backfilled,
                skipped,
            )

    logger.info(
        "Backfill complete: %d backfilled, %d skipped of %d",
        backfilled,
        skipped,
        total,
    )


upgrade_backfill_heading_hierarchy = Upgrade(
    version="0.48.0",
    apply=_apply_backfill_heading_hierarchy,
    description=(
        "Backfill heading_level + tree_depth on document_items, "
        "ensure BTree indexes on document_id / position / self_ref"
    ),
)

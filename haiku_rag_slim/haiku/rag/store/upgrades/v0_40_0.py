import json
import logging

from haiku.rag.store.engine import DocumentItemRecord, Store
from haiku.rag.store.upgrades import Upgrade
from haiku.rag.utils import escape_sql_string

logger = logging.getLogger(__name__)


def _apply_populate_document_items(store: Store) -> None:  # pragma: no cover
    """Populate document_items table from existing docling documents."""
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.store.compression import decompress_json
    from haiku.rag.store.models.document_item import extract_items

    # Get all document IDs that have docling data
    ids = [
        row["id"]
        for row in store.documents_table.search().select(["id"]).to_arrow().to_pylist()
    ]

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
        rows = (
            store.documents_table.search()
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
                records = [
                    DocumentItemRecord(
                        document_id=item.document_id,
                        position=item.position,
                        self_ref=item.self_ref,
                        label=item.label,
                        text=item.text,
                        page_numbers=json.dumps(item.page_numbers),
                    )
                    for item in items
                ]
                store.document_items_table.add(records)

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

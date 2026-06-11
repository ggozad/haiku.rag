import logging
import shutil

from haiku.rag.store.engine import DocumentMetaRecord, Store
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)

_LEGACY_COLUMNS = ["uri", "title", "metadata", "created_at", "updated_at"]


async def _apply_split_document_meta(store: Store) -> None:
    """Move the mutable document attributes (uri/title/metadata/created_at/
    updated_at) out of the blob-bearing `documents` row into `document_meta`.

    After this, a metadata/title/source_revision update writes only the small
    `document_meta` row instead of rewriting the multi-MB docling blob. The
    blobs (content, docling_document, docling_pages) stay in `documents`.

    The `document_meta` table itself is created on open by `_init_tables`; this
    migration populates it and drops the now-relocated columns from `documents`.
    Idempotent: a re-run after a partial failure skips already-moved rows and
    skips the column drop if the columns are already gone.
    """
    schema = await store.documents_table.schema()
    present = [c for c in _LEGACY_COLUMNS if c in schema.names]

    if not present:
        logger.info("documents already split; nothing to move")
        return

    # Resume support: skip documents whose meta row already exists.
    existing_meta = {
        row["document_id"]
        for row in await store.document_meta_table.query()
        .select(["document_id"])
        .to_list()
    }

    rows = await store.documents_table.query().select(["id", *present]).to_list()
    records = []
    for row in rows:
        if row["id"] in existing_meta:
            continue
        meta = row.get("metadata")
        records.append(
            DocumentMetaRecord(
                document_id=row["id"],
                uri=row.get("uri"),
                title=row.get("title"),
                metadata=meta if isinstance(meta, str) and meta else "{}",
                created_at=row.get("created_at") or "",
                updated_at=row.get("updated_at") or "",
            )
        )
    if records:
        logger.info(
            "Moving attributes for %d document(s) into document_meta", len(records)
        )
        await store.document_meta_table.add(records)

    # Drop the relocated columns from documents (a metadata operation — no row
    # rewrite, so it is cheap and safe even on a near-full disk).
    logger.info("Dropping %s from documents", ", ".join(present))
    await store.documents_table.drop_columns(present)

    # Reclaim the bloat accumulated before the fix (superseded docling rows from
    # past metadata churn). retention=0 is safe ONLY because migrate is
    # exclusive/single-writer; this must never become normal ingester behaviour.
    # Compaction rewrites the live blobs once (transient peak ~= current size +
    # one compacted copy), so skip it (best-effort) when free disk cannot cover
    # that — the split is already done; the user can run `haiku-rag vacuum`.
    try:
        # lancedb's .stats() stub claims TableStatistics but returns a plain dict.
        stats: dict = await store.documents_table.stats()  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        live_bytes = int(stats.get("total_bytes", 0))
    except Exception:
        live_bytes = 0
    free_bytes = shutil.disk_usage(store.db_path).free
    if live_bytes and free_bytes < live_bytes:
        logger.warning(
            "Skipping post-migration vacuum: need ~%.2f GB free to compact the "
            "documents table, have %.2f GB. Run `haiku-rag vacuum` once you have "
            "space to reclaim the accumulated bloat.",
            live_bytes / 1e9,
            free_bytes / 1e9,
        )
        return

    logger.info("Vacuuming to reclaim accumulated document bloat")
    await store.vacuum(retention_seconds=0)


upgrade_split_document_meta = Upgrade(
    version="0.58.0",
    apply=_apply_split_document_meta,
    description="Move mutable document attributes into the document_meta table",
)

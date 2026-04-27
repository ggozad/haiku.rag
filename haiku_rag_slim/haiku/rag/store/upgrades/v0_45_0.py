import base64
import binascii
import json
import logging

import pyarrow as pa

from haiku.rag.store.engine import DocumentItemRecord, Store
from haiku.rag.store.upgrades import Upgrade
from haiku.rag.utils import escape_sql_string

logger = logging.getLogger(__name__)

BATCH_SIZE = 5


async def _ensure_picture_data_column(store: Store) -> None:
    """Add ``document_items.picture_data`` (large_binary) if it doesn't exist.

    Idempotent: re-runs and fresh DBs that already declare the column via
    ``_init_tables`` see the column in the schema and skip.
    """
    arrow_schema = await store.document_items_table.schema()
    if any(field.name == "picture_data" for field in arrow_schema):
        return
    logger.info("Adding picture_data column to document_items table")
    await store.document_items_table.add_columns(
        pa.schema([pa.field("picture_data", pa.large_binary())])
    )


async def _apply_extract_picture_bytes(store: Store) -> None:  # pragma: no cover
    """Add the ``picture_data`` column to ``document_items`` (if missing),
    backfill it from existing docling blobs, and strip the inline picture
    URIs out of those blobs.

    Idempotent: documents whose blob already has every ``pictures[i].image``
    set to ``None`` (already migrated, or never had bytes) are skipped without
    error. Documents missing matching items rows (legacy DBs that predated the
    v0.40.0 migration) are also skipped — running the v0.40.0 migration first
    populates the rows, then this one fills in the bytes.
    """
    from haiku.rag.store.compression import compress_json, decompress_json

    await _ensure_picture_data_column(store)

    ids = (await store.documents_table.query().select(["id"]).to_arrow()).to_pylist()
    ids = [row["id"] for row in ids]

    if not ids:
        logger.info("No documents to backfill picture_data for")
        return

    total = len(ids)
    logger.info(
        "Backfilling picture_data and stripping URIs across %d documents", total
    )
    backfilled = 0
    blob_only = 0
    skipped = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_ids = ids[batch_start : batch_start + BATCH_SIZE]
        for doc_id in batch_ids:
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
            blob = rows[0].get("docling_document")
            if not isinstance(blob, bytes):
                skipped += 1
                continue

            try:
                data = json.loads(decompress_json(blob))
            except Exception:
                logger.warning(
                    "Could not decompress docling blob for document %s; skipping",
                    doc_id,
                )
                skipped += 1
                continue

            pictures = data.get("pictures") or []
            updates: list[tuple[str, bytes]] = []
            modified = False
            for picture in pictures:
                if not isinstance(picture, dict):
                    continue
                self_ref = picture.get("self_ref")
                image = picture.get("image")
                if image is None or self_ref is None:
                    continue
                # We're going to strip every picture image from the blob.
                # Whether or not we successfully decode the URI to bytes, the
                # blob gets normalised. modified=True for any picture that
                # had a non-null image — that signals "blob will change".
                modified = True
                uri = image.get("uri") if isinstance(image, dict) else None
                if isinstance(uri, str) and uri.startswith("data:"):
                    try:
                        _, encoded = uri.split(",", 1)
                        updates.append((self_ref, base64.b64decode(encoded)))
                    except (ValueError, binascii.Error):
                        pass
                picture["image"] = None

            if not modified:
                # Already stripped (idempotent re-run) or no pictures present.
                continue

            wrote_items = False
            if updates:
                # Find the matching items rows so we can preserve their
                # position/label/text/page_numbers and just attach bytes.
                self_refs = [u[0] for u in updates]
                ref_clause = ", ".join(f"'{escape_sql_string(r)}'" for r in self_refs)
                existing_items = await (
                    store.document_items_table.query()
                    .where(f"document_id = '{safe_id}' AND self_ref IN ({ref_clause})")
                    .to_list()
                )
                existing_by_ref = {r["self_ref"]: r for r in existing_items}

                new_records: list[DocumentItemRecord] = []
                for ref, img_bytes in updates:
                    existing = existing_by_ref.get(ref)
                    if existing is None:
                        # No item row for this self_ref — likely a legacy DB
                        # where v0.40.0 didn't run. Skip; rerun migrations.
                        continue
                    new_records.append(
                        DocumentItemRecord(
                            document_id=doc_id,
                            position=existing["position"],
                            self_ref=ref,
                            label=existing.get("label", ""),
                            text=existing.get("text", ""),
                            page_numbers=existing.get("page_numbers", "[]"),
                            picture_data=img_bytes,
                        )
                    )

                if new_records:
                    await (
                        store.document_items_table.merge_insert(
                            ["document_id", "self_ref"]
                        )
                        .when_matched_update_all()
                        .when_not_matched_insert_all()
                        .execute(new_records)
                    )
                    wrote_items = True

            new_structure_bytes = compress_json(json.dumps(data))
            await store.documents_table.update(
                {"docling_document": new_structure_bytes},
                where=f"id = '{safe_id}'",
            )

            if wrote_items:
                backfilled += 1
            else:
                blob_only += 1

            done = batch_start + (batch_ids.index(doc_id) + 1)
            if done % 10 == 0 or done == total:
                logger.info(
                    "Progress: %d/%d (%d backfilled, %d blob-only, %d skipped)",
                    done,
                    total,
                    backfilled,
                    blob_only,
                    skipped,
                )

    logger.info(
        "Picture backfill complete: %d backfilled, %d blob-only, %d skipped of %d",
        backfilled,
        blob_only,
        skipped,
        total,
    )


upgrade_extract_picture_bytes = Upgrade(
    version="0.45.0",
    apply=_apply_extract_picture_bytes,
    description="Add picture_data column, backfill from docling_document, strip picture URIs",
)

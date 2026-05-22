import json
import logging

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade
from haiku.rag.utils import escape_sql_string

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 50


def _normalize_metadata(meta: dict) -> tuple[dict, bool]:
    """Return (normalized_metadata, changed). Renames `etag` → `source_revision`
    and `contentType` → `content_type`. Pre-existing canonical keys win on
    conflict so repeat migrations are no-ops."""
    changed = False
    out = dict(meta)

    if "etag" in out:
        if "source_revision" not in out:
            out["source_revision"] = out["etag"]
        del out["etag"]
        changed = True

    if "contentType" in out:
        if "content_type" not in out:
            out["content_type"] = out["contentType"]
        del out["contentType"]
        changed = True

    return out, changed


async def _apply_canonical_metadata_keys(store: Store) -> None:
    """Rewrite document.metadata so revision lives under the source-agnostic
    `source_revision` key and `contentType` becomes `content_type`. Documents
    whose metadata already uses the canonical keys are untouched."""
    rows = (
        await store.documents_table.query().select(["id", "metadata"]).to_arrow()
    ).to_pylist()
    total = len(rows)
    logger.info("Normalising document metadata keys across %d documents", total)
    rewritten = 0
    skipped = 0

    for idx, row in enumerate(rows, 1):
        doc_id = row["id"]
        raw = row.get("metadata") or "{}"
        try:
            meta = json.loads(raw)
        except Exception:  # pragma: no cover
            logger.warning(
                "Could not parse metadata JSON for document %s; skipping", doc_id
            )
            skipped += 1
            continue

        normalized, changed = _normalize_metadata(meta)
        if not changed:
            continue

        safe_id = escape_sql_string(doc_id)
        await store.documents_table.update(
            {"metadata": json.dumps(normalized)},
            where=f"id = '{safe_id}'",
        )
        rewritten += 1

        if idx % PROGRESS_INTERVAL == 0 or idx == total:
            logger.info("Progress: %d/%d (%d rewritten)", idx, total, rewritten)

    logger.info(
        "Metadata key normalisation complete: %d rewritten, %d skipped of %d",
        rewritten,
        skipped,
        total,
    )


upgrade_canonical_metadata_keys = Upgrade(
    version="0.50.0",
    apply=_apply_canonical_metadata_keys,
    description="Rename document.metadata keys: etag→source_revision, contentType→content_type",
)

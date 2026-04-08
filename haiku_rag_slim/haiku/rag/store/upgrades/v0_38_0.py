import gzip
import json
import logging
from datetime import timedelta

import pyarrow as pa
from lancedb.pydantic import LanceModel
from pydantic import Field

from haiku.rag.store.compression import compress_docling_split
from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)

BATCH_SIZE = 5


def _apply_split_pages_zstd(store: Store) -> None:  # pragma: no cover
    """Split docling_document into structure + pages and re-compress with zstd."""

    class DocumentRecordV5(LanceModel):
        id: str
        content: str
        uri: str | None = None
        title: str | None = None
        metadata: str = Field(default="{}")
        docling_document: bytes | None = None
        docling_pages: bytes | None = None
        docling_version: str | None = None
        created_at: str = Field(default_factory=lambda: "")
        updated_at: str = Field(default_factory=lambda: "")

    def get_documents_arrow_schema_v5() -> pa.Schema:
        """Generate Arrow schema with large_binary for both docling columns."""
        base_schema = DocumentRecordV5.to_arrow_schema()
        large_binary_columns = {"docling_document", "docling_pages"}
        fields = []
        for field in base_schema:
            if field.name in large_binary_columns:
                fields.append(pa.field(field.name, pa.large_binary()))
            else:
                fields.append(field)
        return pa.schema(fields)

    def migrate_row(row: dict) -> DocumentRecordV5:
        """Migrate a single row: decompress gzip, split pages, re-compress with zstd."""
        docling_blob = row.get("docling_document")
        structure_bytes: bytes | None = None
        pages_bytes: bytes | None = None

        if docling_blob and isinstance(docling_blob, bytes):
            # Decompress from gzip
            try:
                json_str = gzip.decompress(docling_blob).decode("utf-8")
            except Exception:
                # May already be zstd or uncompressed — try as-is
                json_str = docling_blob.decode("utf-8")

            # Split structure and pages, re-compress with zstd
            structure_bytes, pages_bytes = compress_docling_split(json_str)

        metadata_raw = row.get("metadata")
        metadata_str = (
            metadata_raw
            if isinstance(metadata_raw, str)
            else json.dumps(metadata_raw or {})
        )

        return DocumentRecordV5(
            id=row.get("id") or "",
            content=row.get("content", ""),
            uri=row.get("uri"),
            title=row.get("title"),
            metadata=metadata_str,
            docling_document=structure_bytes,
            docling_pages=pages_bytes,
            docling_version=row.get("docling_version"),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )

    def copy_staging_row(row: dict) -> DocumentRecordV5:
        """Copy a row from the staging table (already migrated)."""
        return DocumentRecordV5(
            id=row["id"],
            content=row["content"],
            uri=row["uri"],
            title=row["title"],
            metadata=row["metadata"],
            docling_document=row["docling_document"],
            docling_pages=row["docling_pages"],
            docling_version=row["docling_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    staging_name = "documents_v5_staging"

    # First pass: collect document IDs to process
    try:
        ids = [
            row["id"]
            for row in store.documents_table.search()
            .select(["id"])
            .to_arrow()
            .to_pylist()
        ]
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, OSError):
        ids = []

    if not ids:
        # Check for staging table from a failed migration
        if staging_name in store.db.list_tables().tables:
            staging_table = store.db.open_table(staging_name)
            staging_ids = [
                row["id"]
                for row in staging_table.search().select(["id"]).to_arrow().to_pylist()
            ]
            if staging_ids:
                logger.info(
                    "Recovering %d documents from failed migration", len(staging_ids)
                )
                store.documents_table = None
                if "documents" in store.db.list_tables().tables:
                    store.db.drop_table("documents")
                store.documents_table = store.db.create_table(
                    "documents", schema=get_documents_arrow_schema_v5()
                )
                total_batches = (len(staging_ids) + BATCH_SIZE - 1) // BATCH_SIZE
                for batch_num, i in enumerate(
                    range(0, len(staging_ids), BATCH_SIZE), 1
                ):
                    batch_ids = staging_ids[i : i + BATCH_SIZE]
                    id_list = ", ".join(f"'{doc_id}'" for doc_id in batch_ids)
                    batch = (
                        staging_table.search()
                        .where(f"id IN ({id_list})")
                        .to_arrow()
                        .to_pylist()
                    )
                    records = [copy_staging_row(row) for row in batch]
                    if records:
                        store.documents_table.add(records)
                        logger.info("Recovered batch %d/%d", batch_num, total_batches)
                store.db.drop_table(staging_name)
                logger.info("Recovery complete")
                return

        # No documents and no staging — recreate table with new schema
        store.documents_table = None
        if "documents" in store.db.list_tables().tables:
            store.db.drop_table("documents")
        store.documents_table = store.db.create_table(
            "documents", schema=get_documents_arrow_schema_v5()
        )
        return

    # Create staging table with new schema
    if staging_name in store.db.list_tables().tables:
        store.db.drop_table(staging_name)
    staging_table = store.db.create_table(
        staging_name, schema=get_documents_arrow_schema_v5()
    )

    # Migrate in batches: read from old, split+recompress, write to staging
    total_docs = len(ids)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(
        "Splitting pages and re-compressing %d documents in %d batches",
        total_docs,
        total_batches,
    )

    for batch_num, i in enumerate(range(0, len(ids), BATCH_SIZE), 1):
        batch_ids = ids[i : i + BATCH_SIZE]
        id_list = ", ".join(f"'{doc_id}'" for doc_id in batch_ids)

        batch = (
            store.documents_table.search()
            .where(f"id IN ({id_list})")
            .to_arrow()
            .to_pylist()
        )

        migrated_batch = [migrate_row(row) for row in batch]
        if migrated_batch:
            staging_table.add(migrated_batch)

        logger.info(
            "Migrated batch %d/%d (%d documents)",
            batch_num,
            total_batches,
            len(migrated_batch),
        )

    # Replace old table with staging table
    store.documents_table = None
    if "documents" in store.db.list_tables().tables:
        store.db.drop_table("documents")
    store.documents_table = store.db.create_table(
        "documents", schema=get_documents_arrow_schema_v5()
    )

    # Copy from staging to final table in batches
    staging_ids = [
        row["id"]
        for row in staging_table.search().select(["id"]).to_arrow().to_pylist()
    ]

    logger.info("Copying %d documents to new table", len(staging_ids))

    for batch_num, i in enumerate(range(0, len(staging_ids), BATCH_SIZE), 1):
        batch_ids = staging_ids[i : i + BATCH_SIZE]
        id_list = ", ".join(f"'{doc_id}'" for doc_id in batch_ids)

        batch = (
            staging_table.search().where(f"id IN ({id_list})").to_arrow().to_pylist()
        )
        records = [copy_staging_row(row) for row in batch]
        if records:
            store.documents_table.add(records)
            logger.info("Copied batch %d/%d", batch_num, total_batches)

    # Cleanup staging table
    if staging_name in store.db.list_tables().tables:
        store.db.drop_table(staging_name)

    # Vacuum all tables
    logger.info("Vacuuming database")
    for table in [store.documents_table, store.chunks_table, store.settings_table]:
        try:
            table.optimize(cleanup_older_than=timedelta(seconds=0))
        except Exception:
            pass

    logger.info("Migration complete")


upgrade_split_pages_zstd = Upgrade(
    version="0.38.0",
    apply=_apply_split_pages_zstd,
    description="Split docling_document pages into separate column and re-compress with zstd",
)

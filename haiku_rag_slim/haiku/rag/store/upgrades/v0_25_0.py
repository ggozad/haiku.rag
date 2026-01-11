import json
import logging
from datetime import timedelta

import pyarrow as pa
from lancedb.pydantic import LanceModel
from pydantic import Field

from haiku.rag.store.compression import compress_json, decompress_json
from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade

logger = logging.getLogger(__name__)

BATCH_SIZE = 10


def _apply_compress_docling_document(store: Store) -> None:  # pragma: no cover
    """Migrate docling_document_json (str) to docling_document (compressed bytes)."""

    class DocumentRecordV4(LanceModel):
        id: str
        content: str
        uri: str | None = None
        title: str | None = None
        metadata: str = Field(default="{}")
        docling_document: bytes | None = None
        docling_version: str | None = None
        created_at: str = Field(default_factory=lambda: "")
        updated_at: str = Field(default_factory=lambda: "")

    def get_documents_arrow_schema_v4() -> pa.Schema:
        """Generate Arrow schema with large_binary for docling_document."""
        base_schema = DocumentRecordV4.to_arrow_schema()
        fields = []
        for field in base_schema:
            if field.name == "docling_document":
                fields.append(pa.field("docling_document", pa.large_binary()))
            else:
                fields.append(field)
        return pa.schema(fields)

    def migrate_row(row: dict) -> DocumentRecordV4:
        """Migrate a single row, compressing docling_document."""
        docling_json = row.get("docling_document_json") or row.get("docling_document")
        docling_bytes: bytes | None = None

        if docling_json:
            if isinstance(docling_json, str):
                docling_bytes = compress_json(docling_json)
            elif isinstance(docling_json, bytes):
                try:
                    decompress_json(docling_json)
                    docling_bytes = docling_json  # Already compressed
                except Exception:
                    docling_bytes = compress_json(docling_json.decode("utf-8"))

        metadata_raw = row.get("metadata")
        metadata_str = (
            metadata_raw
            if isinstance(metadata_raw, str)
            else json.dumps(metadata_raw or {})
        )

        return DocumentRecordV4(
            id=row.get("id") or "",
            content=row.get("content", ""),
            uri=row.get("uri"),
            title=row.get("title"),
            metadata=metadata_str,
            docling_document=docling_bytes,
            docling_version=row.get("docling_version"),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )

    # First pass: collect document IDs to process
    try:
        ids = [
            row["id"]
            for row in store.documents_table.search()
            .select(["id"])
            .to_arrow()
            .to_pylist()
        ]
    except Exception:
        ids = []

    if not ids:
        # No documents, just recreate table with new schema
        try:
            store.db.drop_table("documents")
        except Exception:
            pass
        store.documents_table = store.db.create_table(
            "documents", schema=get_documents_arrow_schema_v4()
        )
        return

    # Create staging table with new schema
    try:
        store.db.drop_table("documents_v4_staging")
    except Exception:
        pass
    staging_table = store.db.create_table(
        "documents_v4_staging", schema=get_documents_arrow_schema_v4()
    )

    # Migrate in batches: read from old, compress, write to staging
    total_docs = len(ids)
    total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("Compressing %d documents in %d batches", total_docs, total_batches)

    for batch_num, i in enumerate(range(0, len(ids), BATCH_SIZE), 1):
        batch_ids = ids[i : i + BATCH_SIZE]
        id_list = ", ".join(f"'{id}'" for id in batch_ids)

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
            "Compressed batch %d/%d (%d documents)",
            batch_num,
            total_batches,
            len(migrated_batch),
        )

    # Replace old table with staging table
    try:
        store.db.drop_table("documents")
    except Exception:
        pass

    store.documents_table = store.db.create_table(
        "documents", schema=get_documents_arrow_schema_v4()
    )

    # Copy from staging to final table in batches
    staging_ids = [
        row["id"]
        for row in staging_table.search().select(["id"]).to_arrow().to_pylist()
    ]

    logger.info("Copying %d documents to new table", len(staging_ids))

    for batch_num, i in enumerate(range(0, len(staging_ids), BATCH_SIZE), 1):
        batch_ids = staging_ids[i : i + BATCH_SIZE]
        id_list = ", ".join(f"'{id}'" for id in batch_ids)

        batch = (
            staging_table.search().where(f"id IN ({id_list})").to_arrow().to_pylist()
        )
        records = [
            DocumentRecordV4(
                id=row["id"],
                content=row["content"],
                uri=row["uri"],
                title=row["title"],
                metadata=row["metadata"],
                docling_document=row["docling_document"],
                docling_version=row["docling_version"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in batch
        ]
        if records:
            store.documents_table.add(records)
            logger.info("Copied batch %d/%d", batch_num, total_batches)

    # Cleanup staging table
    try:
        store.db.drop_table("documents_v4_staging")
    except Exception:
        pass

    # Vacuum all tables (destructive migration, no history preserved)
    logger.info("Vacuuming database")
    for table in [store.documents_table, store.chunks_table, store.settings_table]:
        try:
            table.optimize(cleanup_older_than=timedelta(seconds=0))
        except Exception:
            pass

    logger.info("Migration complete")


upgrade_compress_docling_document = Upgrade(
    version="0.25.0",
    apply=_apply_compress_docling_document,
    description="Compress docling_document with gzip and use large_binary type",
)

import json

from lancedb.pydantic import LanceModel
from pydantic import Field

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade


def _apply_add_docling_document_columns(store: Store) -> None:  # pragma: no cover
    """Add 'docling_document_json' and 'docling_version' columns to documents table."""

    # Read existing rows using Arrow for schema-agnostic access
    try:
        docs_arrow = store.documents_table.search().to_arrow()
        rows = docs_arrow.to_pylist()
    except Exception:
        rows = []

    class DocumentRecordV3(LanceModel):
        id: str
        content: str
        uri: str | None = None
        title: str | None = None
        metadata: str = Field(default="{}")
        docling_document_json: str | None = None
        docling_version: str | None = None
        created_at: str = Field(default_factory=lambda: "")
        updated_at: str = Field(default_factory=lambda: "")

    # Drop and recreate documents table with the new schema
    try:
        store.db.drop_table("documents")
    except Exception:
        pass

    store.documents_table = store.db.create_table("documents", schema=DocumentRecordV3)

    # Reinsert previous rows with new columns as None
    if rows:
        backfilled = []
        for row in rows:
            backfilled.append(
                DocumentRecordV3(
                    id=row.get("id"),
                    content=row.get("content", ""),
                    uri=row.get("uri"),
                    title=row.get("title"),
                    metadata=(
                        row.get("metadata")
                        if isinstance(row.get("metadata"), str)
                        else json.dumps(row.get("metadata") or {})
                    ),
                    docling_document_json=None,
                    docling_version=None,
                    created_at=row.get("created_at", ""),
                    updated_at=row.get("updated_at", ""),
                )
            )

        store.documents_table.add(backfilled)


upgrade_add_docling_document = Upgrade(
    version="0.20.0",
    apply=_apply_add_docling_document_columns,
    description="Add 'docling_document_json' and 'docling_version' columns to documents table",
)

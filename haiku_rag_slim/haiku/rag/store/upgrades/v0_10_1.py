import json

from lancedb.pydantic import LanceModel
from pydantic import Field

from haiku.rag.store.engine import Store
from haiku.rag.store.upgrades import Upgrade


def _apply_add_document_title(store: Store) -> None:
    """Add a nullable 'title' column to the documents table."""

    # Read existing rows using Arrow for schema-agnostic access
    try:
        docs_arrow = store.documents_table.search().to_arrow()
        rows = docs_arrow.to_pylist()
    except Exception:
        rows = []

    class DocumentRecordV2(LanceModel):
        id: str
        content: str
        uri: str | None = None
        title: str | None = None
        metadata: str = Field(default="{}")
        created_at: str = Field(default_factory=lambda: "")
        updated_at: str = Field(default_factory=lambda: "")

    # Drop and recreate documents table with the new schema
    try:
        store.db.drop_table("documents")
    except Exception:
        pass

    store.documents_table = store.db.create_table("documents", schema=DocumentRecordV2)

    # Reinsert previous rows with title=None
    if rows:
        backfilled = []
        for row in rows:
            backfilled.append(
                DocumentRecordV2(
                    id=row.get("id"),
                    content=row.get("content", ""),
                    uri=row.get("uri"),
                    title=None,
                    metadata=(
                        row.get("metadata")
                        if isinstance(row.get("metadata"), str)
                        else json.dumps(row.get("metadata") or {})
                    ),
                    created_at=row.get("created_at", ""),
                    updated_at=row.get("updated_at", ""),
                )
            )

        store.documents_table.add(backfilled)


upgrade_add_title = Upgrade(
    version="0.10.1",
    apply=_apply_add_document_title,
    description="Add nullable 'title' column to documents table",
)

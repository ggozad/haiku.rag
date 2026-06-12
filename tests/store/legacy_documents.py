"""Helpers to seed a `documents` table in its pre-0.58 shape.

Before the document_meta split (v0.58.0), the `documents` table carried
`uri/title/metadata/created_at/updated_at` alongside the content+blobs. The
migration-chain tests need to reproduce that legacy layout so the older
migrations (which read `documents.metadata`, etc.) and v0.58.0 itself have the
columns they operate on.
"""

from uuid import uuid4

import pyarrow as pa
from lancedb.pydantic import LanceModel
from pydantic import Field

from haiku.rag.store.engine import Store


class LegacyDocumentRecord(LanceModel):
    """The pre-0.58 `documents` record (mutable attributes still inline)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: str = Field(default="{}")
    docling_document: bytes | None = None
    docling_pages: bytes | None = None
    docling_version: str | None = None
    created_at: str = Field(default_factory=lambda: "")
    updated_at: str = Field(default_factory=lambda: "")


def legacy_documents_schema() -> pa.Schema:
    base = LegacyDocumentRecord.to_arrow_schema()
    large_binary_columns = {"docling_document", "docling_pages"}
    return pa.schema(
        [
            pa.field(f.name, pa.large_binary()) if f.name in large_binary_columns else f
            for f in base
        ]
    )


async def seed_legacy_documents(
    store: Store, records: list[LegacyDocumentRecord]
) -> None:
    """Recreate the `documents` table with the pre-0.58 schema and add records,
    simulating a database created before the document_meta split."""
    if "documents" in (await store.db.list_tables()).tables:
        await store.db.drop_table("documents")
    store.documents_table = await store.db.create_table(
        "documents", schema=legacy_documents_schema()
    )
    if records:
        await store.documents_table.add(records)

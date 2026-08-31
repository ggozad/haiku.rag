"""Helpers to seed a `documents` table in the shapes it had before 0.58.

Each migration in the early chain rewrites the whole `documents` table, so its
tests need the table as the *previous* version left it. The record classes here
are those frozen shapes, named for the schema version the migration chain gives
them: V2 predates docling, V3 carries it as JSON text (v0.20.0), V4 as one
compressed blob (v0.25.0), and `LegacyDocumentRecord` is V5, the split-blob
shape that stood until the document_meta split (v0.58.0).
"""

from uuid import uuid4

import pyarrow as pa
from lancedb.pydantic import LanceModel
from pydantic import Field

from haiku.rag.store.engine import Store


class DocumentRecordV2(LanceModel):
    """The pre-0.20 `documents` record, before any docling column."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: str = Field(default="{}")
    created_at: str = Field(default_factory=lambda: "")
    updated_at: str = Field(default_factory=lambda: "")


class DocumentRecordV3(LanceModel):
    """The 0.20.0 `documents` record, with docling stored as JSON text."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: str = Field(default="{}")
    docling_document_json: str | None = None
    docling_version: str | None = None
    created_at: str = Field(default_factory=lambda: "")
    updated_at: str = Field(default_factory=lambda: "")


class DocumentRecordV4(LanceModel):
    """The 0.25.0 `documents` record, with docling as one compressed blob."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    uri: str | None = None
    title: str | None = None
    metadata: str = Field(default="{}")
    docling_document: bytes | None = None
    docling_version: str | None = None
    created_at: str = Field(default_factory=lambda: "")
    updated_at: str = Field(default_factory=lambda: "")


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


def documents_schema(model: type[LanceModel]) -> pa.Schema:
    """Arrow schema for a historical record, with large_binary docling columns."""
    blobs = {"docling_document", "docling_pages"}
    return pa.schema(
        [
            pa.field(f.name, pa.large_binary()) if f.name in blobs else f
            for f in model.to_arrow_schema()
        ]
    )


def legacy_documents_schema() -> pa.Schema:
    return documents_schema(LegacyDocumentRecord)


async def seed_documents(store: Store, schema: pa.Schema, records: list) -> None:
    """Recreate the `documents` table with a historical schema and add records."""
    if "documents" in (await store.db.list_tables()).tables:
        await store.db.drop_table("documents")
    store.documents_table = await store.db.create_table("documents", schema=schema)
    if records:
        await store.documents_table.add(records)


async def seed_legacy_documents(
    store: Store, records: list[LegacyDocumentRecord]
) -> None:
    """Recreate the `documents` table with the pre-0.58 schema and add records,
    simulating a database created before the document_meta split."""
    await seed_documents(store, legacy_documents_schema(), records)

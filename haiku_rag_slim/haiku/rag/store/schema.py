"""Table records, Arrow schemas and index specifications.

Describes what the tables are; nothing here opens a connection, mutates a
table, or imports a client layer.
"""

import logging
from typing import cast
from uuid import uuid4

import lancedb
import pyarrow as pa
from lancedb.index import FTS, Bitmap, BTree
from lancedb.pydantic import LanceModel, Vector
from lancedb.query import AsyncQueryBase
from pydantic import Field

logger = logging.getLogger(__name__)


async def query_to_pydantic[T: LanceModel](
    query: "AsyncQueryBase", model: type[T]
) -> list[T]:
    """Typed wrapper around AsyncQueryBase.to_pydantic.

    The upstream stub annotates `.to_pydantic()` as returning `list[LanceModel]`
    regardless of the concrete model passed in. This helper narrows the return
    type to the concrete model so attribute access on the results type-checks
    at call sites without needing per-line cast / ignore comments.
    """
    return cast("list[T]", await query.to_pydantic(model))


class DocumentRecord(LanceModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    docling_document: bytes | None = None
    docling_pages: bytes | None = None
    docling_version: str | None = None


class DocumentMetaRecord(LanceModel):
    """Mutable, lightweight document attributes, kept separate from the
    write-once content/blobs in `documents`. Updating these (metadata, title,
    source_revision) must not rewrite the multi-MB docling row."""

    id: str
    uri: str | None = None
    title: str | None = None
    metadata: str = Field(default="{}")
    created_at: str = Field(default_factory=lambda: "")
    updated_at: str = Field(default_factory=lambda: "")


def get_documents_arrow_schema() -> pa.Schema:
    """Generate Arrow schema for documents table with large_binary for docling_document.

    LanceDB maps Python `bytes` to Arrow's `binary` type, which uses 32-bit offsets
    and is limited to ~2GB per column in a fragment. When many large documents
    (with embedded page images) are grouped in a single fragment, this limit is
    exceeded, causing "byte array offset overflow" panics.

    This function overrides the default mapping to use `large_binary` instead,
    which has 64-bit offsets and no practical size limit.
    """
    base_schema = DocumentRecord.to_arrow_schema()
    large_binary_columns = {"docling_document", "docling_pages"}
    fields = []
    for field in base_schema:
        if field.name in large_binary_columns:
            fields.append(pa.field(field.name, pa.large_binary()))
        else:
            fields.append(field)
    return pa.schema(fields)


class ChunkRecordBase(LanceModel):
    """Static base for ChunkRecord — declares the fields so attribute access
    and constructor calls type-check. The concrete `vector` field is overridden
    by create_chunk_model() with a Vector(dim) whose fixed-size-list dimension
    is only known at runtime.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    content_fts: str = Field(default="")
    metadata: str = Field(default="{}")
    order: int = Field(default=0)
    vector: list[float] = Field(default_factory=list)


def create_chunk_model(vector_dim: int) -> type[ChunkRecordBase]:
    """Create a ChunkRecord model with the specified vector dimension."""

    class ChunkRecord(ChunkRecordBase):
        vector: Vector(vector_dim) = Field(default_factory=lambda: [0.0] * vector_dim)  # type: ignore

    return ChunkRecord


class DocumentItemRecord(LanceModel):
    document_id: str
    position: int
    self_ref: str
    label: str = Field(default="")
    text: str = Field(default="")
    page_numbers: str = Field(default="[]")
    picture_data: bytes | None = None
    heading_level: int = Field(default=0)
    tree_depth: int = Field(default=0)


def get_document_items_arrow_schema() -> pa.Schema:
    """Generate Arrow schema for document_items with large_binary for picture_data.

    LanceDB maps Python `bytes` to Arrow's `binary` type, which uses 32-bit offsets
    and is limited to ~2GB per column in a fragment. Many embedded picture PNGs in
    one fragment can exceed that limit. `large_binary` uses 64-bit offsets and has
    no practical size limit — same reasoning as `docling_document` on the
    documents table.
    """
    base_schema = DocumentItemRecord.to_arrow_schema()
    large_binary_columns = {"picture_data"}
    fields = []
    for field in base_schema:
        if field.name in large_binary_columns:
            fields.append(pa.field(field.name, pa.large_binary()))
        else:
            fields.append(field)
    return pa.schema(fields)


def index_specs(table_name: str) -> list[tuple[str, Bitmap | BTree | FTS]]:
    """The index set each table carries."""
    match table_name:
        case "documents":
            return [("id", BTree())]
        case "document_meta":
            return [("id", BTree()), ("uri", BTree())]
        case "chunks":
            return [
                # Positions and stop words are required for phrase queries.
                ("content_fts", FTS(with_position=True, remove_stop_words=False)),
                ("id", BTree()),
                ("document_id", BTree()),
            ]
        case "document_items":
            return [
                ("document_id", BTree()),
                ("position", BTree()),
                ("self_ref", BTree()),
                ("label", Bitmap()),
            ]
        case _:
            return []


async def ensure_indexes(table: lancedb.AsyncTable, table_name: str) -> list[str]:
    """Create any declared index missing from a column. Returns the columns indexed.

    Matches on index type, not column coverage, so a BTree does not satisfy a
    declared Bitmap. Never drops or converts an index it did not declare.
    Re-creating is not free: `create_index(replace=True)` rebuilds.
    """
    covering: dict[str, set[str]] = {}
    for index in await table.list_indices():
        for column in index.columns:
            covering.setdefault(column, set()).add(index.index_type)

    applied: list[str] = []
    for column, config in index_specs(table_name):
        declared = type(config).__name__
        present = covering.get(column, set())
        if declared in present:
            continue
        if present:
            logger.info(
                f"Adding {declared} index on {table_name}.{column}, which carries "
                f"{', '.join(sorted(present))}"
            )
        await table.create_index(column, config=config, replace=True)
        applied.append(column)
    return applied


class SettingsRecord(LanceModel):
    id: str = Field(default="settings")
    settings: str = Field(default="{}")


REQUIRED_TABLES: tuple[str, ...] = (
    "documents",
    "document_meta",
    "chunks",
    "document_items",
    "settings",
)

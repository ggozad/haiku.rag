"""Tests for the v0.45.0 picture-data backfill migration.

The migration walks every document, decodes picture image URIs out of the
stored docling blob, writes the bytes to ``document_items.picture_data``,
and then strips the URIs from the blob (which lives separately on the
documents table).
"""

import base64
import json

import pytest

from haiku.rag.store import Store
from haiku.rag.store.compression import compress_json, decompress_json
from haiku.rag.store.engine import DocumentItemRecord, DocumentRecord
from haiku.rag.store.upgrades.v0_45_0 import _apply_extract_picture_bytes

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-picture-bytes"
PNG_DATA_URI = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"


def _docling_blob_with_picture(self_ref: str = "#/pictures/0") -> bytes:
    """Build a compressed docling-document blob carrying one picture with
    an inline data URI — the shape v0.45.0 was written to backfill from."""
    doc = {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "name": "test",
        "pictures": [
            {
                "self_ref": self_ref,
                "image": {"mimetype": "image/png", "uri": PNG_DATA_URI},
            }
        ],
        "tables": [],
        "texts": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": [], "label": "unspecified"},
        "furniture": {
            "self_ref": "#/furniture",
            "children": [],
            "label": "unspecified",
        },
    }
    return compress_json(json.dumps(doc))


@pytest.mark.asyncio
async def test_migration_backfills_picture_bytes_and_strips_blob(temp_db_path):
    """Happy path: doc has a picture with bytes inline in the blob plus a
    matching items row (with picture_data NULL). The migration writes the
    bytes onto the items row and clears the URI from the blob."""
    async with Store(temp_db_path, create=True) as store:
        doc_id = "doc-1"
        # Insert a doc whose blob carries a picture data URI.
        await store.documents_table.add(
            [
                DocumentRecord(
                    id=doc_id,
                    content="x",
                    docling_document=_docling_blob_with_picture(),
                    docling_version="1.10.0",
                )
            ]
        )
        # Insert a matching items row, picture_data deliberately empty.
        await store.document_items_table.add(
            [
                DocumentItemRecord(
                    document_id=doc_id,
                    position=0,
                    self_ref="#/pictures/0",
                    label="picture",
                    text="",
                    page_numbers="[1]",
                    picture_data=None,
                )
            ]
        )

        await _apply_extract_picture_bytes(store)

        # picture_data is populated on the items row.
        rows = await (
            store.document_items_table.query()
            .where(f"document_id = '{doc_id}' AND self_ref = '#/pictures/0'")
            .to_list()
        )
        assert len(rows) == 1
        assert rows[0]["picture_data"] == PNG_BYTES

        # The docling blob no longer carries the data URI on the picture.
        doc_rows = await (
            store.documents_table.query()
            .where(f"id = '{doc_id}'")
            .select(["docling_document"])
            .to_list()
        )
        blob = json.loads(decompress_json(doc_rows[0]["docling_document"]))
        assert blob["pictures"][0]["image"] is None


@pytest.mark.asyncio
async def test_migration_is_idempotent(temp_db_path):
    """Running the migration twice on the same DB is a no-op the second
    time — the blob is already stripped."""
    async with Store(temp_db_path, create=True) as store:
        doc_id = "doc-1"
        await store.documents_table.add(
            [
                DocumentRecord(
                    id=doc_id,
                    content="x",
                    docling_document=_docling_blob_with_picture(),
                    docling_version="1.10.0",
                )
            ]
        )
        await store.document_items_table.add(
            [
                DocumentItemRecord(
                    document_id=doc_id,
                    position=0,
                    self_ref="#/pictures/0",
                    label="picture",
                    text="",
                    page_numbers="[1]",
                )
            ]
        )

        await _apply_extract_picture_bytes(store)
        # Capture state after first run.
        rows_first = await (
            store.document_items_table.query()
            .where(f"document_id = '{doc_id}'")
            .to_list()
        )
        first_bytes = rows_first[0]["picture_data"]

        # Re-run.
        await _apply_extract_picture_bytes(store)

        rows_second = await (
            store.document_items_table.query()
            .where(f"document_id = '{doc_id}'")
            .to_list()
        )
        assert rows_second[0]["picture_data"] == first_bytes

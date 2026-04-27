import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.documents import (
    _store_document_with_chunks,
    _update_document_with_chunks,
)
from haiku.rag.store.engine import Store
from haiku.rag.store.models.document_item import (
    DocumentItem,
    extract_item_text,
    extract_items,
)
from haiku.rag.store.repositories.document_item import DocumentItemRepository


def _make_docling_doc():
    """Create a DoclingDocument with mixed item types for testing."""
    from docling_core.types.doc.document import DoclingDocument, TableData
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="test")
    doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Introduction")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="This is the first paragraph.")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="This is the second paragraph.")
    doc.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=[]))
    doc.add_text(label=DocItemLabel.SECTION_HEADER, text="Conclusion")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Final thoughts here.")
    return doc


class TestExtractItems:
    def test_extracts_all_items(self):
        doc = _make_docling_doc()
        items = extract_items("doc-1", doc)

        assert len(items) == 6
        assert all(item.document_id == "doc-1" for item in items)
        assert [item.position for item in items] == [0, 1, 2, 3, 4, 5]

    def test_extracts_labels(self):
        doc = _make_docling_doc()
        items = extract_items("doc-1", doc)

        assert items[0].label == "section_header"
        assert items[1].label == "paragraph"
        assert items[3].label == "table"
        assert items[4].label == "section_header"

    def test_extracts_text(self):
        doc = _make_docling_doc()
        items = extract_items("doc-1", doc)

        assert items[0].text == "Introduction"
        assert items[1].text == "This is the first paragraph."
        assert items[5].text == "Final thoughts here."

    def test_extracts_self_refs(self):
        doc = _make_docling_doc()
        items = extract_items("doc-1", doc)

        assert all(item.self_ref.startswith("#/") for item in items)

    def test_table_gets_markdown_text(self):
        doc = _make_docling_doc()
        items = extract_items("doc-1", doc)

        table_item = items[3]
        assert table_item.label == "table"
        # Table should have some text from export_to_markdown
        assert isinstance(table_item.text, str)


class TestExtractItemText:
    def test_text_item(self):
        from docling_core.types.doc.document import DoclingDocument
        from docling_core.types.doc.labels import DocItemLabel

        doc = DoclingDocument(name="test")
        doc.add_text(label=DocItemLabel.PARAGRAPH, text="Hello world")
        item, _ = next(iter(doc.iterate_items()))
        assert extract_item_text(item, doc) == "Hello world"

    def test_returns_none_for_empty_item(self):
        from docling_core.types.doc.document import DoclingDocument

        doc = DoclingDocument(name="test")
        # An empty doc has no items to extract text from
        items = extract_items("doc-1", doc)
        assert items == []


@pytest.mark.asyncio
class TestDocumentItemRepository:
    async def test_create_and_get_range(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            repo = DocumentItemRepository(rag.store)

            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=i,
                    self_ref=f"#/texts/{i}",
                    label="paragraph",
                    text=f"Item {i}",
                    page_numbers=[1],
                )
                for i in range(10)
            ]
            await repo.create_items("doc-1", items)

            result = await repo.get_items_in_range("doc-1", 3, 7)
            assert len(result) == 5
            assert result[0].position == 3
            assert result[-1].position == 7
            assert result[0].text == "Item 3"

    async def test_resolve_refs(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            repo = DocumentItemRepository(rag.store)

            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=i,
                    self_ref=f"#/texts/{i}",
                    label="paragraph",
                    text=f"Item {i}",
                )
                for i in range(10)
            ]
            await repo.create_items("doc-1", items)

            refs = await repo.resolve_refs(
                "doc-1", ["#/texts/2", "#/texts/7", "#/texts/999"]
            )
            assert refs == {"#/texts/2": 2, "#/texts/7": 7}

    async def test_get_item_count(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            repo = DocumentItemRepository(rag.store)

            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=i,
                    self_ref=f"#/texts/{i}",
                    label="paragraph",
                    text=f"Item {i}",
                )
                for i in range(15)
            ]
            await repo.create_items("doc-1", items)

            assert await repo.get_item_count("doc-1") == 15
            assert await repo.get_item_count("nonexistent") == 0

    async def test_delete_by_document_id(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            repo = DocumentItemRepository(rag.store)

            for doc_id in ["doc-1", "doc-2"]:
                items = [
                    DocumentItem(
                        document_id=doc_id,
                        position=i,
                        self_ref=f"#/texts/{i}",
                        label="paragraph",
                        text=f"Item {i}",
                    )
                    for i in range(5)
                ]
                await repo.create_items(doc_id, items)

            await repo.delete_by_document_id("doc-1")

            assert await repo.get_item_count("doc-1") == 0
            assert await repo.get_item_count("doc-2") == 5

    async def test_empty_refs_returns_empty(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            repo = DocumentItemRepository(rag.store)
            assert await repo.resolve_refs("doc-1", []) == {}

    async def test_items_sorted_by_position(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            repo = DocumentItemRepository(rag.store)

            # Insert in reverse order
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=9 - i,
                    self_ref=f"#/texts/{9 - i}",
                    label="paragraph",
                    text=f"Item {9 - i}",
                )
                for i in range(10)
            ]
            await repo.create_items("doc-1", items)

            result = await repo.get_items_in_range("doc-1", 0, 9)
            positions = [item.position for item in result]
            assert positions == sorted(positions)


@pytest.mark.asyncio
class TestDocumentItemPopulation:
    async def test_store_document_populates_items(self, temp_db_path):
        """Test that _store_document_with_chunks populates items when given a docling_document."""
        from haiku.rag.store.models.document import Document

        docling_doc = _make_docling_doc()

        async with HaikuRAG(temp_db_path, create=True) as rag:
            document = Document(
                content="test content",
                uri="test://doc",
            )
            document.set_docling(docling_doc)

            # Use _store_document_with_chunks directly with empty chunks
            # to avoid needing embeddings
            created = await _store_document_with_chunks(rag, document, [], docling_doc)
            assert created.id is not None

            count = await rag.document_item_repository.get_item_count(created.id)
            assert count == 6

            items = await rag.document_item_repository.get_items_in_range(
                created.id, 0, count
            )
            assert items[0].label == "section_header"
            assert items[0].text == "Introduction"
            assert items[1].label == "paragraph"

    async def test_update_document_replaces_items(self, temp_db_path):
        """Test that _update_document_with_chunks replaces items."""
        from docling_core.types.doc.document import DoclingDocument
        from docling_core.types.doc.labels import DocItemLabel

        from haiku.rag.store.models.document import Document

        docling_doc = _make_docling_doc()

        async with HaikuRAG(temp_db_path, create=True) as rag:
            document = Document(
                content="test content",
                uri="test://doc",
            )
            document.set_docling(docling_doc)
            created = await _store_document_with_chunks(rag, document, [], docling_doc)
            assert created.id is not None
            assert await rag.document_item_repository.get_item_count(created.id) == 6

            # Update with a simpler document
            new_doc = DoclingDocument(name="updated")
            new_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Only one item now.")
            created.set_docling(new_doc)

            await _update_document_with_chunks(rag, created, [], new_doc)
            assert await rag.document_item_repository.get_item_count(created.id) == 1

    async def test_delete_document_cascades_items(self, temp_db_path):
        """Test that deleting a document also deletes its items."""
        from haiku.rag.store.models.document import Document

        docling_doc = _make_docling_doc()

        async with HaikuRAG(temp_db_path, create=True) as rag:
            document = Document(
                content="test content",
                uri="test://doc",
            )
            document.set_docling(docling_doc)
            created = await _store_document_with_chunks(rag, document, [], docling_doc)
            assert created.id is not None
            assert await rag.document_item_repository.get_item_count(created.id) == 6

            await rag.delete_document(created.id)
            assert await rag.document_item_repository.get_item_count(created.id) == 0


@pytest.mark.asyncio
class TestDocumentItemMigration:
    async def test_migration_populates_items_for_existing_documents(self, temp_db_path):
        """Test that the v0.40.0 migration populates items for pre-existing documents."""
        from haiku.rag.store.compression import compress_docling_split
        from haiku.rag.store.engine import DocumentRecord

        docling_doc = _make_docling_doc()
        json_str = docling_doc.model_dump_json()
        structure, pages = compress_docling_split(json_str)

        # Create a database at a pre-migration version with a document
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.39.0")
            doc_record = DocumentRecord(
                id="test-doc-1",
                content="test content",
                uri="test://doc",
                docling_document=structure,
                docling_pages=pages,
                docling_version=docling_doc.version,
            )
            await store.documents_table.add([doc_record])

            # Verify no items exist yet
            assert await store.document_items_table.count_rows() == 0

        # Re-open with skip_migration_check and run migration
        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()

            # Should have applied the v0.40.0 migration
            assert any("document_items" in desc for desc in applied)

            # Items should now exist
            item_count = await store.document_items_table.count_rows(
                filter="document_id = 'test-doc-1'"
            )
            assert item_count == 6

            # Verify item content
            items = await (
                store.document_items_table.query()
                .where("document_id = 'test-doc-1'")
                .to_list()
            )
            labels = {row["label"] for row in items}
            assert "section_header" in labels
            assert "paragraph" in labels
            assert "table" in labels

    async def test_migration_skips_documents_without_docling(self, temp_db_path):
        """Test that migration handles documents without docling data."""
        from haiku.rag.store.engine import DocumentRecord

        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.39.0")
            doc_record = DocumentRecord(
                id="no-docling",
                content="plain text document",
            )
            await store.documents_table.add([doc_record])

        async with Store(temp_db_path, skip_migration_check=True) as store:
            await store.migrate()

            # No items should have been created
            assert await store.document_items_table.count_rows() == 0


@pytest.mark.asyncio
class TestPictureDataStorage:
    async def test_create_and_get_picture_bytes(self, temp_db_path):
        """Round-trip picture bytes through DocumentItem and the repository."""
        async with HaikuRAG(temp_db_path, create=True) as rag:
            repo = DocumentItemRepository(rag.store)

            png_bytes = b"\x89PNG\r\n\x1a\nfake-picture-bytes"
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="paragraph",
                    text="Some text",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/pictures/0",
                    label="picture",
                    text="",
                    picture_data=png_bytes,
                ),
            ]
            await repo.create_items("doc-1", items)

            # Single-ref lookup
            got = await repo.get_picture_bytes("doc-1", "#/pictures/0")
            assert got == png_bytes
            # Missing ref returns None
            assert await repo.get_picture_bytes("doc-1", "#/pictures/999") is None
            # Non-picture row has no bytes
            assert await repo.get_picture_bytes("doc-1", "#/texts/0") is None

            # Batch lookup omits refs without bytes
            batch = await repo.get_pictures_for_chunk(
                "doc-1", ["#/pictures/0", "#/texts/0", "#/pictures/999"]
            )
            assert batch == {"#/pictures/0": png_bytes}

            # Empty refs returns empty dict
            assert await repo.get_pictures_for_chunk("doc-1", []) == {}

    async def test_hot_paths_exclude_picture_data(self, temp_db_path):
        """Light read paths must NOT pull picture_data into memory."""
        async with HaikuRAG(temp_db_path, create=True) as rag:
            repo = DocumentItemRepository(rag.store)

            heavy = b"x" * 1024
            await repo.create_items(
                "doc-1",
                [
                    DocumentItem(
                        document_id="doc-1",
                        position=0,
                        self_ref="#/pictures/0",
                        label="picture",
                        text="",
                        picture_data=heavy,
                    ),
                ],
            )

            for item in await repo.get_all_items("doc-1"):
                assert item.picture_data is None
            for item in await repo.get_items_in_range("doc-1", 0, 10):
                assert item.picture_data is None
            grouped = await repo.get_all_items_grouped(["doc-1"])
            for item in grouped["doc-1"]:
                assert item.picture_data is None

            # But the picture-byte accessors still work
            assert (await repo.get_picture_bytes("doc-1", "#/pictures/0")) == heavy

    async def test_fresh_db_has_picture_data_column(self, temp_db_path):
        """A newly-created DB has picture_data on document_items via _init_tables."""
        async with HaikuRAG(temp_db_path, create=True) as rag:
            schema = await rag.store.document_items_table.schema()
            assert "picture_data" in {f.name for f in schema}
def _docling_doc_with_picture():
    """Build a tiny DoclingDocument with one PictureItem carrying real PNG bytes
    via ImageRef.from_pil. Used by the picture-extraction tests."""
    from docling_core.types.doc.document import DoclingDocument, ImageRef
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import Image as PilImageModule

    img = PilImageModule.new("RGB", (8, 8), "red")
    doc = DoclingDocument(name="test-with-picture")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Hello world")
    doc.add_picture(image=ImageRef.from_pil(img, dpi=72))
    return doc


class TestExtractItemsPictureBytes:
    """A2b: extract_items decodes picture image bytes from data URIs."""

    def test_decodes_picture_bytes_from_live_doc(self):
        doc = _docling_doc_with_picture()
        items = extract_items("doc-1", doc)
        picture_items = [i for i in items if i.label == "picture"]
        assert len(picture_items) == 1
        data = picture_items[0].picture_data
        assert data is not None and len(data) > 0
        # PNG magic header — confirms we round-tripped real bytes, not a mangled URI.
        assert data.startswith(b"\x89PNG")

    def test_existing_picture_data_used_when_image_stripped(self):
        """Rebuild round-trip: live docling has image=None, snapshot fills the gap."""
        doc = _docling_doc_with_picture()
        for picture in doc.pictures:
            picture.image = None
        snapshot = {"#/pictures/0": b"snapshot-picture-bytes"}
        items = extract_items("doc-1", doc, existing_picture_data=snapshot)
        picture_items = [i for i in items if i.label == "picture"]
        assert picture_items[0].picture_data == b"snapshot-picture-bytes"


class TestExtractItemTextDescription:
    """A2b: extract_item_text returns VLM description text for PictureItems."""

    def test_returns_description_text_when_present(self):
        from docling_core.types.doc.document import (
            DescriptionAnnotation,
            DoclingDocument,
        )

        doc = DoclingDocument(name="t")
        doc.add_picture(
            annotations=[
                DescriptionAnnotation(text="A small red square", provenance="test")
            ]
        )
        items = extract_items("doc-1", doc)
        picture_items = [i for i in items if i.label == "picture"]
        assert picture_items[0].text == "A small red square"


class TestCompressDoclingSplitStripsPictureUris:
    """A2b: compress_docling_split removes inline picture URIs from the structure."""

    def test_picture_image_set_to_none_in_structure(self):
        import json

        from haiku.rag.store.compression import (
            compress_docling_split,
            decompress_json,
        )

        doc_json = {
            "schema_name": "DoclingDocument",
            "version": "1.10.0",
            "name": "test",
            "pictures": [
                {
                    "self_ref": "#/pictures/0",
                    "image": {
                        "mimetype": "image/png",
                        "uri": "data:image/png;base64,abc",
                    },
                },
                {"self_ref": "#/pictures/1", "image": None},
            ],
            "pages": {},
        }

        structure_bytes, pages_bytes = compress_docling_split(json.dumps(doc_json))
        decoded = json.loads(decompress_json(structure_bytes))

        for pic in decoded["pictures"]:
            assert pic["image"] is None
        assert pages_bytes is None  # no pages in this fixture


@pytest.mark.asyncio
class TestPictureDataMigrationBackfill:
    """A2b: v0.45.0 migration backfills picture_data and strips URIs from blobs."""

    async def test_backfill_populates_column_and_strips_blob(self, temp_db_path):
        import base64
        import json

        from haiku.rag.store.compression import compress_json, decompress_json
        from haiku.rag.store.engine import DocumentItemRecord, DocumentRecord

        fake_png = b"\x89PNG\r\n\x1a\nlegacy-picture-bytes-for-test"
        data_uri = "data:image/png;base64," + base64.b64encode(fake_png).decode("ascii")
        blob_data = {
            "schema_name": "DoclingDocument",
            "version": "1.10.0",
            "name": "legacy",
            "pictures": [
                {
                    "self_ref": "#/pictures/0",
                    "label": "picture",
                    "image": {"mimetype": "image/png", "uri": data_uri},
                },
            ],
        }
        blob_bytes = compress_json(json.dumps(blob_data))

        # Build a legacy-state DB at v0.44.0 *without* the picture_data column,
        # mirroring users coming from main's 0.44.0 release. The 0.45.0
        # migration must add the column AND backfill it from the blob in one
        # pass.
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.44.0")
            await store.documents_table.add(
                [
                    DocumentRecord(
                        id="legacy-doc",
                        content="legacy",
                        docling_document=blob_bytes,
                    )
                ]
            )
            # Items row exists (v0.40.0 would have placed it) but no picture_data yet.
            await store.document_items_table.add(
                [
                    DocumentItemRecord(
                        document_id="legacy-doc",
                        position=0,
                        self_ref="#/pictures/0",
                        label="picture",
                        text="",
                        page_numbers="[]",
                    )
                ]
            )
            # Drop the column so the migration's column-add path is exercised.
            await store.document_items_table.drop_columns(["picture_data"])
            schema_before = await store.document_items_table.schema()
            assert "picture_data" not in {f.name for f in schema_before}

        async with Store(temp_db_path, skip_migration_check=True) as store:
            applied = await store.migrate()
            assert any("picture" in d.lower() for d in applied)

            # Column was added by the migration
            schema_after = await store.document_items_table.schema()
            assert "picture_data" in {f.name for f in schema_after}

            # picture_data backfilled with the legacy bytes
            rows = await (
                store.document_items_table.query()
                .select(["self_ref", "picture_data"])
                .where("document_id = 'legacy-doc'")
                .to_list()
            )
            picture_rows = [r for r in rows if r["self_ref"] == "#/pictures/0"]
            assert len(picture_rows) == 1
            assert picture_rows[0]["picture_data"] == fake_png

            # docling_document blob has been re-compressed with image=None
            doc_rows = await (
                store.documents_table.query()
                .select(["docling_document"])
                .where("id = 'legacy-doc'")
                .to_list()
            )
            decoded = json.loads(decompress_json(doc_rows[0]["docling_document"]))
            assert decoded["pictures"][0]["image"] is None


@pytest.mark.asyncio
class TestPictureDataPreservedThroughRoundTrip:
    """A2b: snapshot/merge keeps picture bytes through update / rebuild cycles."""

    async def test_update_preserves_picture_data_when_blob_round_tripped(
        self, temp_db_path
    ):
        """update_document on a docling pulled from the (stripped) blob must
        not clobber picture_data — the snapshot/merge in
        _update_document_with_chunks handles it."""
        from haiku.rag.client.documents import (
            _store_document_with_chunks,
            _update_document_with_chunks,
        )
        from haiku.rag.store.models.document import Document

        docling_doc = _docling_doc_with_picture()

        async with HaikuRAG(temp_db_path, create=True) as rag:
            document = Document(content="Hello world", uri="test://doc")
            document.set_docling(docling_doc)
            created = await _store_document_with_chunks(rag, document, [], docling_doc)
            assert created.id is not None

            original = await rag.document_item_repository.get_all_picture_data(
                created.id
            )
            assert original.get("#/pictures/0") is not None

            # Re-load the docling from the stored blob — pictures now have image=None
            from_blob = created.get_docling_document()
            assert from_blob is not None
            assert all(p.image is None for p in from_blob.pictures)

            await _update_document_with_chunks(rag, created, [], from_blob)

            after = await rag.document_item_repository.get_all_picture_data(created.id)
            assert after.get("#/pictures/0") == original.get("#/pictures/0")

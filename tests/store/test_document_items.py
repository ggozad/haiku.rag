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
from tests.conftest import writing


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


def _make_docling_doc_with_levels():
    """DoclingDocument with explicit multi-level headings for hierarchy tests."""
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="leveled")
    doc.add_heading(text="Introduction", level=1)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Opening paragraph.")
    doc.add_heading(text="Background", level=2)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Background paragraph.")
    doc.add_heading(text="Prior Work", level=3)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Prior-work paragraph.")
    doc.add_heading(text="Methods", level=1)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Methods paragraph.")
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


class TestExtractItemsHierarchy:
    """heading_level and tree_depth populated from docling structure."""

    def test_heading_level_on_section_headers(self):
        doc = _make_docling_doc_with_levels()
        items = extract_items("doc-1", doc)
        headers = [i for i in items if i.label == "section_header"]
        assert [h.heading_level for h in headers] == [1, 2, 3, 1]

    def test_heading_level_zero_on_non_headers(self):
        doc = _make_docling_doc_with_levels()
        items = extract_items("doc-1", doc)
        non_headers = [i for i in items if i.label != "section_header"]
        assert non_headers
        assert all(i.heading_level == 0 for i in non_headers)

    def test_tree_depth_set_for_all_items(self):
        doc = _make_docling_doc_with_levels()
        items = extract_items("doc-1", doc)
        assert all(i.tree_depth > 0 for i in items)

    def test_plain_doc_has_zero_heading_level(self):
        from docling_core.types.doc.document import DoclingDocument
        from docling_core.types.doc.labels import DocItemLabel

        doc = DoclingDocument(name="plain")
        doc.add_text(label=DocItemLabel.PARAGRAPH, text="One.")
        doc.add_text(label=DocItemLabel.PARAGRAPH, text="Two.")
        items = extract_items("doc-1", doc)
        assert items
        assert all(i.heading_level == 0 for i in items)


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


def _doc_with_captioned_picture(*captions: str):
    from docling_core.types.doc.document import DoclingDocument, ImageRef
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import Image

    doc = DoclingDocument(name="pics")
    caption_items = [doc.add_text(label=DocItemLabel.CAPTION, text=c) for c in captions]
    img = ImageRef.from_pil(Image.new("RGB", (8, 8), "red"), dpi=72)
    doc.add_picture(image=img, caption=caption_items[0] if caption_items else None)
    pic = doc.pictures[0]
    for extra in caption_items[1:]:
        pic.captions.append(extra.get_ref())
    return doc, pic


def _doc_with_tables(n: int):
    from docling_core.types.doc.document import DoclingDocument, TableCell, TableData

    doc = DoclingDocument(name="tables")
    for _ in range(n):
        cells = [
            TableCell(
                text=f"r{r}c{c}",
                row_span=1,
                col_span=1,
                start_row_offset_idx=r,
                end_row_offset_idx=r + 1,
                start_col_offset_idx=c,
                end_col_offset_idx=c + 1,
            )
            for r in range(2)
            for c in range(2)
        ]
        doc.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=cells))
    return doc


class TestExtractItemTextPictures:
    """Description-less pictures derive text from captions, without a serializer."""

    def test_multiple_captions_space_joined(self):
        doc, pic = _doc_with_captioned_picture("First", "Second")
        assert extract_item_text(pic, doc) == "First Second"

    def test_description_wins_over_captions(self):
        from docling_core.types.doc.document import DescriptionMetaField, PictureMeta

        doc, pic = _doc_with_captioned_picture("A caption")
        pic.meta = PictureMeta(description=DescriptionMetaField(text="A red square."))
        assert extract_item_text(pic, doc) == "A red square."

    def test_picture_path_does_not_export_markdown(self, monkeypatch):
        from docling_core.types.doc.document import PictureItem

        doc, pic = _doc_with_captioned_picture("Only caption")

        def _boom(*args, **kwargs):
            raise AssertionError("picture path must not build a serializer")

        monkeypatch.setattr(PictureItem, "export_to_markdown", _boom)
        assert extract_item_text(pic, doc) == "Only caption"

    def test_picture_path_never_requests_serializer(self):
        doc, pic = _doc_with_captioned_picture("Only caption")

        def _explode():
            raise AssertionError("picture path must not request a serializer")

        assert extract_item_text(pic, doc, get_serializer=_explode) == "Only caption"


class TestExtractItemsTableSerializer:
    """Table text is unchanged; one serializer is reused across the whole pass."""

    def test_table_text_matches_export_to_markdown(self):
        doc = _doc_with_tables(1)
        expected = doc.tables[0].export_to_markdown(doc)
        items = extract_items("doc-1", doc)
        table_items = [i for i in items if i.label == "table"]
        assert len(table_items) == 1
        assert table_items[0].text == expected

    def test_one_serializer_built_for_multiple_tables(self, monkeypatch):
        import docling_core.transforms.serializer.markdown as md

        count = {"n": 0}
        base = md.MarkdownDocSerializer

        class Counting(base):
            def __init__(self, *args, **kwargs):
                count["n"] += 1
                super().__init__(*args, **kwargs)

        doc = _doc_with_tables(3)
        monkeypatch.setattr(md, "MarkdownDocSerializer", Counting)

        items = extract_items("doc-1", doc)
        assert sum(1 for i in items if i.label == "table") == 3
        assert count["n"] == 1

    def test_direct_table_call_builds_one_off_serializer(self):
        doc = _doc_with_tables(1)
        expected = doc.tables[0].export_to_markdown(doc)
        assert extract_item_text(doc.tables[0], doc) == expected


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

            result = (await repo.get_items_in_ranges({"doc-1": (3, 7)})).get(
                "doc-1", []
            )
            assert len(result) == 5
            assert result[0].position == 3
            assert result[-1].position == 7
            assert result[0].text == "Item 3"

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

    async def test_round_trip_preserves_heading_level_and_tree_depth(
        self, temp_db_path
    ):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            repo = DocumentItemRepository(rag.store)
            items = [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="section_header",
                    text="Intro",
                    heading_level=1,
                    tree_depth=1,
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="section_header",
                    text="Background",
                    heading_level=2,
                    tree_depth=2,
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/texts/2",
                    label="paragraph",
                    text="A paragraph.",
                    heading_level=0,
                    tree_depth=2,
                ),
            ]
            await repo.create_items("doc-1", items)

            got = await repo.get_all_items("doc-1")
            assert [(i.heading_level, i.tree_depth) for i in got] == [
                (1, 1),
                (2, 2),
                (0, 2),
            ]

            in_range = (await repo.get_items_in_ranges({"doc-1": (0, 2)})).get(
                "doc-1", []
            )
            assert [(i.heading_level, i.tree_depth) for i in in_range] == [
                (1, 1),
                (2, 2),
                (0, 2),
            ]

            assert [
                (i.heading_level, i.tree_depth)
                for i in await repo.get_all_items("doc-1")
            ] == [
                (1, 1),
                (2, 2),
                (0, 2),
            ]

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

            result = (await repo.get_items_in_ranges({"doc-1": (0, 9)})).get(
                "doc-1", []
            )
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
            created = await _store_document_with_chunks(
                writing(rag), document, [], docling_doc
            )
            assert created.id is not None

            count = await rag.document_item_repository.get_item_count(created.id)
            assert count == 6

            items = (
                await rag.document_item_repository.get_items_in_ranges(
                    {created.id: (0, count)}
                )
            ).get(created.id, [])
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
            created = await _store_document_with_chunks(
                writing(rag), document, [], docling_doc
            )
            assert created.id is not None
            assert await rag.document_item_repository.get_item_count(created.id) == 6

            # Update with a simpler document
            new_doc = DoclingDocument(name="updated")
            new_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Only one item now.")
            created.set_docling(new_doc)

            await _update_document_with_chunks(writing(rag), created, [], new_doc)
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
            created = await _store_document_with_chunks(
                writing(rag), document, [], docling_doc
            )
            assert created.id is not None
            assert await rag.document_item_repository.get_item_count(created.id) == 6

            await rag.delete_document(created.id)
            assert await rag.document_item_repository.get_item_count(created.id) == 0


@pytest.mark.asyncio
class TestDocumentItemMigration:
    async def test_migration_populates_items_for_existing_documents(self, temp_db_path):
        """Test that the v0.40.0 migration populates items for pre-existing documents."""
        from haiku.rag.store.compression import compress_docling_split
        from haiku.rag.store.schema import DocumentRecord
        from haiku.rag.store.upgrades.v0_40_0 import _apply_populate_document_items

        docling_doc = _make_docling_doc()
        structure, pages = compress_docling_split(docling_doc.model_dump(mode="json"))

        # Create a database at a pre-migration version with a document
        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.39.0")
            doc_record = DocumentRecord(
                id="test-doc-1",
                content="test content",
                docling_document=structure,
                docling_pages=pages,
                docling_version=docling_doc.version,
            )
            await store.documents_table.add([doc_record])

            # Verify no items exist yet
            assert await store.document_items_table.count_rows() == 0

        # Re-open and apply the v0.40.0 migration in isolation (the full chain
        # would also run later migrations that touch documents.metadata, absent
        # from this docling-only fixture).
        async with Store(temp_db_path, skip_migration_check=True) as store:
            await _apply_populate_document_items(store)

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
        from haiku.rag.store.schema import DocumentRecord
        from haiku.rag.store.upgrades.v0_40_0 import _apply_populate_document_items

        async with Store(temp_db_path, create=True, skip_migration_check=True) as store:
            await store.set_haiku_version("0.39.0")
            doc_record = DocumentRecord(
                id="no-docling",
                content="plain text document",
            )
            await store.documents_table.add([doc_record])

        async with Store(temp_db_path, skip_migration_check=True) as store:
            await _apply_populate_document_items(store)

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
            for item in (await repo.get_items_in_ranges({"doc-1": (0, 10)})).get(
                "doc-1", []
            ):
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
    """extract_items decodes picture image bytes from data URIs."""

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

    def test_live_image_uri_wins_over_existing_picture_data(self):
        """When both an inline URI and a snapshot are available, the URI wins."""
        doc = _docling_doc_with_picture()
        snapshot = {"#/pictures/0": b"snapshot-bytes-should-not-be-used"}
        items = extract_items("doc-1", doc, existing_picture_data=snapshot)
        picture_items = [i for i in items if i.label == "picture"]
        data = picture_items[0].picture_data
        assert data is not None
        assert data.startswith(b"\x89PNG")
        assert data != b"snapshot-bytes-should-not-be-used"


class TestExtractItemTextDescription:
    """extract_item_text returns VLM description text for PictureItems."""

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
    """compress_docling_split removes inline picture URIs from the structure."""

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

        structure_bytes, pages_bytes = compress_docling_split(doc_json)
        decoded = json.loads(decompress_json(structure_bytes))

        for pic in decoded["pictures"]:
            assert pic["image"] is None
        assert pages_bytes is None  # no pages in this fixture


@pytest.mark.asyncio
class TestPictureDataMigrationBackfill:
    """0.45.0 migration backfills picture_data and strips URIs from blobs."""

    async def test_backfill_populates_column_and_strips_blob(self, temp_db_path):
        import base64
        import json

        from haiku.rag.store.compression import compress_json, decompress_json
        from haiku.rag.store.schema import DocumentItemRecord, DocumentRecord
        from haiku.rag.store.upgrades.v0_45_0 import _apply_extract_picture_bytes

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
            # Apply the v0.45.0 migration in isolation (the full chain would also
            # run later migrations that touch documents.metadata, absent here).
            await _apply_extract_picture_bytes(store)

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
    """Snapshot/merge keeps picture bytes through update / rebuild cycles."""

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
            created = await _store_document_with_chunks(
                writing(rag), document, [], docling_doc
            )
            assert created.id is not None

            original = await rag.document_item_repository.get_all_picture_data(
                created.id
            )
            assert original.get("#/pictures/0") is not None

            # Re-load the docling from the stored blob — pictures now have image=None
            from_blob = created.get_docling_document()
            assert from_blob is not None
            assert all(p.image is None for p in from_blob.pictures)

            await _update_document_with_chunks(writing(rag), created, [], from_blob)

            after = await rag.document_item_repository.get_all_picture_data(created.id)
            assert after.get("#/pictures/0") == original.get("#/pictures/0")


@pytest.mark.asyncio
async def test_replace_for_document_with_no_items_deletes_existing(temp_db_path):
    """Replacing with an empty list clears the document's items."""
    from haiku.rag.store.engine import Store
    from haiku.rag.store.repositories.document_item import DocumentItemRepository

    async with Store(temp_db_path, create=True) as store:
        repo = DocumentItemRepository(store)
        docling_doc = _make_docling_doc()
        await repo.replace_for_document("doc-1", extract_items("doc-1", docling_doc))
        assert await repo.get_all_items("doc-1")

        await repo.replace_for_document("doc-1", [])

        assert await repo.get_all_items("doc-1") == []


def _doc_with_inline_group_list_item(
    second_child_group: bool = False, group_before_inline: bool = False
):
    """A ListItem whose content docling pushed into a child InlineGroup
    (e.g. a list item mixing plain text with a code span). A heading gets
    the same shape. A body-level paragraph does not: its InlineGroup hangs
    directly off `#/body` with no owner item to fold the text back into.
    Optionally add a sibling nested ListGroup, matching a list item that
    also has sub-items; `group_before_inline` adds it before the InlineGroup
    instead of after, so a lookup by child position alone would miss it.
    """
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel, GroupLabel

    doc = DoclingDocument(name="inline")
    list_group = doc.add_group(label=GroupLabel.LIST)
    list_item = doc.add_list_item(text="", parent=list_group)
    if second_child_group and group_before_inline:
        nested = doc.add_group(label=GroupLabel.LIST, parent=list_item)
        doc.add_list_item(text="Sub-item.", parent=nested)
    inline_group = doc.add_group(label=GroupLabel.INLINE, parent=list_item)
    doc.add_text(label=DocItemLabel.TEXT, text="Run snake_case & ", parent=inline_group)
    doc.add_code(text="pytest", parent=inline_group)
    doc.add_text(label=DocItemLabel.TEXT, text=" to test.", parent=inline_group)
    if second_child_group and not group_before_inline:
        nested = doc.add_group(label=GroupLabel.LIST, parent=list_item)
        doc.add_list_item(text="Sub-item.", parent=nested)
    return doc, list_item


class TestExtractItemTextInlineGroup:
    """A container item's own .text is empty when docling pushes mixed
    inline content into a child InlineGroup; extract_item_text must recover
    it from that group instead of leaving the item's row blank.
    """

    def test_list_item_recovers_text_from_inline_group(self):
        doc, list_item = _doc_with_inline_group_list_item()

        assert list_item.text == ""
        text = extract_item_text(list_item, doc)
        assert text is not None
        assert "Run" in text
        assert "pytest" in text
        assert "to test." in text

    def test_recovered_text_is_not_markdown_escaped(self):
        """The fallback serializer must not escape underscores or HTML
        entities: the recovered text is stored for search, not rendered
        as markdown, and an escaped `snake\\_case` or `&amp;` would not
        match the original run.
        """
        doc, list_item = _doc_with_inline_group_list_item()

        text = extract_item_text(list_item, doc)
        assert text is not None
        assert "snake_case" in text
        assert "\\_" not in text
        assert " & " in text
        assert "&amp;" not in text

    def test_extract_items_stores_non_empty_text_for_inline_group_item(self):
        doc, _ = _doc_with_inline_group_list_item()
        items = extract_items("doc-1", doc)

        list_items = [i for i in items if i.label == "list_item"]
        assert len(list_items) == 1
        assert list_items[0].text != ""

    def test_nested_list_group_not_folded_into_parent_text(self):
        """A second child (a nested ListGroup of sub-items) must not be
        pulled into the parent's serialized text: those sub-items are
        walked and stored as their own rows by extract_items already, so
        including them here would duplicate their content.
        """
        doc, list_item = _doc_with_inline_group_list_item(second_child_group=True)

        text = extract_item_text(list_item, doc)
        assert text is not None
        assert "Sub-item" not in text

    def test_inline_group_recovered_regardless_of_child_position(self):
        """A list item can carry a nested ListGroup of sub-items ordered
        before its InlineGroup, not only after: the lookup must find the
        InlineGroup among all children, not assume it sits at position 0.
        """
        doc, list_item = _doc_with_inline_group_list_item(
            second_child_group=True, group_before_inline=True
        )

        text = extract_item_text(list_item, doc)
        assert text is not None
        assert "pytest" in text
        assert "Sub-item" not in text

    def test_reuses_passed_serializer(self, monkeypatch):
        import docling_core.transforms.serializer.markdown as md

        count = {"n": 0}
        base = md.MarkdownDocSerializer

        class Counting(base):
            def __init__(self, *args, **kwargs):
                count["n"] += 1
                super().__init__(*args, **kwargs)

        doc, _ = _doc_with_inline_group_list_item()
        monkeypatch.setattr(md, "MarkdownDocSerializer", Counting)

        items = extract_items("doc-1", doc)
        assert any(i.label == "list_item" and i.text for i in items)
        assert count["n"] == 1

    def test_returns_none_when_serialization_fails(self):
        """A serializer that raises leaves the item with no extractable text
        rather than aborting the extraction pass, matching the table path.
        """
        doc, list_item = _doc_with_inline_group_list_item()

        class _Boom:
            def serialize(self, item):
                raise RuntimeError("serializer exploded")

        assert extract_item_text(list_item, doc, get_serializer=_Boom) is None

    def test_leaf_item_without_children_still_returns_none(self):
        """A genuinely empty item (no text, no children) is unaffected:
        this is not the InlineGroup case and must not be treated as one.
        """
        from docling_core.types.doc.document import DoclingDocument
        from docling_core.types.doc.labels import DocItemLabel

        doc = DoclingDocument(name="empty-leaf")
        item = doc.add_text(label=DocItemLabel.PARAGRAPH, text="")
        assert item.children == []
        assert extract_item_text(item, doc) is None

    async def test_import_document_recovers_inline_group_text_without_moving_refs(
        self, temp_db_path
    ):
        """import_document stores the caller's DoclingDocument and chunks as
        given, never converting or flattening it, so a chunk's doc_item_refs
        index into that document directly. This is the path #621's
        conversion-time flatten does not reach, and the one the recovered
        text needs to survive without renumbering anything.
        """
        from haiku.rag.config import get_config
        from haiku.rag.store.models.chunk import Chunk

        doc, list_item = _doc_with_inline_group_list_item()
        dim = get_config().embeddings.model.vector_dim
        chunks = [
            Chunk(
                content="Run `pytest` to test.",
                metadata={"doc_item_refs": [list_item.self_ref]},
                order=0,
                embedding=[0.1] * dim,
            )
        ]

        async with HaikuRAG(temp_db_path, create=True) as client:
            imported = await client.import_document(
                doc, chunks, uri="mem://import-inline"
            )

            items = await client.document_item_repository.get_all_items(imported.id)
            by_ref = {item.self_ref: item for item in items}

            recovered = by_ref[list_item.self_ref]
            assert "Run" in recovered.text
            assert "pytest" in recovered.text
            assert "to test." in recovered.text

            # The group's children are stored as their own rows beside the
            # recovered item, since this path stores the document as given,
            # so the recovered item and its fragments both land in expanded
            # context.
            fragment_texts = {
                item.text for ref, item in by_ref.items() if ref != list_item.self_ref
            }
            assert {"Run snake_case & ", "pytest", " to test."} <= fragment_texts

            stored_chunk = (
                await client.chunk_repository.get_by_document_id(imported.id)
            )[0]
            assert stored_chunk.metadata["doc_item_refs"] == [list_item.self_ref]


class TestExtractItemTextFallbacks:
    def test_table_returns_none_when_serialization_fails(self):
        """A serializer that raises leaves the table with no extractable text
        rather than aborting the extraction pass."""
        doc = _doc_with_tables(1)

        class _Boom:
            def serialize(self, item):
                raise RuntimeError("serializer exploded")

        assert extract_item_text(doc.tables[0], doc, get_serializer=_Boom) is None

    def test_picture_bytes_are_recompressed(self):
        """A picture stores fewer bytes than docling handed us, same pixels."""
        import base64
        from io import BytesIO

        import cv2
        import numpy as np
        from docling_core.types.doc.document import ImageRef
        from PIL import Image

        from haiku.rag.store.models.document_item import _decode_picture_bytes

        rng = np.random.default_rng(0)
        pixels = np.full((240, 200, 3), 255, dtype=np.uint8)
        for y in range(10, 230, 14):
            pixels[y : y + 6, 10 : 10 + int(rng.integers(60, 170))] = 30
        ok, buffer = cv2.imencode(".png", pixels)
        assert ok
        raw = buffer.tobytes()

        doc, pic = _doc_with_captioned_picture("caption")
        pic.image = ImageRef.model_validate(
            {
                "mimetype": "image/png",
                "dpi": 72,
                "size": {"width": 200, "height": 240},
                "uri": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
            }
        )

        stored = _decode_picture_bytes(pic)

        assert stored is not None
        assert len(stored) < len(raw)
        with Image.open(BytesIO(raw)) as a, Image.open(BytesIO(stored)) as b:
            assert a.convert("RGB").tobytes() == b.convert("RGB").tobytes()

    def test_file_backed_picture_has_no_inline_bytes(self):
        """A picture whose ImageRef points at a file rather than a data: URI
        carries nothing to decode."""
        from docling_core.types.doc.document import ImageRef

        from haiku.rag.store.models.document_item import _decode_picture_bytes

        doc, pic = _doc_with_captioned_picture("caption")
        pic.image = ImageRef.model_validate(
            {
                "mimetype": "image/png",
                "dpi": 72,
                "size": {"width": 1, "height": 1},
                "uri": "file:///tmp/picture.png",
            }
        )

        assert _decode_picture_bytes(pic) is None

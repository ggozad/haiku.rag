import pytest

from haiku.rag.client import HaikuRAG
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
            created = await rag._store_document_with_chunks(document, [], docling_doc)
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
            created = await rag._store_document_with_chunks(document, [], docling_doc)
            assert created.id is not None
            assert await rag.document_item_repository.get_item_count(created.id) == 6

            # Update with a simpler document
            new_doc = DoclingDocument(name="updated")
            new_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Only one item now.")
            created.set_docling(new_doc)

            await rag._update_document_with_chunks(created, [], new_doc)
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
            created = await rag._store_document_with_chunks(document, [], docling_doc)
            assert created.id is not None
            assert await rag.document_item_repository.get_item_count(created.id) == 6

            await rag.delete_document(created.id)
            assert await rag.document_item_repository.get_item_count(created.id) == 0


class TestDocumentItemMigration:
    def test_migration_populates_items_for_existing_documents(self, temp_db_path):
        """Test that the v0.40.0 migration populates items for pre-existing documents."""
        from haiku.rag.store.compression import compress_docling_split
        from haiku.rag.store.engine import DocumentRecord

        docling_doc = _make_docling_doc()
        json_str = docling_doc.model_dump_json()
        structure, pages = compress_docling_split(json_str)

        # Create a database at a pre-migration version with a document
        store = Store(temp_db_path, create=True, skip_migration_check=True)
        store.set_haiku_version("0.39.0")
        doc_record = DocumentRecord(
            id="test-doc-1",
            content="test content",
            uri="test://doc",
            docling_document=structure,
            docling_pages=pages,
            docling_version=docling_doc.version,
        )
        store.documents_table.add([doc_record])

        # Verify no items exist yet
        assert store.document_items_table.count_rows() == 0
        store.close()

        # Re-open with skip_migration_check and run migration
        store = Store(temp_db_path, skip_migration_check=True)
        applied = store.migrate()

        # Should have applied the v0.40.0 migration
        assert any("document_items" in desc for desc in applied)

        # Items should now exist
        item_count = store.document_items_table.count_rows(
            filter="document_id = 'test-doc-1'"
        )
        assert item_count == 6

        # Verify item content
        items = (
            store.document_items_table.search()
            .where("document_id = 'test-doc-1'")
            .to_list()
        )
        labels = {row["label"] for row in items}
        assert "section_header" in labels
        assert "paragraph" in labels
        assert "table" in labels

        store.close()

    def test_migration_skips_documents_without_docling(self, temp_db_path):
        """Test that migration handles documents without docling data."""
        from haiku.rag.store.engine import DocumentRecord

        store = Store(temp_db_path, create=True, skip_migration_check=True)
        store.set_haiku_version("0.39.0")
        doc_record = DocumentRecord(
            id="no-docling",
            content="plain text document",
        )
        store.documents_table.add([doc_record])
        store.close()

        store = Store(temp_db_path, skip_migration_check=True)
        store.migrate()

        # No items should have been created
        assert store.document_items_table.count_rows() == 0
        store.close()

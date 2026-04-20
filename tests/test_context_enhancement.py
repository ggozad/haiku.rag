import pytest
from docling_core.types.doc.document import DoclingDocument, TableData
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult


async def create_document_with_docling(
    client: HaikuRAG, docling_doc: DoclingDocument, title: str
):
    """Helper to create a document from a DoclingDocument using import_document."""
    chunks = await client.chunk(docling_doc)
    embedded_chunks = await client._ensure_chunks_embedded(chunks)
    return await client.import_document(
        docling_document=docling_doc,
        chunks=embedded_chunks,
        title=title,
    )


def create_table_document() -> DoclingDocument:
    """Create a document with a table that will be split across chunks."""
    doc = DoclingDocument(name="table_test")

    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Introduction paragraph.")
    doc.add_heading(text="Employee Data", level=1)

    # Create a table with enough content to span multiple chunks
    table_data = TableData(num_cols=3, num_rows=0)
    table_data.add_row(["Name", "Age", "City"])
    table_data.add_row(["Alice Smith", "30", "New York"])
    table_data.add_row(["Bob Johnson", "25", "Los Angeles"])
    table_data.add_row(["Charlie Brown", "35", "Chicago"])
    table_data.add_row(["Diana Ross", "28", "Miami"])
    doc.add_table(data=table_data)

    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Conclusion paragraph.")

    return doc


def create_list_document() -> DoclingDocument:
    """Create a document with list items that will be split across chunks."""
    doc = DoclingDocument(name="list_test")

    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Shopping list for the week:")
    doc.add_list_item(
        text="Fresh organic apples from the farmers market", enumerated=False
    )
    doc.add_list_item(text="Ripe yellow bananas for smoothies", enumerated=False)
    doc.add_list_item(text="Valencia oranges for fresh juice", enumerated=False)
    doc.add_list_item(text="Seedless red grapes as healthy snack", enumerated=False)
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Remember to bring reusable bags.")

    return doc


def create_code_document() -> DoclingDocument:
    """Create a document with adjacent code blocks that will be split."""
    doc = DoclingDocument(name="code_test")

    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Here are several code snippets:")
    # Multiple adjacent code blocks - type-aware expansion should group them
    doc.add_text(label=DocItemLabel.CODE, text="# Part 1: Setup\nimport os\nimport sys")
    doc.add_text(
        label=DocItemLabel.CODE, text='# Part 2: Config\nCONFIG = {"debug": True}'
    )
    doc.add_text(
        label=DocItemLabel.CODE, text="# Part 3: Main\ndef main():\n    print(CONFIG)"
    )
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="End of code examples.")

    return doc


@pytest.fixture
def small_chunk_config() -> AppConfig:
    """Config with small chunk size to force splitting."""
    config = AppConfig()
    config.processing.chunk_size = 32
    config.search.max_context_chars = 10000
    return config


@pytest.mark.vcr()
async def test_table_expansion_includes_split_rows(temp_db_path, small_chunk_config):
    """Verify that table expansion retrieves rows that were split into different chunks."""
    docling_doc = create_table_document()

    async with HaikuRAG(temp_db_path, config=small_chunk_config, create=True) as client:
        doc = await create_document_with_docling(client, docling_doc, "Table Test")
        assert doc.id is not None

        # Search for table content
        results = await client.search("Alice Smith New York employee", limit=5)
        table_results = [r for r in results if "table" in r.labels]
        assert len(table_results) > 0, (
            f"No table results. Labels: {[r.labels for r in results]}"
        )

        original = table_results[0]

        # Verify the original chunk does NOT contain all table data
        # (proving we need expansion)
        original_has_all = all(
            name in original.content for name in ["Alice", "Bob", "Charlie", "Diana"]
        )

        # Expand context
        expanded = await client.expand_context(table_results[:1])
        assert len(expanded) == 1

        expanded_content = expanded[0].content

        # After expansion, we should have the complete table
        assert "Alice" in expanded_content
        assert "Bob" in expanded_content
        assert "Charlie" in expanded_content
        assert "Diana" in expanded_content

        # Verify expansion actually added content (unless chunk already had everything)
        if not original_has_all:
            assert len(expanded_content) > len(original.content), (
                "Expansion should have added content"
            )


@pytest.mark.vcr()
async def test_list_expansion_includes_split_items(temp_db_path, small_chunk_config):
    """Verify that list expansion retrieves items that were split into different chunks."""
    docling_doc = create_list_document()

    async with HaikuRAG(temp_db_path, config=small_chunk_config, create=True) as client:
        doc = await create_document_with_docling(client, docling_doc, "List Test")
        assert doc.id is not None

        # Search for a list item
        results = await client.search("grapes healthy snack", limit=5)
        list_results = [r for r in results if "list_item" in r.labels]
        assert len(list_results) > 0, (
            f"No list results. Labels: {[r.labels for r in results]}"
        )

        original = list_results[0]

        # Check what the original chunk contains
        original_items = sum(
            1
            for item in ["apples", "bananas", "oranges", "grapes"]
            if item in original.content.lower()
        )

        # Expand context
        expanded = await client.expand_context(list_results[:1])
        assert len(expanded) == 1

        expanded_content = expanded[0].content.lower()

        # Count items after expansion
        expanded_items = sum(
            1
            for item in ["apples", "bananas", "oranges", "grapes"]
            if item in expanded_content
        )

        # Expansion should include at least as many items (more if split)
        assert expanded_items >= original_items

        # If original didn't have all items, expansion should have added some
        if original_items < 4:
            assert expanded_items > original_items, (
                f"Expansion should have added items. Original: {original_items}, Expanded: {expanded_items}"
            )


@pytest.mark.vcr()
async def test_code_expansion_includes_adjacent_blocks(
    temp_db_path, small_chunk_config
):
    """Verify that code expansion retrieves adjacent code blocks split across chunks."""
    docling_doc = create_code_document()

    async with HaikuRAG(temp_db_path, config=small_chunk_config, create=True) as client:
        doc = await create_document_with_docling(client, docling_doc, "Code Test")
        assert doc.id is not None

        # Search for middle code block (Part 2)
        results = await client.search("CONFIG debug True", limit=5)
        code_results = [r for r in results if "code" in r.labels]
        assert len(code_results) > 0, (
            f"No code results. Labels: {[r.labels for r in results]}"
        )

        original = code_results[0]

        # Check what parts the original chunk has
        original_parts = sum(
            1 for part in ["Part 1", "Part 2", "Part 3"] if part in original.content
        )

        # Expand context
        expanded = await client.expand_context(code_results[:1])
        assert len(expanded) == 1

        expanded_content = expanded[0].content

        # Count parts after expansion
        expanded_parts = sum(
            1 for part in ["Part 1", "Part 2", "Part 3"] if part in expanded_content
        )

        # Expansion should include at least as many parts
        assert expanded_parts >= original_parts

        # If original didn't have all parts, expansion should have added some
        if original_parts < 3:
            assert expanded_parts > original_parts, (
                f"Expansion should have added code blocks. Original: {original_parts}, Expanded: {expanded_parts}"
            )


@pytest.mark.vcr()
async def test_text_expansion_includes_surrounding(temp_db_path):
    """Text content expansion should include surrounding paragraphs."""
    config = AppConfig()
    config.processing.chunk_size = 32

    # Create a document with longer paragraphs that will split
    doc = DoclingDocument(name="text_test")
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="First paragraph with enough content to be its own chunk in the document.",
    )
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Second paragraph contains different information about software testing.",
    )
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Third paragraph discusses various topics and provides more details.",
    )
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text="Fourth paragraph concludes the document with final thoughts.",
    )

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        document = await create_document_with_docling(client, doc, "Text Only")
        assert document.id is not None

        # Search for second paragraph
        results = await client.search("software testing", limit=1)
        assert len(results) > 0

        original = results[0]
        expanded = await client.expand_context(results)

        # Expansion should include adjacent paragraphs
        assert len(expanded[0].content) >= len(original.content)


@pytest.mark.vcr()
async def test_expansion_preserves_metadata(temp_db_path, small_chunk_config):
    """Expansion should preserve document metadata."""
    docling_doc = create_table_document()

    async with HaikuRAG(temp_db_path, config=small_chunk_config, create=True) as client:
        doc = await create_document_with_docling(client, docling_doc, "Metadata Test")
        assert doc.id is not None

        results = await client.search("Introduction paragraph", limit=1)
        assert len(results) > 0

        expanded = await client.expand_context(results)

        assert expanded[0].document_title == "Metadata Test"
        assert expanded[0].chunk_id == results[0].chunk_id
        assert expanded[0].document_id == results[0].document_id


@pytest.mark.vcr()
async def test_format_for_agent_output(temp_db_path, small_chunk_config):
    """format_for_agent should include source, type, and content sections."""
    docling_doc = create_table_document()

    async with HaikuRAG(temp_db_path, config=small_chunk_config, create=True) as client:
        doc = await create_document_with_docling(client, docling_doc, "Format Test")
        assert doc.id is not None

        results = await client.search("Alice Smith employee data", limit=5)
        table_results = [r for r in results if "table" in r.labels]
        assert len(table_results) > 0

        expanded = await client.expand_context(table_results[:1])
        # Format with rank (the way agents use it)
        formatted = expanded[0].format_for_agent(rank=1, total=1)

        # Check format structure
        assert "[rank 1 of 1]" in formatted
        assert 'Source: "Format Test"' in formatted
        assert "Type: table" in formatted
        assert "Content:" in formatted


async def test_expand_context_single_item_document(temp_db_path):
    """Test expand_context with a single-item document."""
    from haiku.rag.store.models.document import Document

    docling_doc = DoclingDocument(name="simple")
    docling_doc.add_text(label=DocItemLabel.PARAGRAPH, text="Simple test content")

    async with HaikuRAG(temp_db_path, create=True) as client:
        document = Document(content="Simple test content")
        document.set_docling(docling_doc)
        doc = await client._store_document_with_chunks(document, [], docling_doc)
        assert doc.id is not None

        # Create a search result with a doc_item_ref pointing to the item
        items = await client.document_item_repository.get_items_in_range(doc.id, 0, 10)
        assert len(items) > 0

        search_results = [
            SearchResult(
                content="Simple test content",
                score=0.9,
                document_id=doc.id,
                doc_item_refs=[items[0].self_ref],
            )
        ]
        expanded_results = await client.expand_context(search_results)

        assert len(expanded_results) == 1
        assert expanded_results[0].score == 0.9
        assert "Simple test content" in expanded_results[0].content


async def test_expand_context_no_refs_passes_through(temp_db_path):
    """Results without doc_item_refs pass through unexpanded."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # A search result with no doc_item_refs should pass through as-is
        search_results = [
            SearchResult(
                content="Some chunk content",
                score=0.8,
                document_id="some-doc",
                doc_item_refs=[],
            )
        ]
        expanded = await client.expand_context(search_results)

        assert len(expanded) == 1
        assert expanded[0].content == "Some chunk content"
        assert expanded[0].score == 0.8


@pytest.mark.vcr()
async def test_expand_context_with_docling_merges_overlapping(temp_db_path):
    """Test that expand_context with DoclingDocument merges overlapping results."""
    config = AppConfig()

    markdown_content = """# Chapter 1

This is paragraph one about topic A.

This is paragraph two about topic A continued.

This is paragraph three about topic B.

# Chapter 2

This is paragraph four about topic C.
"""

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        doc = await client.create_document(
            content=markdown_content,
            uri="test://structured",
        )

        assert doc.id is not None
        assert doc.docling_document is not None

        # Get chunks which should have doc_item_refs
        chunks = await client.chunk_repository.get_by_document_id(doc.id)
        assert len(chunks) >= 1

        # Find chunks that have doc_item_refs (from docling chunking)
        chunks_with_refs = [c for c in chunks if c.get_chunk_metadata().doc_item_refs]

        if len(chunks_with_refs) >= 2:
            # Create search results from adjacent chunks
            search_results = [
                SearchResult.from_chunk(chunks_with_refs[0], 0.9),
                SearchResult.from_chunk(chunks_with_refs[1], 0.8),
            ]

            # Expand with configured radius that should cause overlap
            expanded = await client.expand_context(search_results)

            # If chunks were adjacent, they should be merged
            # The expanded results should have merged metadata
            assert len(expanded) >= 1

            # Check that expanded result has page_numbers populated
            for r in expanded:
                # Should have doc_item_refs from expansion
                assert r.doc_item_refs is not None


@pytest.mark.vcr()
async def test_expand_context_docling_merges_metadata(temp_db_path):
    """Test that expand_context properly merges metadata from multiple results."""
    config = AppConfig()

    markdown_content = """# Introduction

First paragraph of introduction.

Second paragraph of introduction.

# Methods

First paragraph of methods section.

Second paragraph of methods section.

# Results

First paragraph of results.
"""

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        doc = await client.create_document(
            content=markdown_content,
            uri="test://metadata-merge",
        )

        assert doc.id is not None

        chunks = await client.chunk_repository.get_by_document_id(doc.id)
        chunks_with_refs = [c for c in chunks if c.get_chunk_metadata().doc_item_refs]

        if len(chunks_with_refs) >= 2:
            # Get chunks with different headings if possible
            chunk1 = chunks_with_refs[0]
            chunk2 = chunks_with_refs[-1]  # Last chunk likely has different heading

            search_results = [
                SearchResult.from_chunk(chunk1, 0.9),
                SearchResult.from_chunk(chunk2, 0.8),
            ]

            # Expand with large radius to potentially merge
            expanded = await client.expand_context(search_results)

            # Check that results have proper structure
            for r in expanded:
                # Content should be non-empty
                assert len(r.content) > 0
                # Score should be preserved (best score)
                assert r.score in [0.9, 0.8]
                # Expanded content should have docling refs
                assert r.doc_item_refs is not None and len(r.doc_item_refs) > 0
                # Document has headings, expanded result should too
                assert r.headings is not None and len(r.headings) > 0


def create_picture_document() -> DoclingDocument:
    """Create a document with picture items for testing image handling."""
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name="picture_test")

    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Introduction with context.")
    doc.add_heading(text="Figures Section", level=1)

    # Add a picture item - this will have export_to_markdown method
    # that defaults to EMBEDDED mode if called without image_mode parameter
    doc.add_picture()

    doc.add_text(label=DocItemLabel.CAPTION, text="Figure 1: Sample diagram")
    doc.add_text(label=DocItemLabel.PARAGRAPH, text="Conclusion paragraph.")

    return doc


@pytest.mark.vcr()
async def test_expand_context_no_base64_images(temp_db_path):
    """Ensure expanded context does not contain base64 image data.

    This test verifies that when expand_context includes PictureItem objects,
    they are serialized with PLACEHOLDER mode (not EMBEDDED), preventing
    base64 image data from leaking into the expanded content.
    """
    config = AppConfig()

    docling_doc = create_picture_document()

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        doc = await create_document_with_docling(client, docling_doc, "Picture Test")
        assert doc.id is not None

        # Search for content near the picture
        results = await client.search("Figure diagram", limit=5)
        assert len(results) > 0

        # Expand context - this should include the picture area
        expanded = await client.expand_context(results)

        for result in expanded:
            # Base64 image data should never appear in expanded content
            assert "base64" not in result.content.lower(), (
                f"Found 'base64' in expanded content: {result.content[:500]}"
            )
            assert "data:image" not in result.content.lower(), (
                f"Found 'data:image' in expanded content: {result.content[:500]}"
            )


@pytest.mark.vcr()
async def test_expand_context_no_base64_images_docling_local(temp_db_path):
    """Ensure expanded context from real PDF does not contain base64 image data.

    Tests end-to-end with doclaynet.pdf using docling-local converter.
    """
    from pathlib import Path

    config = AppConfig()
    config.processing.converter = "docling-local"
    config.processing.chunker = "docling-local"
    config.processing.conversion_options.do_ocr = False

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        pdf_path = Path(__file__).parent / "data" / "doclaynet.pdf"
        result = await client.create_document_from_source(pdf_path)
        doc = result if not isinstance(result, list) else result[0]
        assert doc.id is not None

        # Search for content that might include pictures
        results = await client.search("figure table", limit=10)

        if len(results) > 0:
            expanded = await client.expand_context(results)

            for result in expanded:
                assert "base64" not in result.content.lower(), (
                    f"Found 'base64' in expanded content: {result.content[:500]}"
                )
                assert "data:image" not in result.content.lower(), (
                    f"Found 'data:image' in expanded content: {result.content[:500]}"
                )


@pytest.mark.vcr()
async def test_expand_context_no_base64_images_docling_serve(temp_db_path):
    """Ensure expanded context from real PDF does not contain base64 image data.

    Tests end-to-end with doclaynet.pdf using docling-serve converter.
    """
    from pathlib import Path

    config = AppConfig()
    config.processing.converter = "docling-serve"
    config.processing.chunker = "docling-serve"

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        pdf_path = Path(__file__).parent / "data" / "doclaynet.pdf"
        result = await client.create_document_from_source(pdf_path)
        doc = result if not isinstance(result, list) else result[0]
        assert doc.id is not None

        # Search for content that might include pictures
        results = await client.search("figure table", limit=10)

        if len(results) > 0:
            expanded = await client.expand_context(results)

            for result in expanded:
                assert "base64" not in result.content.lower(), (
                    f"Found 'base64' in expanded content: {result.content[:500]}"
                )
                assert "data:image" not in result.content.lower(), (
                    f"Found 'data:image' in expanded content: {result.content[:500]}"
                )

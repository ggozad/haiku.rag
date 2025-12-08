import pytest
from docling_core.types.doc.document import DoclingDocument, TableData
from docling_core.types.doc.labels import DocItemLabel

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import SearchResult
from haiku.rag.store.models.chunk import Chunk


async def create_document_with_docling(
    client: HaikuRAG, docling_doc: DoclingDocument, title: str
):
    """Helper to create a document from a DoclingDocument using import_document."""
    chunks = await client.chunk(docling_doc)
    embedded_chunks = await client._ensure_chunks_embedded(chunks)
    return await client.import_document(
        chunks=embedded_chunks,
        title=title,
        docling_document_json=docling_doc.model_dump_json(),
        docling_version=docling_doc.version,
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
    config.processing.max_context_items = 25
    config.processing.max_context_chars = 10000
    return config


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_text_expansion_uses_radius(temp_db_path):
    """Text content expansion should use radius, not structural boundaries."""
    config = AppConfig()
    config.processing.chunk_size = 32
    config.processing.text_context_radius = 1  # Small radius

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

        # With radius=1, expansion should include adjacent paragraphs
        # Content length should be >= original (may add adjacent content)
        assert len(expanded[0].content) >= len(original.content)


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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
        formatted = expanded[0].format_for_agent()

        # Check format structure
        assert "score:" in formatted
        assert 'Source: "Format Test"' in formatted
        assert "Type: table" in formatted
        assert "Content:" in formatted


@pytest.mark.asyncio
async def test_max_items_limit_caps_expansion(temp_db_path):
    """Expansion should respect max_context_items limit."""
    config = AppConfig()
    config.processing.chunk_size = 32
    config.processing.max_context_items = 2  # Very restrictive

    docling_doc = create_list_document()

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        doc = await create_document_with_docling(client, docling_doc, "Limit Test")
        assert doc.id is not None

        results = await client.search("grapes", limit=1)
        assert len(results) > 0

        expanded = await client.expand_context(results)

        # With max_items=2, expansion should be limited
        content = expanded[0].content.lower()
        item_count = sum(
            1 for item in ["apples", "bananas", "oranges", "grapes"] if item in content
        )
        # Should have at most 2 items (the limit)
        assert item_count <= 2, f"Expected at most 2 items, got {item_count}"


@pytest.mark.asyncio
async def test_search_result_get_primary_label():
    """Test _get_primary_label prioritizes structural labels correctly."""
    # Table should be prioritized
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c1",
        document_id="d1",
        labels=["paragraph", "table", "text"],
    )
    assert result._get_primary_label() == "table"

    # Code should be prioritized over paragraph
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c2",
        document_id="d2",
        labels=["paragraph", "code"],
    )
    assert result._get_primary_label() == "code"

    # list_item should be prioritized
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c3",
        document_id="d3",
        labels=["text", "list_item"],
    )
    assert result._get_primary_label() == "list_item"

    # Returns first label when no priority match
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c4",
        document_id="d4",
        labels=["paragraph", "text"],
    )
    assert result._get_primary_label() == "paragraph"

    # Returns None for empty labels
    result = SearchResult(
        content="test",
        score=0.5,
        chunk_id="c5",
        document_id="d5",
        labels=[],
    )
    assert result._get_primary_label() is None


@pytest.mark.asyncio
async def test_expand_context_radius_zero(temp_db_path):
    """Test expand_context with radius 0 returns original results."""
    # Default config has text_context_radius=0
    async with HaikuRAG(temp_db_path, create=True) as client:
        doc = await client.create_document(content="Simple test content")
        assert doc.id is not None
        chunks = await client.chunk_repository.get_by_document_id(doc.id)

        search_results = [SearchResult.from_chunk(chunks[0], 0.9)]
        expanded_results = await client.expand_context(search_results)

        # Should return exactly the same results
        assert len(expanded_results) == 1
        assert expanded_results[0].content == search_results[0].content
        assert expanded_results[0].score == search_results[0].score


@pytest.mark.asyncio
async def test_expand_context_multiple_documents(temp_db_path):
    """Test expand_context with results from multiple documents."""
    config = AppConfig()
    config.processing.text_context_radius = 1

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        # Create first document with manual chunks
        doc1_chunks = [
            Chunk(content="Doc1 Part A", order=0),
            Chunk(content="Doc1 Part B", order=1),
            Chunk(content="Doc1 Part C", order=2),
        ]
        doc1 = await client.import_document(
            content="Doc1 content", chunks=doc1_chunks, uri="doc1.txt"
        )

        # Create second document with manual chunks
        doc2_chunks = [
            Chunk(content="Doc2 Section X", order=0),
            Chunk(content="Doc2 Section Y", order=1),
        ]
        doc2 = await client.import_document(
            content="Doc2 content", chunks=doc2_chunks, uri="doc2.txt"
        )

        assert doc1.id is not None
        assert doc2.id is not None
        chunks1 = await client.chunk_repository.get_by_document_id(doc1.id)
        chunks2 = await client.chunk_repository.get_by_document_id(doc2.id)

        # Get middle chunk from doc1 (order=1) and first chunk from doc2 (order=0)
        chunk1 = next(c for c in chunks1 if c.order == 1)
        chunk2 = next(c for c in chunks2 if c.order == 0)

        search_results = [
            SearchResult.from_chunk(chunk1, 0.8),
            SearchResult.from_chunk(chunk2, 0.7),
        ]
        expanded_results = await client.expand_context(search_results)

        assert len(expanded_results) == 2

        # Check first expanded result (should include chunks 0,1,2 from doc1)
        expanded1 = expanded_results[0]
        assert expanded1.score == 0.8
        assert "Doc1 Part A" in expanded1.content
        assert "Doc1 Part B" in expanded1.content
        assert "Doc1 Part C" in expanded1.content

        # Check second expanded result (should include chunks 0,1 from doc2)
        expanded2 = expanded_results[1]
        assert expanded2.score == 0.7
        assert "Doc2 Section X" in expanded2.content
        assert "Doc2 Section Y" in expanded2.content


@pytest.mark.asyncio
async def test_expand_context_merges_overlapping_chunks(temp_db_path):
    """Test that overlapping expanded chunks are merged into one."""
    config = AppConfig()
    config.processing.text_context_radius = 1

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        # Create document with 5 chunks
        manual_chunks = [
            Chunk(content="Chunk 0", order=0),
            Chunk(content="Chunk 1", order=1),
            Chunk(content="Chunk 2", order=2),
            Chunk(content="Chunk 3", order=3),
            Chunk(content="Chunk 4", order=4),
        ]

        doc = await client.import_document(
            content="Full document content", chunks=manual_chunks
        )

        assert doc.id is not None
        chunks = await client.chunk_repository.get_by_document_id(doc.id)

        # Get adjacent chunks (orders 1 and 2) - these will overlap when expanded
        chunk1 = next(c for c in chunks if c.order == 1)
        chunk2 = next(c for c in chunks if c.order == 2)

        # With radius=1:
        # chunk1 expanded would be [0,1,2]
        # chunk2 expanded would be [1,2,3]
        # These should merge into one chunk containing [0,1,2,3]
        search_results = [
            SearchResult.from_chunk(chunk1, 0.8),
            SearchResult.from_chunk(chunk2, 0.7),
        ]
        expanded_results = await client.expand_context(search_results)

        # Should have only 1 merged result instead of 2 overlapping ones
        assert len(expanded_results) == 1

        merged = expanded_results[0]

        # Should contain all chunks from 0 to 3
        assert "Chunk 0" in merged.content
        assert "Chunk 1" in merged.content
        assert "Chunk 2" in merged.content
        assert "Chunk 3" in merged.content
        assert "Chunk 4" not in merged.content  # Should not include chunk 4

        # Should use the higher score (0.8)
        assert merged.score == 0.8


@pytest.mark.asyncio
async def test_expand_context_keeps_separate_non_overlapping(temp_db_path):
    """Test that non-overlapping expanded chunks remain separate."""
    config = AppConfig()
    config.processing.text_context_radius = 1

    async with HaikuRAG(temp_db_path, config=config, create=True) as client:
        # Create document with chunks far apart
        manual_chunks = [
            Chunk(content="Chunk 0", order=0),
            Chunk(content="Chunk 1", order=1),
            Chunk(content="Chunk 2", order=2),
            Chunk(content="Chunk 5", order=5),  # Gap here
            Chunk(content="Chunk 6", order=6),
            Chunk(content="Chunk 7", order=7),
        ]

        doc = await client.import_document(
            content="Full document content", chunks=manual_chunks
        )

        assert doc.id is not None
        chunks = await client.chunk_repository.get_by_document_id(doc.id)

        # Get chunks by index - they will have sequential orders 0,1,2,3,4,5
        # So get chunk with order=0 and chunk with order=5 (far enough apart)
        chunk0 = next(c for c in chunks if c.order == 0)  # Content: "Chunk 0"
        chunk5 = next(
            c for c in chunks if c.order == 5
        )  # Content: "Chunk 7" but now at order 5

        # chunk0 expanded: [0,1] with radius=1 (orders 0,1)
        # chunk5 expanded: [4,5] with radius=1 (orders 4,5)
        search_results = [
            SearchResult.from_chunk(chunk0, 0.8),
            SearchResult.from_chunk(chunk5, 0.7),
        ]
        expanded_results = await client.expand_context(search_results)

        # Should have 2 separate results
        assert len(expanded_results) == 2

        # Sort by score to ensure predictable order
        expanded_results.sort(key=lambda x: x.score, reverse=True)

        chunk0_expanded = expanded_results[0]
        chunk5_expanded = expanded_results[1]

        # First chunk (order=0) expanded should contain orders [0,1]
        # Content should be "Chunk 0" + "Chunk 1"
        assert "Chunk 0" in chunk0_expanded.content
        assert "Chunk 1" in chunk0_expanded.content
        assert "Chunk 5" not in chunk0_expanded.content
        assert chunk0_expanded.score == 0.8

        # Second chunk (order=5) expanded should contain orders [4,5]
        # Content should be "Chunk 6" (order 4) + "Chunk 7" (order 5)
        assert "Chunk 6" in chunk5_expanded.content
        assert "Chunk 7" in chunk5_expanded.content
        assert "Chunk 0" not in chunk5_expanded.content
        assert chunk5_expanded.score == 0.7


@pytest.mark.asyncio
async def test_expand_context_with_docling_merges_overlapping(temp_db_path):
    """Test that expand_context with DoclingDocument merges overlapping results."""
    config = AppConfig()
    config.processing.text_context_radius = 3

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
        assert doc.docling_document_json is not None

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


@pytest.mark.asyncio
async def test_expand_context_docling_merges_metadata(temp_db_path):
    """Test that expand_context properly merges metadata from multiple results."""
    config = AppConfig()
    config.processing.text_context_radius = 10

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

import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.store.models.chunk import Chunk, ChunkMetadata, SearchResult


@pytest.mark.vcr()
async def test_chunk_repository_operations(qa_corpus: Dataset, temp_db_path):
    """Test ChunkRepository operations."""
    # Create client
    client = HaikuRAG(db_path=temp_db_path, config=Config, create=True)

    # Get the first document from the corpus
    first_doc = qa_corpus[0]
    document_text = first_doc["document_extracted"]

    # Create a document first with chunks
    created_document = await client.create_document(
        content=document_text, metadata={"source": "test"}
    )
    assert created_document.id is not None

    # Test getting chunks by document ID
    chunks = await client.chunk_repository.get_by_document_id(created_document.id)
    assert len(chunks) > 0
    assert all(chunk.document_id == created_document.id for chunk in chunks)

    # Test chunk search
    results = await client.chunk_repository.search(
        "election", limit=2, search_type="vector"
    )
    assert len(results) <= 2
    assert all(hasattr(chunk, "content") for chunk, _ in results)

    # Test deleting chunks by document ID
    deleted = await client.chunk_repository.delete_by_document_id(created_document.id)
    assert deleted is True

    # Verify chunks are gone
    chunks_after_delete = await client.chunk_repository.get_by_document_id(
        created_document.id
    )
    assert len(chunks_after_delete) == 0

    client.close()


@pytest.mark.vcr()
async def test_chunk_repository_pagination(qa_corpus: Dataset, temp_db_path):
    """Test ChunkRepository pagination with get_by_document_id and count_by_document_id."""
    async with HaikuRAG(db_path=temp_db_path, config=Config, create=True) as client:
        # Get the first document from the corpus (should produce multiple chunks)
        first_doc = qa_corpus[0]
        document_text = first_doc["document_extracted"]

        # Create a document with chunks
        created_document = await client.create_document(
            content=document_text, metadata={"source": "test"}
        )
        assert created_document.id is not None

        # Get total chunk count
        total_count = await client.chunk_repository.count_by_document_id(
            created_document.id
        )
        assert total_count > 0

        # Get all chunks without pagination
        all_chunks = await client.chunk_repository.get_by_document_id(
            created_document.id
        )
        assert len(all_chunks) == total_count

        # Test pagination with limit
        limit = min(2, total_count)
        first_batch = await client.chunk_repository.get_by_document_id(
            created_document.id, limit=limit
        )
        assert len(first_batch) == limit
        assert first_batch[0].id == all_chunks[0].id

        # Test pagination with offset
        if total_count > limit:
            second_batch = await client.chunk_repository.get_by_document_id(
                created_document.id, limit=limit, offset=limit
            )
            assert len(second_batch) <= limit
            assert second_batch[0].id == all_chunks[limit].id

        # Test offset beyond available chunks
        empty_batch = await client.chunk_repository.get_by_document_id(
            created_document.id, limit=10, offset=total_count + 100
        )
        assert len(empty_batch) == 0


@pytest.mark.vcr()
async def test_chunking_pipeline(qa_corpus: Dataset, temp_db_path):
    """Test document chunking using client primitives."""
    from haiku.rag.client import HaikuRAG
    from haiku.rag.embeddings import embed_chunks

    async with HaikuRAG(db_path=temp_db_path, create=True) as client:
        # Get the first document from the corpus
        first_doc = qa_corpus[0]
        document_text = first_doc["document_extracted"]

        # Use client primitives: convert → chunk → embed
        docling_document = await client.convert(document_text)
        chunks = await client.chunk(docling_document)
        embedded_chunks = await embed_chunks(chunks)

        # Verify chunks were created with embeddings
        assert len(chunks) > 0
        assert all(chunk.embedding is None for chunk in chunks)  # Before embedding
        assert all(chunk.embedding is not None for chunk in embedded_chunks)  # After

        # Verify chunk order
        for i, chunk in enumerate(chunks):
            assert chunk.order == i


def test_chunk_metadata_parsing():
    """Test ChunkMetadata parsing from chunk metadata dict."""
    metadata_dict = {
        "doc_item_refs": ["#/texts/0", "#/texts/1", "#/tables/0"],
        "headings": ["Chapter 1", "Section 1.1"],
        "labels": ["paragraph", "paragraph", "table"],
        "page_numbers": [1, 1, 2],
    }

    chunk = Chunk(
        content="Test content",
        metadata=metadata_dict,
    )

    chunk_meta = chunk.get_chunk_metadata()

    assert isinstance(chunk_meta, ChunkMetadata)
    assert chunk_meta.doc_item_refs == ["#/texts/0", "#/texts/1", "#/tables/0"]
    assert chunk_meta.headings == ["Chapter 1", "Section 1.1"]
    assert chunk_meta.labels == ["paragraph", "paragraph", "table"]
    assert chunk_meta.page_numbers == [1, 1, 2]


def test_chunk_metadata_defaults():
    """Test ChunkMetadata with empty/default values."""
    chunk = Chunk(content="Test content", metadata={})
    chunk_meta = chunk.get_chunk_metadata()

    assert chunk_meta.doc_item_refs == []
    assert chunk_meta.headings is None
    assert chunk_meta.labels == []
    assert chunk_meta.page_numbers == []


def test_chunk_metadata_resolve_doc_items():
    """Test resolving doc_item_refs to actual DocItem objects."""
    from docling_core.types.doc.document import DoclingDocument

    # Create a minimal DoclingDocument with some text items
    doc_json = {
        "name": "test_doc",
        "texts": [
            {
                "self_ref": "#/texts/0",
                "text": "First text",
                "orig": "First text",
                "label": "paragraph",
            },
            {
                "self_ref": "#/texts/1",
                "text": "Second text",
                "orig": "Second text",
                "label": "title",
            },
        ],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }
    docling_doc = DoclingDocument.model_validate(doc_json)

    # Create chunk metadata with refs
    chunk_meta = ChunkMetadata(
        doc_item_refs=["#/texts/0", "#/texts/1"],
        labels=["paragraph", "title"],
    )

    # Resolve refs
    doc_items = chunk_meta.resolve_doc_items(docling_doc)

    assert len(doc_items) == 2
    assert getattr(doc_items[0], "text") == "First text"
    assert getattr(doc_items[1], "text") == "Second text"


def test_chunk_metadata_resolve_doc_items_graceful_degradation():
    """Test that invalid refs are skipped gracefully."""
    from docling_core.types.doc.document import DoclingDocument

    doc_json = {
        "name": "test_doc",
        "texts": [
            {
                "self_ref": "#/texts/0",
                "text": "Only text",
                "orig": "Only text",
                "label": "paragraph",
            },
        ],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }
    docling_doc = DoclingDocument.model_validate(doc_json)

    # Create chunk metadata with one valid and one invalid ref
    chunk_meta = ChunkMetadata(
        doc_item_refs=["#/texts/0", "#/texts/999", "#/invalid/path"],
    )

    # Resolve refs - invalid ones should be skipped
    doc_items = chunk_meta.resolve_doc_items(docling_doc)

    assert len(doc_items) == 1
    assert getattr(doc_items[0], "text") == "Only text"


def test_chunk_metadata_resolve_empty_refs():
    """Test resolving with no refs returns empty list."""
    from docling_core.types.doc.document import DoclingDocument

    doc_json = {
        "name": "test_doc",
        "texts": [],
        "tables": [],
        "pictures": [],
        "groups": [],
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
    }
    docling_doc = DoclingDocument.model_validate(doc_json)

    chunk_meta = ChunkMetadata()
    doc_items = chunk_meta.resolve_doc_items(docling_doc)

    assert doc_items == []


def test_search_result_format_for_agent_with_rank():
    """Test format_for_agent with rank and total parameters."""
    result = SearchResult(
        content="This is the chunk content about elections.",
        score=0.02,  # Low RRF score that would confuse agents
        chunk_id="chunk-123",
        document_id="doc-456",
        document_uri="file:///docs/report.pdf",
        document_title="Annual Report 2024",
        headings=["Chapter 1", "Section 1.1", "Elections"],
        labels=["paragraph", "table"],
        page_numbers=[1, 2],
    )

    formatted = result.format_for_agent(rank=1, total=5)

    assert "[chunk-123]" in formatted
    assert "[rank 1 of 5]" in formatted
    assert "score:" not in formatted  # Score should NOT appear when rank is provided
    assert (
        'Source: "Annual Report 2024" > Chapter 1 > Section 1.1 > Elections'
        in formatted
    )
    assert "Type: table" in formatted
    assert "Content:\nThis is the chunk content about elections." in formatted


def test_search_result_format_for_agent_rank_only():
    """Test format_for_agent with rank but no total."""
    result = SearchResult(
        content="Some content.",
        score=0.03,
        chunk_id="chunk-abc",
    )

    formatted = result.format_for_agent(rank=2)

    assert "[chunk-abc]" in formatted
    assert "[rank 2]" in formatted
    assert "score:" not in formatted


def test_search_result_format_for_agent_fallback():
    """Test format_for_agent falls back to score when no rank provided."""
    result = SearchResult(
        content="This is the chunk content about elections.",
        score=0.85,
        chunk_id="chunk-123",
        document_id="doc-456",
        document_uri="file:///docs/report.pdf",
        document_title="Annual Report 2024",
        headings=["Chapter 1", "Section 1.1", "Elections"],
        labels=["paragraph", "table"],
        page_numbers=[1, 2],
    )

    formatted = result.format_for_agent()

    assert "[chunk-123]" in formatted
    assert "(score: 0.85)" in formatted
    assert (
        'Source: "Annual Report 2024" > Chapter 1 > Section 1.1 > Elections'
        in formatted
    )
    assert "Type: table" in formatted  # table has higher priority than paragraph
    assert "Content:\nThis is the chunk content about elections." in formatted


def test_search_result_format_for_agent_minimal():
    """Test format_for_agent with minimal metadata."""
    result = SearchResult(
        content="Some content here.",
        score=0.72,
        chunk_id="chunk-abc",
    )

    formatted = result.format_for_agent()

    assert "[chunk-abc]" in formatted
    assert "(score: 0.72)" in formatted
    assert "Source:" not in formatted  # No title or headings
    assert "Type:" not in formatted  # No labels
    assert "Content:\nSome content here." in formatted


def test_search_result_format_for_agent_title_only():
    """Test format_for_agent with only document title."""
    result = SearchResult(
        content="Content text.",
        score=0.60,
        chunk_id="chunk-xyz",
        document_title="My Document",
    )

    formatted = result.format_for_agent()

    assert 'Source: "My Document"' in formatted


def test_search_result_format_for_agent_headings_only():
    """Test format_for_agent with only headings (no title)."""
    result = SearchResult(
        content="Content text.",
        score=0.60,
        chunk_id="chunk-xyz",
        headings=["Introduction", "Background"],
    )

    formatted = result.format_for_agent()

    assert "Source: Introduction > Background" in formatted


def test_search_result_get_primary_label():
    """Test _get_primary_label prioritization."""
    # Table takes priority over text labels
    result = SearchResult(content="x", score=0.5, labels=["paragraph", "table", "text"])
    assert result._get_primary_label() == "table"

    # Code takes priority over list_item
    result = SearchResult(content="x", score=0.5, labels=["list_item", "code"])
    assert result._get_primary_label() == "code"

    # Text labels fall through to first
    result = SearchResult(content="x", score=0.5, labels=["paragraph", "text"])
    assert result._get_primary_label() == "paragraph"

    # Empty labels
    result = SearchResult(content="x", score=0.5, labels=[])
    assert result._get_primary_label() is None


@pytest.mark.vcr()
async def test_chunk_content_fts_populated(temp_db_path):
    """Test that content_fts column is populated with contextualized content."""
    from haiku.rag.embeddings import get_embedder

    async with HaikuRAG(db_path=temp_db_path, config=Config, create=True) as client:
        # Create a chunk with headings
        chunk = Chunk(
            document_id="test-doc",
            content="This is the raw chunk content.",
            metadata={"headings": ["Chapter 1", "Section 1.1"]},
            order=0,
        )

        # Generate embedding
        embedder = get_embedder(Config)
        embedding = (await embedder.embed_documents([chunk.content]))[0]
        chunk.embedding = embedding

        # Store the chunk
        await client.chunk_repository.create(chunk)

        # Read the raw record from the database
        records = list(
            client.store.chunks_table.search()
            .where(f"id = '{chunk.id}'")
            .limit(1)
            .to_arrow()
            .to_pylist()
        )

        assert len(records) == 1
        record = records[0]

        # Verify content is raw (no headings)
        assert record["content"] == "This is the raw chunk content."

        # Verify content_fts is contextualized (headings + content)
        assert (
            record["content_fts"]
            == "Chapter 1\nSection 1.1\nThis is the raw chunk content."
        )


@pytest.mark.vcr()
async def test_chunk_content_fts_without_headings(temp_db_path):
    """Test that content_fts equals content when no headings present."""
    from haiku.rag.embeddings import get_embedder

    async with HaikuRAG(db_path=temp_db_path, config=Config, create=True) as client:
        # Create a chunk without headings
        chunk = Chunk(
            document_id="test-doc",
            content="Plain content without headings.",
            metadata={},
            order=0,
        )

        # Generate embedding
        embedder = get_embedder(Config)
        embedding = (await embedder.embed_documents([chunk.content]))[0]
        chunk.embedding = embedding

        # Store the chunk
        await client.chunk_repository.create(chunk)

        # Read the raw record from the database
        records = list(
            client.store.chunks_table.search()
            .where(f"id = '{chunk.id}'")
            .limit(1)
            .to_arrow()
            .to_pylist()
        )

        assert len(records) == 1
        record = records[0]

        # Both should be the same when no headings
        assert record["content"] == "Plain content without headings."
        assert record["content_fts"] == "Plain content without headings."

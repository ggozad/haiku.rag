import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import Chunk


@pytest.mark.asyncio
async def test_client_document_crud(qa_corpus: Dataset):
    """Test HaikuRAG CRUD operations for documents."""
    async with HaikuRAG(":memory:") as client:
        # Get test data
        first_doc = qa_corpus[0]
        document_text = first_doc["document_extracted"]
        test_uri = "file:///path/to/test.txt"
        test_metadata = {"source": "test", "topic": "testing"}

        # Test create_document
        created_doc = await client.create_document(
            content=document_text, uri=test_uri, metadata=test_metadata
        )

        assert created_doc.id is not None
        assert created_doc.content == document_text
        assert created_doc.uri == test_uri
        assert created_doc.metadata == test_metadata

        # Test get_document_by_id
        retrieved_doc = await client.get_document_by_id(created_doc.id)
        assert retrieved_doc is not None
        assert retrieved_doc.id == created_doc.id
        assert retrieved_doc.content == document_text
        assert retrieved_doc.uri == test_uri

        # Test get_document_by_uri
        retrieved_by_uri = await client.get_document_by_uri(test_uri)
        assert retrieved_by_uri is not None
        assert retrieved_by_uri.id == created_doc.id
        assert retrieved_by_uri.content == document_text

        # Test get_document_by_uri with non-existent URI
        non_existent = await client.get_document_by_uri("file:///non/existent.txt")
        assert non_existent is None

        # Test update_document
        retrieved_doc.content = "Updated content"
        retrieved_doc.uri = "file:///updated/path.txt"
        updated_doc = await client.update_document(retrieved_doc)
        assert updated_doc.content == "Updated content"
        assert updated_doc.uri == "file:///updated/path.txt"

        # Test list_documents
        all_docs = await client.list_documents()
        assert len(all_docs) == 1
        assert all_docs[0].id == created_doc.id

        # Test list_documents with pagination
        limited_docs = await client.list_documents(limit=10, offset=0)
        assert len(limited_docs) == 1

        # Test delete_document
        deleted = await client.delete_document(created_doc.id)
        assert deleted is True

        # Verify document is gone
        retrieved_doc = await client.get_document_by_id(created_doc.id)
        assert retrieved_doc is None

        # Test delete non-existent document
        deleted_again = await client.delete_document(created_doc.id)
        assert deleted_again is False


@pytest.mark.asyncio
async def test_client_create_document_from_source():
    """Test creating a document from a file source."""
    async with HaikuRAG(":memory:") as client:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_content = "This is test content from a file."
            temp_path = Path(temp_dir) / "test.txt"
            temp_path.write_text(test_content)

            # Test create_document_from_source with Path
            doc = await client.create_document_from_source(source=temp_path)

            assert doc.id is not None
            assert doc.content == test_content
            assert doc.uri == temp_path.as_uri()
            assert "contentType" in doc.metadata
            assert "md5" in doc.metadata
            assert doc.metadata["contentType"] == "text/plain"

            # Test create_document_from_source with string path
            doc2 = await client.create_document_from_source(source=str(temp_path))

            assert doc2.id is not None
            assert doc2.content == test_content
            assert doc2.uri == temp_path.as_uri()
            assert "contentType" in doc2.metadata
            assert "md5" in doc2.metadata


@pytest.mark.asyncio
async def test_client_create_document_from_source_unsupported():
    """Test creating a document from an unsupported file type."""
    async with HaikuRAG(":memory:") as client:
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".unsupported", delete=False
        ) as f:
            f.write("content")
            temp_path = Path(f.name)

            # Should raise ValueError for unsupported extension
            with pytest.raises(ValueError, match="Unsupported file extension"):
                await client.create_document_from_source(temp_path)


@pytest.mark.asyncio
async def test_client_create_document_from_source_nonexistent():
    """Test creating a document from a non-existent file."""
    async with HaikuRAG(":memory:") as client:
        non_existent_path = Path("/non/existent/file.txt")

        # Should raise ValueError when file doesn't exist
        with pytest.raises(ValueError, match="File does not exist"):
            await client.create_document_from_source(non_existent_path)


@pytest.mark.asyncio
async def test_client_create_document_from_url():
    """Test creating a document from a URL."""
    async with HaikuRAG(":memory:") as client:
        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.content = b"<html><body><h1>Test Page</h1><p>This is test content from a webpage.</p></body></html>"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = AsyncMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            doc = await client.create_document_from_source(
                source="https://example.com/test.html", metadata={"source_type": "web"}
            )

            assert doc.id is not None
            assert "Test Page" in doc.content
            assert "test content" in doc.content
            assert doc.uri == "https://example.com/test.html"
            assert doc.metadata["source_type"] == "web"
            assert "contentType" in doc.metadata
            assert "md5" in doc.metadata
            assert doc.metadata["contentType"] == "text/html"


@pytest.mark.asyncio
async def test_client_create_document_from_url_with_different_content_types():
    """Test creating documents from URLs with different content types."""
    async with HaikuRAG(":memory:") as client:
        # Test JSON content
        mock_json_response = AsyncMock()
        mock_json_response.content = (
            b'{"title": "Test JSON", "content": "This is JSON content"}'
        )
        mock_json_response.headers = {"content-type": "application/json"}
        mock_json_response.raise_for_status = AsyncMock()

        with patch("httpx.AsyncClient.get", return_value=mock_json_response):
            doc = await client.create_document_from_source(
                "https://api.example.com/data.json"
            )

            assert doc.id is not None
            assert "Test JSON" in doc.content
            assert doc.uri == "https://api.example.com/data.json"
            assert "contentType" in doc.metadata
            assert "md5" in doc.metadata
            assert doc.metadata["contentType"] == "application/json"

        # Test plain text content
        mock_text_response = AsyncMock()
        mock_text_response.content = b"This is plain text content from a URL."
        mock_text_response.headers = {"content-type": "text/plain"}
        mock_text_response.raise_for_status = AsyncMock()

        with patch("httpx.AsyncClient.get", return_value=mock_text_response):
            doc = await client.create_document_from_source(
                "https://example.com/readme.txt"
            )

            assert doc.id is not None
            assert doc.content == "This is plain text content from a URL."
            assert doc.uri == "https://example.com/readme.txt"
            assert "contentType" in doc.metadata
            assert "md5" in doc.metadata
            assert doc.metadata["contentType"] == "text/plain"


@pytest.mark.asyncio
async def test_client_create_document_from_url_unsupported_content():
    """Test creating a document from URL with unsupported content type."""
    async with HaikuRAG(":memory:") as client:
        # Mock response with unsupported content type
        mock_response = AsyncMock()
        mock_response.content = b"binary content"
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.raise_for_status = AsyncMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            with pytest.raises(ValueError, match="Unsupported content type"):
                await client.create_document_from_source(
                    "https://example.com/binary.bin"
                )


@pytest.mark.asyncio
async def test_client_create_document_from_url_http_error():
    """Test handling HTTP errors when creating document from URL."""
    async with HaikuRAG(":memory:") as client:
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                "404 Not Found",
                request=httpx.Request("GET", "https://example.com/notfound.html"),
                response=httpx.Response(404),
            )

            with pytest.raises(httpx.HTTPStatusError):
                await client.create_document_from_source(
                    "https://example.com/notfound.html"
                )


@pytest.mark.asyncio
async def test_get_extension_from_content_type_or_url():
    """Test the helper method for determining file extensions."""
    async with HaikuRAG(":memory:") as client:
        # Test content type mappings
        assert (
            client._get_extension_from_content_type_or_url("", "text/html") == ".html"
        )
        assert (
            client._get_extension_from_content_type_or_url("", "application/pdf")
            == ".pdf"
        )
        assert (
            client._get_extension_from_content_type_or_url("", "text/plain") == ".txt"
        )

        # Test URL extension detection
        assert (
            client._get_extension_from_content_type_or_url(
                "https://example.com/doc.pdf", ""
            )
            == ".pdf"
        )
        assert (
            client._get_extension_from_content_type_or_url(
                "https://example.com/data.json", ""
            )
            == ".json"
        )

        # Test default fallback
        assert (
            client._get_extension_from_content_type_or_url("https://example.com/", "")
            == ".html"
        )

        # Test content type priority over URL extension
        assert (
            client._get_extension_from_content_type_or_url(
                "https://example.com/file.txt", "application/pdf"
            )
            == ".pdf"
        )


@pytest.mark.asyncio
async def test_client_metadata_content_type_and_md5():
    """Test that contentType and md5 metadata are correctly set."""
    import hashlib

    async with HaikuRAG(":memory:") as client:
        # Create a temporary file with known content
        test_content = "Test content for MD5 calculation."
        expected_md5 = hashlib.md5(test_content.encode()).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.txt"
            temp_path.write_text(test_content)

            doc = await client.create_document_from_source(temp_path)

            assert doc.metadata["contentType"] == "text/plain"
            assert doc.metadata["md5"] == expected_md5

            mock_response = AsyncMock()
            mock_response.content = test_content.encode()
            mock_response.headers = {"content-type": "text/plain"}
            mock_response.raise_for_status = AsyncMock()

            with patch("httpx.AsyncClient.get", return_value=mock_response):
                url_doc = await client.create_document_from_source(
                    "https://example.com/test.txt"
                )

                assert url_doc.metadata["contentType"] == "text/plain"
                assert url_doc.metadata["md5"] == expected_md5


@pytest.mark.asyncio
async def test_client_create_update_no_op_behavior():
    """Test create/update/no-op behavior based on MD5 changes."""
    async with HaikuRAG(":memory:") as client:
        # Create a temporary file
        test_content = "Original content for testing."
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test.txt"
            temp_path.write_text(test_content)

            # First call - should create new document
            doc1 = await client.create_document_from_source(temp_path)
            assert doc1.id is not None
            assert doc1.content == test_content
            original_id = doc1.id

            # Second call with same content - should return existing document (no-op)
            doc2 = await client.create_document_from_source(temp_path)
            assert doc2.id == original_id  # Same document
            assert doc2.content == test_content

            # Modify file content
            updated_content = "Updated content for testing."
            temp_path.write_text(updated_content)

            # Third call with changed content - should update existing document
            doc3 = await client.create_document_from_source(temp_path)
            assert doc3.id == original_id  # Same document ID
            assert doc3.content == updated_content  # Updated content

            # Verify the document was actually updated in database
            retrieved_doc = await client.get_document_by_id(original_id)
            assert retrieved_doc is not None
            assert retrieved_doc.content == updated_content


@pytest.mark.asyncio
async def test_client_url_create_update_no_op_behavior():
    """Test create/update/no-op behavior for URLs based on MD5 changes."""
    async with HaikuRAG(":memory:") as client:
        url = "https://example.com/test.txt"
        original_content = b"Original URL content"
        updated_content = b"Updated URL content"

        # Mock first response
        mock_response1 = AsyncMock()
        mock_response1.content = original_content
        mock_response1.headers = {"content-type": "text/plain"}
        mock_response1.raise_for_status = AsyncMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response1):
            # First call - should create new document
            doc1 = await client.create_document_from_source(url)
            assert doc1.id is not None
            original_id = doc1.id

            # Second call with same content - should return existing document (no-op)
            doc2 = await client.create_document_from_source(url)
            assert doc2.id == original_id  # Same document

        mock_response2 = AsyncMock()
        mock_response2.content = updated_content
        mock_response2.headers = {"content-type": "text/plain"}
        mock_response2.raise_for_status = AsyncMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response2):
            # Third call with changed content - should update existing document
            doc3 = await client.create_document_from_source(url)
            assert doc3.id == original_id  # Same document ID
            assert doc3.content == updated_content.decode()  # Updated content


@pytest.mark.asyncio
async def test_client_search():
    """Test HaikuRAG search functionality."""
    async with HaikuRAG(":memory:") as client:
        # Add multiple documents to search from
        doc1_text = "Python is a high-level programming language known for its simplicity and readability."
        doc2_text = "Machine learning algorithms help computers learn patterns from data without explicit programming."
        doc3_text = "Data science combines statistics, programming, and domain expertise to extract insights."

        # Create documents
        doc1 = await client.create_document(
            content=doc1_text, uri="doc1.txt", metadata={"topic": "python"}
        )
        doc2 = await client.create_document(
            content=doc2_text, uri="doc2.txt", metadata={"topic": "ml"}
        )
        await client.create_document(
            content=doc3_text, uri="doc3.txt", metadata={"topic": "data_science"}
        )

        # Test search with keyword that should match doc1
        results = await client.search("Python programming", limit=3)

        assert len(results) > 0
        assert all(len(result) == 2 for result in results)

        # Verify first result is from the Python document (doc1)
        first_chunk, _ = results[0]
        assert first_chunk.document_id == doc1.id

        # Test search with different query
        ml_results = await client.search("machine learning data", limit=2)
        assert len(ml_results) > 0

        # Verify first result is from the machine learning document (doc2)
        first_ml_chunk, _ = ml_results[0]
        assert first_ml_chunk.document_id == doc2.id

        # Test search with limit parameter
        limited_results = await client.search("programming", limit=1)
        assert len(limited_results) <= 1


@pytest.mark.asyncio
async def test_client_async_context_manager():
    """Test HaikuRAG as async context manager."""

    # Test that context manager works and auto-closes
    async with HaikuRAG(":memory:") as client:
        # Create a document to ensure the client works
        doc = await client.create_document(
            content="Test content for context manager",
            uri="test://context",
            metadata={"test": "context_manager"},
        )

        assert doc.id is not None
        assert doc.content == "Test content for context manager"

        # Test search works within context
        results = await client.search("Test content", limit=1)
        assert len(results) > 0

    # Context manager should have automatically closed the connection
    # We can't easily test that the connection is closed without accessing internals,
    # but the test passing means the context manager methods work correctly


@pytest.mark.asyncio
async def test_client_create_document_with_custom_chunks():
    """Test creating a document with pre-created chunks."""
    async with HaikuRAG(":memory:") as client:
        # Create some custom chunks with and without embeddings
        chunks = [
            Chunk(content="This is the first chunk", metadata={"custom": "metadata1"}),
            Chunk(
                content="This is the second chunk",
                metadata={"custom": "metadata2"},
                embedding=[0.1] * 1024,
            ),  # With embedding
            Chunk(content="This is the third chunk", metadata={"custom": "metadata3"}),
        ]

        # Create document with custom chunks
        document = await client.create_document(
            content="Full document content", chunks=chunks
        )

        assert document.id is not None
        assert document.content == "Full document content"

        # Verify the chunks were created correctly
        doc_chunks = await client.chunk_repository.get_by_document_id(document.id)
        assert len(doc_chunks) == 3

        # Check chunks have correct content, document_id, and order from list position
        for i, chunk in enumerate(doc_chunks):
            assert chunk.document_id == document.id
            assert chunk.content == chunks[i].content
            assert (
                chunk.metadata["order"] == i
            )  # Order should be set from list position
            assert (
                chunk.metadata["custom"] == f"metadata{i + 1}"
            )  # Original metadata preserved


@pytest.mark.asyncio
async def test_client_ask_without_cite():
    """Test asking questions without citations."""
    async with HaikuRAG(":memory:") as client:
        # Mock the QA agent
        mock_qa_agent = AsyncMock()
        mock_qa_agent.answer.return_value = "Test answer"

        with patch("haiku.rag.qa.get_qa_agent", return_value=mock_qa_agent):
            answer = await client.ask("What is Python?")

        assert answer == "Test answer"
        mock_qa_agent.answer.assert_called_once_with("What is Python?")


@pytest.mark.asyncio
async def test_client_ask_with_cite():
    """Test asking questions with citations."""
    async with HaikuRAG(":memory:") as client:
        # Mock the QA agent
        mock_qa_agent = AsyncMock()
        mock_qa_agent.answer.return_value = "Test answer with citations [1]"

        with patch("haiku.rag.qa.get_qa_agent", return_value=mock_qa_agent):
            answer = await client.ask("What is Python?", cite=True)

        assert answer == "Test answer with citations [1]"
        mock_qa_agent.answer.assert_called_once_with("What is Python?")


@pytest.mark.asyncio
async def test_client_expand_context():
    """Test expanding search results with adjacent chunks."""
    # Mock Config to have CONTEXT_CHUNK_RADIUS = 2
    with patch("haiku.rag.client.Config.CONTEXT_CHUNK_RADIUS", 2):
        async with HaikuRAG(":memory:") as client:
            # Create chunks manually
            manual_chunks = [
                Chunk(content="Chunk 0 content", metadata={"order": 0}),
                Chunk(content="Chunk 1 content", metadata={"order": 1}),
                Chunk(content="Chunk 2 content", metadata={"order": 2}),
                Chunk(content="Chunk 3 content", metadata={"order": 3}),
                Chunk(content="Chunk 4 content", metadata={"order": 4}),
            ]

            doc = await client.create_document(
                content="Full document content",
                uri="test_doc.txt",
                chunks=manual_chunks,
            )

            # Get all chunks for the document
            assert doc.id is not None
            chunks = await client.chunk_repository.get_by_document_id(doc.id)
            assert len(chunks) == 5

            # Find the middle chunk (order=2)
            middle_chunk = next(c for c in chunks if c.metadata.get("order") == 2)
            search_results = [(middle_chunk, 0.8)]

            # Test expand_context
            expanded_results = await client.expand_context(search_results)

            assert len(expanded_results) == 1
            expanded_chunk, score = expanded_results[0]

            # Check that the expanded chunk has combined content
            assert expanded_chunk.id == middle_chunk.id
            assert score == 0.8
            assert "Chunk 2 content" in expanded_chunk.content

            # Should include all chunks (radius=2 from chunk 2 = chunks 0,1,2,3,4)
            assert "Chunk 0 content" in expanded_chunk.content
            assert "Chunk 1 content" in expanded_chunk.content
            assert "Chunk 2 content" in expanded_chunk.content
            assert "Chunk 3 content" in expanded_chunk.content
            assert "Chunk 4 content" in expanded_chunk.content


@pytest.mark.asyncio
async def test_client_expand_context_radius_zero():
    """Test expand_context with radius 0 returns original results."""
    with patch("haiku.rag.client.Config.CONTEXT_CHUNK_RADIUS", 0):
        async with HaikuRAG(":memory:") as client:
            # Create a simple document
            doc = await client.create_document(content="Simple test content")
            assert doc.id is not None
            chunks = await client.chunk_repository.get_by_document_id(doc.id)

            search_results = [(chunks[0], 0.9)]
            expanded_results = await client.expand_context(search_results)

            # Should return exactly the same results
            assert expanded_results == search_results


@pytest.mark.asyncio
async def test_client_expand_context_multiple_chunks():
    """Test expand_context with multiple search results."""
    with patch("haiku.rag.client.Config.CONTEXT_CHUNK_RADIUS", 1):
        async with HaikuRAG(":memory:") as client:
            # Create first document with manual chunks
            doc1_chunks = [
                Chunk(content="Doc1 Part A", metadata={"order": 0}),
                Chunk(content="Doc1 Part B", metadata={"order": 1}),
                Chunk(content="Doc1 Part C", metadata={"order": 2}),
            ]
            doc1 = await client.create_document(
                content="Doc1 content", uri="doc1.txt", chunks=doc1_chunks
            )

            # Create second document with manual chunks
            doc2_chunks = [
                Chunk(content="Doc2 Section X", metadata={"order": 0}),
                Chunk(content="Doc2 Section Y", metadata={"order": 1}),
            ]
            doc2 = await client.create_document(
                content="Doc2 content", uri="doc2.txt", chunks=doc2_chunks
            )

            assert doc1.id is not None
            assert doc2.id is not None
            chunks1 = await client.chunk_repository.get_by_document_id(doc1.id)
            chunks2 = await client.chunk_repository.get_by_document_id(doc2.id)

            # Get middle chunk from doc1 (order=1) and first chunk from doc2 (order=0)
            chunk1 = next(c for c in chunks1 if c.metadata.get("order") == 1)
            chunk2 = next(c for c in chunks2 if c.metadata.get("order") == 0)

            search_results = [(chunk1, 0.8), (chunk2, 0.7)]
            expanded_results = await client.expand_context(search_results)

            assert len(expanded_results) == 2

            # Check first expanded result (should include chunks 0,1,2 from doc1)
            expanded1, score1 = expanded_results[0]
            assert expanded1.id == chunk1.id
            assert score1 == 0.8
            assert "Doc1 Part A" in expanded1.content
            assert "Doc1 Part B" in expanded1.content
            assert "Doc1 Part C" in expanded1.content

            # Check second expanded result (should include chunks 0,1 from doc2)
            expanded2, score2 = expanded_results[1]
            assert expanded2.id == chunk2.id
            assert score2 == 0.7
            assert "Doc2 Section X" in expanded2.content
            assert "Doc2 Section Y" in expanded2.content


@pytest.mark.asyncio
async def test_client_expand_context_merges_overlapping_chunks():
    """Test that overlapping expanded chunks are merged into one."""
    async with HaikuRAG(":memory:") as client:
        # Create document with 5 chunks
        manual_chunks = [
            Chunk(content="Chunk 0", metadata={"order": 0}),
            Chunk(content="Chunk 1", metadata={"order": 1}),
            Chunk(content="Chunk 2", metadata={"order": 2}),
            Chunk(content="Chunk 3", metadata={"order": 3}),
            Chunk(content="Chunk 4", metadata={"order": 4}),
        ]

        doc = await client.create_document(
            content="Full document content", chunks=manual_chunks
        )

        assert doc.id is not None
        chunks = await client.chunk_repository.get_by_document_id(doc.id)

        # Get adjacent chunks (orders 1 and 2) - these will overlap when expanded
        chunk1 = next(c for c in chunks if c.metadata.get("order") == 1)
        chunk2 = next(c for c in chunks if c.metadata.get("order") == 2)

        # With radius=1:
        # chunk1 expanded would be [0,1,2]
        # chunk2 expanded would be [1,2,3]
        # These should merge into one chunk containing [0,1,2,3]
        search_results = [(chunk1, 0.8), (chunk2, 0.7)]
        expanded_results = await client.expand_context(search_results, radius=1)

        # Should have only 1 merged result instead of 2 overlapping ones
        assert len(expanded_results) == 1

        merged_chunk, score = expanded_results[0]

        # Should contain all chunks from 0 to 3
        assert "Chunk 0" in merged_chunk.content
        assert "Chunk 1" in merged_chunk.content
        assert "Chunk 2" in merged_chunk.content
        assert "Chunk 3" in merged_chunk.content
        assert "Chunk 4" not in merged_chunk.content  # Should not include chunk 4

        # Should use the higher score (0.8)
        assert score == 0.8


@pytest.mark.asyncio
async def test_client_expand_context_keeps_separate_non_overlapping():
    """Test that non-overlapping expanded chunks remain separate."""
    async with HaikuRAG(":memory:") as client:
        # Create document with chunks far apart
        manual_chunks = [
            Chunk(content="Chunk 0", metadata={"order": 0}),
            Chunk(content="Chunk 1", metadata={"order": 1}),
            Chunk(content="Chunk 2", metadata={"order": 2}),
            Chunk(content="Chunk 5", metadata={"order": 5}),  # Gap here
            Chunk(content="Chunk 6", metadata={"order": 6}),
            Chunk(content="Chunk 7", metadata={"order": 7}),
        ]

        doc = await client.create_document(
            content="Full document content", chunks=manual_chunks
        )

        assert doc.id is not None
        chunks = await client.chunk_repository.get_by_document_id(doc.id)

        # Get chunks by index - they will have sequential orders 0,1,2,3,4,5
        # So get chunk with order=0 and chunk with order=5 (far enough apart)
        chunk0 = next(
            c for c in chunks if c.metadata.get("order") == 0
        )  # Content: "Chunk 0"
        chunk5 = next(
            c for c in chunks if c.metadata.get("order") == 5
        )  # Content: "Chunk 7"

        # chunk0 expanded: [0,1] with radius=1 (orders 0,1)
        # chunk5 expanded: [4,5] with radius=1 (orders 4,5)
        # These should remain separate (max_order 1 < min_order 4 - 1)
        search_results = [(chunk0, 0.8), (chunk5, 0.7)]
        expanded_results = await client.expand_context(search_results, radius=1)

        # Should have 2 separate results
        assert len(expanded_results) == 2

        # Sort by score to ensure predictable order
        expanded_results.sort(key=lambda x: x[1], reverse=True)

        chunk0_expanded, score1 = expanded_results[0]
        chunk5_expanded, score2 = expanded_results[1]

        # First chunk (order=0) expanded should contain orders [0,1]
        # Content should be "Chunk 0" + "Chunk 1"
        assert "Chunk 0" in chunk0_expanded.content
        assert "Chunk 1" in chunk0_expanded.content
        assert (
            "Chunk 7" not in chunk0_expanded.content
        )  # Should not have chunk 7 content
        assert score1 == 0.8

        # Second chunk (order=5) expanded should contain orders [4,5]
        # Content should be "Chunk 6" + "Chunk 7" (orders 4 and 5)
        assert "Chunk 6" in chunk5_expanded.content  # Order 4 content
        assert "Chunk 7" in chunk5_expanded.content  # Order 5 content
        assert "Chunk 0" not in chunk5_expanded.content
        assert score2 == 0.7

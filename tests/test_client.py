import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG


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

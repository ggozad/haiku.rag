from unittest.mock import Mock, patch

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from haiku.rag.chunkers import get_chunker
from haiku.rag.chunkers.docling_local import DoclingLocalChunker
from haiku.rag.chunkers.docling_serve import DoclingServeChunker
from haiku.rag.config import AppConfig, Config
from haiku.rag.converters import get_converter


@pytest.mark.asyncio
async def test_local_chunker(qa_corpus: Dataset):
    """Test DoclingLocalChunker with real document."""
    chunker = DoclingLocalChunker()
    doc_text = qa_corpus[0]["document_extracted"]

    # Convert text to DoclingDocument
    converter = get_converter(Config)
    doc = converter.convert_text(doc_text, name="test.md")

    chunks = await chunker.chunk(doc)

    # Ensure that the text is split into multiple chunks
    assert len(chunks) > 1

    # Load tokenizer for verification
    tokenizer = AutoTokenizer.from_pretrained(chunker.tokenizer_name)

    # Ensure that chunks are reasonably sized (allowing more flexibility for structure-aware chunking)
    total_tokens = 0
    for chunk in chunks:
        encoded_tokens = tokenizer.encode(chunk, add_special_tokens=False)
        token_count = len(encoded_tokens)
        total_tokens += token_count

        # Each chunk should be reasonably sized (allowing more flexibility than the old strict limits)
        assert (
            token_count <= chunker.chunk_size * 1.2
        )  # Allow some flexibility for semantic boundaries
        assert token_count > 5  # Ensure chunks aren't too small

    # Ensure that all chunks together contain roughly the same content as original
    original_tokens = len(tokenizer.encode(doc_text, add_special_tokens=False))

    # Due to structure-aware chunking, we might have some variation in token count
    # but it should be reasonable
    assert abs(total_tokens - original_tokens) <= original_tokens * 0.1


@pytest.mark.asyncio
async def test_local_chunker_custom_config():
    """Test DoclingLocalChunker with custom configuration."""
    config = AppConfig()
    config.processing.chunk_size = 128
    config.processing.chunking_tokenizer = "Qwen/Qwen3-Embedding-0.6B"

    chunker = DoclingLocalChunker(config)
    assert chunker.chunk_size == 128
    assert chunker.tokenizer_name == "Qwen/Qwen3-Embedding-0.6B"


def test_get_chunker_docling_local():
    """Test factory returns DoclingLocalChunker for docling-local."""
    config = AppConfig()
    config.processing.chunker = "docling-local"
    chunker = get_chunker(config)
    assert isinstance(chunker, DoclingLocalChunker)


def test_get_chunker_invalid():
    """Test factory raises error for invalid chunker."""
    config = AppConfig()
    config.processing.chunker = "invalid-chunker"
    with pytest.raises(ValueError, match="Unsupported chunker"):
        get_chunker(config)


@pytest.mark.asyncio
async def test_local_chunker_hierarchical(qa_corpus: Dataset):
    """Test DoclingLocalChunker with hierarchical chunking."""
    config = AppConfig()
    config.processing.chunker_type = "hierarchical"
    chunker = DoclingLocalChunker(config)

    doc_text = qa_corpus[0]["document_extracted"]
    converter = get_converter(Config)
    doc = converter.convert_text(doc_text, name="test.md")

    chunks = await chunker.chunk(doc)

    # Hierarchical chunker should produce chunks
    assert len(chunks) > 0
    # Each chunk should be non-empty
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_local_chunker_invalid_type():
    """Test DoclingLocalChunker raises error for invalid chunker_type."""
    config = AppConfig()
    config.processing.chunker_type = "invalid-type"
    with pytest.raises(ValueError, match="Unsupported chunker_type"):
        DoclingLocalChunker(config)


@pytest.mark.asyncio
async def test_local_chunker_markdown_tables():
    """Test DoclingLocalChunker with markdown table serialization."""
    markdown_with_table = """# Test Document

| Column 1 | Column 2 |
|----------|----------|
| Value A  | Value B  |
| Value D  | Value E  |
"""

    converter = get_converter(Config)
    doc = converter.convert_text(markdown_with_table, name="test.md")

    # Test with markdown tables enabled
    config_md = AppConfig()
    config_md.processing.chunking_use_markdown_tables = True
    chunker_md = DoclingLocalChunker(config_md)
    chunks_md = await chunker_md.chunk(doc)

    # Should contain markdown table format
    assert any("|" in chunk for chunk in chunks_md)
    assert any("Column 1" in chunk for chunk in chunks_md)

    # Test with markdown tables disabled (narrative format)
    config_narrative = AppConfig()
    config_narrative.processing.chunking_use_markdown_tables = False
    chunker_narrative = DoclingLocalChunker(config_narrative)
    chunks_narrative = await chunker_narrative.chunk(doc)

    # Should contain narrative format (no pipe characters in table)
    table_content = [chunk for chunk in chunks_narrative if "Value" in chunk][0]
    # Narrative format uses commas, not pipes for table structure
    assert "," in table_content and "|" not in table_content


def test_get_chunker_docling_serve():
    """Test factory returns DoclingServeChunker for docling-serve."""
    config = AppConfig()
    config.processing.chunker = "docling-serve"
    chunker = get_chunker(config)
    assert isinstance(chunker, DoclingServeChunker)


class TestDoclingServeChunker:
    """Tests for DoclingServeChunker (mocked)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = AppConfig()
        config.providers.docling_serve.base_url = "http://localhost:5001"
        config.providers.docling_serve.api_key = ""
        config.providers.docling_serve.timeout = 300
        config.processing.chunk_size = 256
        config.processing.chunking_tokenizer = "Qwen/Qwen3-Embedding-0.6B"
        return config

    @pytest.fixture
    def chunker(self, config):
        """Create DoclingServeChunker instance."""
        return DoclingServeChunker(config)

    @pytest.mark.asyncio
    @patch("haiku.rag.chunkers.docling_serve.requests.post")
    async def test_chunk_success(self, mock_post, chunker):
        """Test successful chunking via docling-serve."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chunks": [
                {"text": "Chunk 1", "chunk_index": 0},
                {"text": "Chunk 2", "chunk_index": 1},
            ]
        }
        mock_post.return_value = mock_response

        # Create a simple document
        converter = get_converter(Config)
        doc = converter.convert_text("# Test\n\nContent", name="test.md")

        chunks = await chunker.chunk(doc)
        assert len(chunks) == 2
        assert chunks[0] == "Chunk 1"
        assert chunks[1] == "Chunk 2"
        mock_post.assert_called_once()

    @pytest.mark.asyncio
    @patch("haiku.rag.chunkers.docling_serve.requests.post")
    async def test_chunk_with_api_key(self, mock_post, config):
        """Test that API key is included in request headers."""
        config.providers.docling_serve.api_key = "test-key"
        chunker = DoclingServeChunker(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chunks": [{"text": "Chunk 1", "chunk_index": 0}]
        }
        mock_post.return_value = mock_response

        converter = get_converter(Config)
        doc = converter.convert_text("# Test", name="test.md")
        await chunker.chunk(doc)

        call_kwargs = mock_post.call_args.kwargs
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Api-Key"] == "test-key"

    @pytest.mark.asyncio
    @patch("haiku.rag.chunkers.docling_serve.requests.post")
    async def test_chunk_hierarchical_endpoint(self, mock_post, config):
        """Test that hierarchical chunker uses correct endpoint."""
        config.processing.chunker_type = "hierarchical"
        chunker = DoclingServeChunker(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chunks": [{"text": "Chunk 1", "chunk_index": 0}]
        }
        mock_post.return_value = mock_response

        converter = get_converter(Config)
        doc = converter.convert_text("# Test", name="test.md")
        await chunker.chunk(doc)

        call_args = mock_post.call_args
        assert "/v1/chunk/hierarchical/file" in call_args[0][0]

    @pytest.mark.asyncio
    @patch("haiku.rag.chunkers.docling_serve.requests.post")
    async def test_chunk_passes_config_parameters(self, mock_post, config):
        """Test that all config parameters are passed to API."""
        config.processing.chunk_size = 512
        config.processing.chunking_merge_peers = False
        config.processing.chunking_use_markdown_tables = True
        chunker = DoclingServeChunker(config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "chunks": [{"text": "Chunk 1", "chunk_index": 0}]
        }
        mock_post.return_value = mock_response

        converter = get_converter(Config)
        doc = converter.convert_text("# Test", name="test.md")
        await chunker.chunk(doc)

        call_kwargs = mock_post.call_args.kwargs
        data = call_kwargs["data"]
        assert data["chunking_max_tokens"] == "512"
        assert data["chunking_merge_peers"] == "false"
        assert data["chunking_use_markdown_tables"] == "true"

    @pytest.mark.asyncio
    @patch("haiku.rag.chunkers.docling_serve.requests.post")
    async def test_chunk_connection_error(self, mock_post, chunker):
        """Test handling of connection errors."""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        converter = get_converter(Config)
        doc = converter.convert_text("# Test", name="test.md")

        with pytest.raises(ValueError, match="Could not connect to docling-serve"):
            await chunker.chunk(doc)

    @pytest.mark.asyncio
    @patch("haiku.rag.chunkers.docling_serve.requests.post")
    async def test_chunk_timeout_error(self, mock_post, chunker):
        """Test handling of timeout errors."""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout("Timeout")

        converter = get_converter(Config)
        doc = converter.convert_text("# Test", name="test.md")

        with pytest.raises(ValueError, match="timed out"):
            await chunker.chunk(doc)

    @pytest.mark.asyncio
    @patch("haiku.rag.chunkers.docling_serve.requests.post")
    async def test_chunk_auth_error(self, mock_post, chunker):
        """Test handling of authentication errors."""
        import requests

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_post.return_value = mock_response

        converter = get_converter(Config)
        doc = converter.convert_text("# Test", name="test.md")

        with pytest.raises(ValueError, match="Authentication failed"):
            await chunker.chunk(doc)

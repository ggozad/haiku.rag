from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from haiku.rag.chunkers import get_chunker
from haiku.rag.chunkers.docling_local import DoclingLocalChunker
from haiku.rag.chunkers.docling_serve import DoclingServeChunker
from haiku.rag.config import AppConfig, Config
from haiku.rag.converters import get_converter


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent / "cassettes" / "test_chunker")


@pytest.mark.asyncio
async def test_local_chunker(qa_corpus: Dataset):
    """Test DoclingLocalChunker with real document."""
    chunker = DoclingLocalChunker()
    doc_text = qa_corpus[0]["document_extracted"]

    # Convert text to DoclingDocument
    converter = get_converter(Config)
    doc = await converter.convert_text(doc_text, name="test.md")

    chunks = await chunker.chunk(doc)

    # Ensure that the text is split into multiple chunks
    assert len(chunks) > 1

    # Load tokenizer for verification
    tokenizer = AutoTokenizer.from_pretrained(chunker.tokenizer_name)

    # Ensure that chunks are reasonably sized (allowing more flexibility for structure-aware chunking)
    total_tokens = 0
    for chunk in chunks:
        encoded_tokens = tokenizer.encode(chunk.content, add_special_tokens=False)
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
    doc = await converter.convert_text(doc_text, name="test.md")

    chunks = await chunker.chunk(doc)

    # Hierarchical chunker should produce chunks
    assert len(chunks) > 0
    # Each chunk should be non-empty
    for chunk in chunks:
        assert len(chunk.content.strip()) > 0


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
    doc = await converter.convert_text(markdown_with_table, name="test.md")

    # Test with markdown tables enabled
    config_md = AppConfig()
    config_md.processing.chunking_use_markdown_tables = True
    chunker_md = DoclingLocalChunker(config_md)
    chunks_md = await chunker_md.chunk(doc)

    # Should contain markdown table format
    assert any("|" in chunk.content for chunk in chunks_md)
    assert any("Column 1" in chunk.content for chunk in chunks_md)

    # Test with markdown tables disabled (narrative format)
    config_narrative = AppConfig()
    config_narrative.processing.chunking_use_markdown_tables = False
    chunker_narrative = DoclingLocalChunker(config_narrative)
    chunks_narrative = await chunker_narrative.chunk(doc)

    # Should contain narrative format (no pipe characters in table)
    table_content = [
        chunk.content for chunk in chunks_narrative if "Value" in chunk.content
    ][0]
    # Narrative format uses commas, not pipes for table structure
    assert "," in table_content and "|" not in table_content


@pytest.mark.asyncio
async def test_local_chunker_sets_order():
    """Test that DoclingLocalChunker sets sequential order on chunks."""
    sample_md = """# Introduction

First paragraph with some content.

## Section One

Second paragraph.

## Section Two

Third paragraph.
"""
    converter = get_converter(Config)
    doc = await converter.convert_text(sample_md, name="test.md")

    chunker = DoclingLocalChunker()
    chunks = await chunker.chunk(doc)

    assert len(chunks) > 0
    # Verify order is set sequentially starting from 0
    for i, chunk in enumerate(chunks):
        assert chunk.order == i, f"Chunk {i} has order {chunk.order}, expected {i}"


@pytest.mark.asyncio
async def test_local_chunker_metadata_extraction():
    """Test that DoclingLocalChunker extracts metadata correctly."""
    sample_md = """# Chapter 1: Introduction

This is the first paragraph of the introduction.

## Section 1.1: Background

Here is some background information.

| Header 1 | Header 2 |
|----------|----------|
| Value 1  | Value 2  |
"""
    converter = get_converter(Config)
    doc = await converter.convert_text(sample_md, name="test.md")

    chunker = DoclingLocalChunker()
    chunks = await chunker.chunk(doc)

    assert len(chunks) > 0

    # Check that at least one chunk has doc_item_refs
    all_refs = []
    all_labels = []
    all_headings = []
    for chunk in chunks:
        meta = chunk.get_chunk_metadata()
        all_refs.extend(meta.doc_item_refs)
        all_labels.extend(meta.labels)
        if meta.headings:
            all_headings.extend(meta.headings)

    # Should have JSON pointer refs like #/texts/0, #/tables/0
    assert len(all_refs) > 0
    assert any(ref.startswith("#/") for ref in all_refs)

    # Should have labels
    assert len(all_labels) > 0
    assert "text" in all_labels or "table" in all_labels

    # Should have headings
    assert len(all_headings) > 0
    assert any("Chapter" in h or "Section" in h for h in all_headings)


def test_get_chunker_docling_serve():
    """Test factory returns DoclingServeChunker for docling-serve."""
    config = AppConfig()
    config.processing.chunker = "docling-serve"
    chunker = get_chunker(config)
    assert isinstance(chunker, DoclingServeChunker)


def create_async_workflow_mocks(
    result_data: dict, task_id: str = "test-task-123"
) -> tuple[Mock, Mock, Mock]:
    """Create mock responses for docling-serve async workflow."""
    submit_response = Mock()
    submit_response.status_code = 200
    submit_response.json.return_value = {"task_id": task_id, "task_status": "pending"}
    submit_response.raise_for_status = Mock()

    poll_response = Mock()
    poll_response.status_code = 200
    poll_response.json.return_value = {"task_id": task_id, "task_status": "success"}
    poll_response.raise_for_status = Mock()

    result_response = Mock()
    result_response.status_code = 200
    result_response.json.return_value = result_data
    result_response.raise_for_status = Mock()

    return submit_response, poll_response, result_response


class TestDoclingServeChunker:
    """Tests for DoclingServeChunker (mocked)."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = AppConfig()
        config.providers.docling_serve.base_url = "http://localhost:5001"
        config.providers.docling_serve.api_key = ""
        config.processing.chunk_size = 256
        config.processing.chunking_tokenizer = "Qwen/Qwen3-Embedding-0.6B"
        return config

    @pytest.fixture
    def chunker(self, config):
        """Create DoclingServeChunker instance."""
        return DoclingServeChunker(config)

    @pytest.mark.asyncio
    @patch("haiku.rag.providers.docling_serve.httpx.AsyncClient")
    async def test_chunk_success(self, mock_client_class, chunker):
        """Test successful chunking via docling-serve async workflow."""
        result_data = {
            "chunks": [
                {"text": "Chunk 1", "chunk_index": 0},
                {"text": "Chunk 2", "chunk_index": 1},
            ]
        }
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(result_data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Create a simple document
        converter = get_converter(Config)
        doc = await converter.convert_text("# Test\n\nContent", name="test.md")

        chunks = await chunker.chunk(doc)
        assert len(chunks) == 2
        assert chunks[0].content == "Chunk 1"
        assert chunks[1].content == "Chunk 2"
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    @patch("haiku.rag.providers.docling_serve.httpx.AsyncClient")
    async def test_chunk_with_api_key(self, mock_client_class, config):
        """Test that API key is included in request headers."""
        config.providers.docling_serve.api_key = "test-key"
        chunker = DoclingServeChunker(config)

        result_data = {"chunks": [{"text": "Chunk 1", "chunk_index": 0}]}
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(result_data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
        mock_client_class.return_value.__aenter__.return_value = mock_client

        converter = get_converter(Config)
        doc = await converter.convert_text("# Test", name="test.md")
        await chunker.chunk(doc)

        call_kwargs = mock_client.post.call_args.kwargs
        assert "headers" in call_kwargs
        assert call_kwargs["headers"]["X-Api-Key"] == "test-key"

    @pytest.mark.asyncio
    @patch("haiku.rag.providers.docling_serve.httpx.AsyncClient")
    async def test_chunk_hierarchical_endpoint(self, mock_client_class, config):
        """Test that hierarchical chunker uses correct endpoint."""
        config.processing.chunker_type = "hierarchical"
        chunker = DoclingServeChunker(config)

        result_data = {"chunks": [{"text": "Chunk 1", "chunk_index": 0}]}
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(result_data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
        mock_client_class.return_value.__aenter__.return_value = mock_client

        converter = get_converter(Config)
        doc = await converter.convert_text("# Test", name="test.md")
        await chunker.chunk(doc)

        call_args = mock_client.post.call_args
        assert "/v1/chunk/hierarchical/file/async" in call_args[0][0]

    @pytest.mark.asyncio
    @patch("haiku.rag.providers.docling_serve.httpx.AsyncClient")
    async def test_chunk_passes_config_parameters(self, mock_client_class, config):
        """Test that all config parameters are passed to API."""
        config.processing.chunk_size = 512
        config.processing.chunking_merge_peers = False
        config.processing.chunking_use_markdown_tables = True
        config.processing.conversion_options.do_ocr = False
        config.processing.conversion_options.force_ocr = True
        config.processing.conversion_options.ocr_engine = "tesseract"
        config.processing.conversion_options.ocr_lang = ["en", "de"]
        chunker = DoclingServeChunker(config)

        result_data = {"chunks": [{"text": "Chunk 1", "chunk_index": 0}]}
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(result_data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
        mock_client_class.return_value.__aenter__.return_value = mock_client

        converter = get_converter(Config)
        doc = await converter.convert_text("# Test", name="test.md")
        await chunker.chunk(doc)

        call_kwargs = mock_client.post.call_args.kwargs
        data = call_kwargs["data"]
        assert data["chunking_max_tokens"] == "512"
        assert data["chunking_merge_peers"] == "false"
        assert data["chunking_use_markdown_tables"] == "true"
        # OCR options from conversion_options
        assert data["convert_do_ocr"] == "false"
        assert data["convert_force_ocr"] == "true"
        assert data["convert_ocr_engine"] == "tesseract"
        assert data["convert_ocr_lang"] == ["en", "de"]

    @pytest.mark.asyncio
    @patch("haiku.rag.providers.docling_serve.httpx.AsyncClient")
    async def test_chunk_omits_empty_ocr_lang(self, mock_client_class, config):
        """Test that ocr_lang is omitted when empty (default)."""
        # Ensure ocr_lang is empty (default)
        config.processing.conversion_options.ocr_lang = []
        chunker = DoclingServeChunker(config)

        result_data = {"chunks": [{"text": "Chunk 1", "chunk_index": 0}]}
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(result_data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
        mock_client_class.return_value.__aenter__.return_value = mock_client

        converter = get_converter(Config)
        doc = await converter.convert_text("# Test", name="test.md")
        await chunker.chunk(doc)

        call_kwargs = mock_client.post.call_args.kwargs
        data = call_kwargs["data"]
        # OCR options should use defaults
        assert data["convert_do_ocr"] == "true"
        assert data["convert_force_ocr"] == "false"
        assert data["convert_ocr_engine"] == "auto"
        # ocr_lang should NOT be present when empty
        assert "convert_ocr_lang" not in data

    @pytest.mark.asyncio
    @patch("haiku.rag.providers.docling_serve.httpx.AsyncClient")
    async def test_chunk_connection_error(self, mock_client_class, chunker):
        """Test handling of connection errors."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        converter = get_converter(Config)
        doc = await converter.convert_text("# Test", name="test.md")

        with pytest.raises(ValueError, match="Could not connect to docling-serve"):
            await chunker.chunk(doc)

    @pytest.mark.asyncio
    @patch("haiku.rag.providers.docling_serve.httpx.AsyncClient")
    async def test_chunk_timeout_error(self, mock_client_class, chunker):
        """Test handling of timeout errors."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        converter = get_converter(Config)
        doc = await converter.convert_text("# Test", name="test.md")

        with pytest.raises(ValueError, match="timed out"):
            await chunker.chunk(doc)

    @pytest.mark.asyncio
    @patch("haiku.rag.providers.docling_serve.httpx.AsyncClient")
    async def test_chunk_auth_error(self, mock_client_class, chunker):
        """Test handling of authentication errors."""
        import httpx

        mock_request = Mock()
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=mock_request, response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        converter = get_converter(Config)
        doc = await converter.convert_text("# Test", name="test.md")

        with pytest.raises(ValueError, match="Authentication failed"):
            await chunker.chunk(doc)

    @pytest.mark.asyncio
    @patch("haiku.rag.providers.docling_serve.httpx.AsyncClient")
    async def test_chunk_metadata_extraction(self, mock_client_class, chunker):
        """Test that metadata is correctly extracted from API response.

        Labels are resolved from the DoclingDocument using the refs, so we need
        to create a document with matching structure for the mocked API response.
        """
        result_data = {
            "chunks": [
                {
                    "text": "Chapter 1\nThis is content.",
                    "doc_items": ["#/texts/0", "#/texts/1"],
                    "headings": ["Chapter 1"],
                    "page_numbers": [1],
                },
                {
                    "text": "Table content here.",
                    "doc_items": ["#/tables/0"],
                    "headings": ["Chapter 1", "Section 1.1"],
                    "page_numbers": [1, 2],
                },
            ]
        }
        submit_resp, poll_resp, result_resp = create_async_workflow_mocks(result_data)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=submit_resp)
        mock_client.get = AsyncMock(side_effect=[poll_resp, result_resp])
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Create a document with texts and tables that match the mocked refs
        converter = get_converter(Config)
        doc = await converter.convert_text(
            """# Chapter 1

This is content.

| Col1 | Col2 |
|------|------|
| A    | B    |
""",
            name="test.md",
        )

        chunks = await chunker.chunk(doc)

        assert len(chunks) == 2

        # First chunk - labels resolved from document
        assert chunks[0].content == "Chapter 1\nThis is content."
        meta0 = chunks[0].get_chunk_metadata()
        assert meta0.doc_item_refs == ["#/texts/0", "#/texts/1"]
        # texts[0] is title (# heading), texts[1] is text (paragraph)
        assert meta0.labels == ["title", "text"]
        assert meta0.headings == ["Chapter 1"]
        assert meta0.page_numbers == [1]

        # Second chunk - label resolved from document
        assert chunks[1].content == "Table content here."
        meta1 = chunks[1].get_chunk_metadata()
        assert meta1.doc_item_refs == ["#/tables/0"]
        assert meta1.labels == ["table"]
        assert meta1.headings == ["Chapter 1", "Section 1.1"]
        assert meta1.page_numbers == [1, 2]


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_local_and_serve_chunkers_produce_same_output():
    """Test that local and serve chunkers produce identical output for the same document.

    Note: Labels are resolved from the DoclingDocument since docling-serve API
    only returns ref strings, not labels. See:
    https://github.com/docling-project/docling-serve/issues/448
    """
    from pathlib import Path

    from haiku.rag.chunkers.docling_local import DoclingLocalChunker
    from haiku.rag.chunkers.docling_serve import DoclingServeChunker
    from haiku.rag.converters.docling_serve import DoclingServeConverter

    # Use docling-serve to convert the PDF (ensures same conversion for both chunkers)
    converter = DoclingServeConverter(Config)
    pdf_path = Path("tests/data/doclaynet.pdf")
    doc = await converter.convert_file(pdf_path)

    # Create both chunkers with same config
    config = AppConfig()
    config.processing.chunk_size = 256
    config.processing.chunker_type = "hybrid"
    config.processing.chunking_merge_peers = True
    config.processing.chunking_use_markdown_tables = True

    local_chunker = DoclingLocalChunker(config)
    serve_chunker = DoclingServeChunker(config)

    # Chunk with both
    local_chunks = await local_chunker.chunk(doc)
    serve_chunks = await serve_chunker.chunk(doc)

    # Same number of chunks
    assert len(local_chunks) == len(serve_chunks), (
        f"Chunk count mismatch: local={len(local_chunks)}, serve={len(serve_chunks)}"
    )

    # Compare each chunk
    for i, (local, serve) in enumerate(zip(local_chunks, serve_chunks)):
        # Text should match
        assert local.content == serve.content, f"Chunk {i} content mismatch"

        local_meta = local.get_chunk_metadata()
        serve_meta = serve.get_chunk_metadata()

        # doc_item_refs should match
        assert local_meta.doc_item_refs == serve_meta.doc_item_refs, (
            f"Chunk {i} doc_item_refs mismatch: "
            f"local={local_meta.doc_item_refs}, serve={serve_meta.doc_item_refs}"
        )

        # Labels should match (now that serve resolves from document)
        assert local_meta.labels == serve_meta.labels, (
            f"Chunk {i} labels mismatch: "
            f"local={local_meta.labels}, serve={serve_meta.labels}"
        )

        # Headings should match
        assert local_meta.headings == serve_meta.headings, (
            f"Chunk {i} headings mismatch: "
            f"local={local_meta.headings}, serve={serve_meta.headings}"
        )

        # Page numbers should match
        assert local_meta.page_numbers == serve_meta.page_numbers, (
            f"Chunk {i} page_numbers mismatch: "
            f"local={local_meta.page_numbers}, serve={serve_meta.page_numbers}"
        )

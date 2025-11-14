import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from haiku.rag.chunkers import get_chunker
from haiku.rag.chunkers.docling_local import DoclingLocalChunker
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

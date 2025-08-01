import pytest
from datasets import Dataset

from haiku.rag.chunker import Chunker


@pytest.mark.asyncio
async def test_chunker(qa_corpus: Dataset):
    chunker = Chunker()
    doc = qa_corpus[0]["document_extracted"]
    chunks = await Chunker().chunk(doc)

    # Ensure that the text is split into multiple chunks
    assert len(chunks) > 1

    # Ensure that chunks are reasonably sized (allowing more flexibility for structure-aware chunking)
    total_tokens = 0
    for chunk in chunks:
        encoded_tokens = Chunker.encoder.encode(chunk, disallowed_special=())
        token_count = len(encoded_tokens)
        total_tokens += token_count

        # Each chunk should be reasonably sized (allowing more flexibility than the old strict limits)
        assert (
            token_count <= chunker.chunk_size * 1.2
        )  # Allow some flexibility for semantic boundaries
        assert token_count > 5  # Ensure chunks aren't too small

    # Ensure that all chunks together contain roughly the same content as original
    original_tokens = len(Chunker.encoder.encode(doc, disallowed_special=()))

    # Due to structure-aware chunking, we might have some variation in token count
    # but it should be reasonable
    assert abs(total_tokens - original_tokens) <= original_tokens * 0.1

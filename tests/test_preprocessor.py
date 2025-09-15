from pathlib import Path

import pytest

from haiku.rag.config import Config
from haiku.rag.store.engine import Store
from haiku.rag.store.models.document import Document
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.utils import text_to_docling_document


@pytest.mark.parametrize(
    "is_async, marker",
    [
        (False, "MARKER_LINE"),
        (True, "ASYNC_MARKER"),
    ],
)
@pytest.mark.asyncio
async def test_markdown_preprocessor_applied_parametrized(
    is_async: bool, marker: str, tmp_path: Path, temp_db_path: Path
):
    """Ensure MARKDOWN_PREPROCESSOR (sync or async) transforms markdown before chunking."""
    pre_file = tmp_path / ("pre_async.py" if is_async else "pre.py")
    if is_async:
        pre_file.write_text(
            """
import asyncio

async def add_marker(text: str) -> str:
    await asyncio.sleep(0)
    return text + "\\n\\nASYNC_MARKER\\n"
"""
        )
    else:
        pre_file.write_text(
            """
def add_marker(text: str) -> str:
    return text + "\\n\\nMARKER_LINE\\n"
"""
        )

    original_pre = Config.MARKDOWN_PREPROCESSOR
    try:
        Config.MARKDOWN_PREPROCESSOR = f"{pre_file}:add_marker"

        store = Store(temp_db_path)
        chunk_repo = ChunkRepository(store)
        doc_repo = DocumentRepository(store)

        document = Document(content="Hello world")
        created_doc = await doc_repo.create(document)
        assert created_doc.id is not None

        # Stub embeddings to avoid network
        dim = chunk_repo.embedder._vector_dim

        async def fake_embed(x):  # type: ignore[override]
            if isinstance(x, list):
                return [[0.0] * dim for _ in x]
            return [0.0] * dim

        chunk_repo.embedder.embed = fake_embed  # type: ignore[assignment]

        docling = text_to_docling_document(document.content, name="test.md")
        chunks = await chunk_repo.create_chunks_for_document(created_doc.id, docling)

        assert any(marker in c.content for c in chunks)
    finally:
        Config.MARKDOWN_PREPROCESSOR = original_pre

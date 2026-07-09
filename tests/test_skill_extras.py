"""Tests for non-tool utilities from ``haiku.rag.skills._tools.create_skill_extras``."""

from docling_core.types.doc.document import DoclingDocument

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig
from haiku.rag.skills._tools import create_skill_extras
from haiku.rag.store.models.chunk import Chunk


async def _seed_chunk(db_path) -> str:
    docling_doc = DoclingDocument(name="extras-test")
    chunk = Chunk(
        content="Some content.",
        metadata={"doc_item_refs": ["#/texts/0"], "page_numbers": [1]},
        order=0,
        embedding=[0.1] * 2560,
    )
    async with HaikuRAG(db_path, create=True) as client:
        doc = await client.import_document(docling_doc, [chunk], uri="test://extras")
        stored = await client.chunk_repository.get_by_document_id(doc.id)
        return stored[0].id


async def test_extras_visualize_chunk_accepts_str_and_list(temp_db_path):
    chunk_id = await _seed_chunk(temp_db_path)
    extras = create_skill_extras(temp_db_path, AppConfig())
    visualize_chunk = extras["visualize_chunk"]

    # A document imported without page images yields no visualizations, but the
    # str and list inputs must both resolve the chunk and reach visualize_chunk.
    assert await visualize_chunk(chunk_id) == []
    assert await visualize_chunk([chunk_id]) == []


async def test_extras_visualize_chunk_unknown_id_returns_empty(temp_db_path):
    await _seed_chunk(temp_db_path)
    extras = create_skill_extras(temp_db_path, AppConfig())
    visualize_chunk = extras["visualize_chunk"]

    assert await visualize_chunk("does-not-exist") == []
    assert await visualize_chunk(["does-not-exist"]) == []

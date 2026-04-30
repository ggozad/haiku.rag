"""A3: SearchResult.image_data, expand_context preservation, multimodal ToolReturn."""

import base64
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from haiku.rag.client import HaikuRAG
from haiku.rag.client.search import _populate_image_data
from haiku.rag.config import Config
from haiku.rag.context import expand_with_items
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.document_item import DocumentItem
from haiku.rag.tools.search import create_search_toolset

PICTURE_BYTES = b"\x89PNG\r\n\x1a\nfake-picture-bytes"
PICTURE_B64 = base64.b64encode(PICTURE_BYTES).decode("ascii")


@pytest.mark.asyncio
async def test_populate_image_data_attaches_base64(temp_db_path):
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.document_item_repository.create_items(
            "doc-1",
            [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="paragraph",
                    text="Some text",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/pictures/0",
                    label="picture",
                    text="",
                    picture_data=PICTURE_BYTES,
                ),
            ],
        )

        text_only = SearchResult(
            content="Some text",
            score=1.0,
            document_id="doc-1",
            doc_item_refs=["#/texts/0"],
            labels=["paragraph"],
        )
        with_picture = SearchResult(
            content="A figure caption",
            score=0.9,
            document_id="doc-1",
            doc_item_refs=["#/texts/0", "#/pictures/0"],
            labels=["paragraph", "picture"],
        )

        await _populate_image_data(rag, [text_only, with_picture])

        # Text-only result is unchanged
        assert text_only.image_data is None
        # Picture-bearing result has the bytes attached
        assert with_picture.image_data == {"#/pictures/0": PICTURE_B64}


@pytest.mark.asyncio
async def test_client_search_include_images_false_skips_lookup(temp_db_path):
    """include_images=False must short-circuit the picture-bytes lookup."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.document_item_repository.create_items(
            "doc-1",
            [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/pictures/0",
                    label="picture",
                    picture_data=PICTURE_BYTES,
                ),
            ],
        )
        # Spy that we never reach the picture-bytes accessor
        rag.document_item_repository.get_pictures_for_chunk = AsyncMock(  # type: ignore[method-assign]
            wraps=rag.document_item_repository.get_pictures_for_chunk
        )

        from haiku.rag.client.search import search

        # Stub the chunk-search results so we don't depend on embeddings/FTS
        async def fake_chunk_search(*args, **kwargs):
            return []

        rag.chunk_repository.search = fake_chunk_search  # type: ignore[method-assign]

        await search(rag, "anything", include_images=False)
        rag.document_item_repository.get_pictures_for_chunk.assert_not_called()


@pytest.mark.asyncio
async def test_expand_context_preserves_picture_refs_with_empty_text(temp_db_path):
    """A picture item with empty text must keep its self_ref through expansion."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        # Build an items table with a section header + a paragraph match + an
        # adjacent picture row that has no text. expand_with_items used to
        # filter the picture out via the `if item.text:` guard; A3 keeps it.
        await rag.document_item_repository.create_items(
            "doc-1",
            [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="section_header",
                    text="Methods",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="paragraph",
                    text="The figure below shows the architecture.",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/pictures/0",
                    label="picture",
                    text="",
                    picture_data=PICTURE_BYTES,
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=3,
                    self_ref="#/texts/2",
                    label="paragraph",
                    text="More commentary follows.",
                ),
            ],
        )

        # Match on the paragraph that mentions the figure.
        seed = SearchResult(
            content="The figure below shows the architecture.",
            score=1.0,
            document_id="doc-1",
            doc_item_refs=["#/texts/1"],
            labels=["paragraph"],
        )
        expanded = await expand_with_items(
            rag.document_item_repository,
            "doc-1",
            [seed],
            max_chars=10_000,
        )
        assert len(expanded) == 1
        out = expanded[0]
        assert "#/pictures/0" in out.doc_item_refs, (
            "Picture self_ref should survive expansion even with empty text"
        )
        assert "picture" in out.labels


@pytest.mark.asyncio
async def test_rechunk_preserves_picture_data(temp_db_path):
    """``rebuild --rechunk`` keeps ``picture_data`` for every picture row."""
    from haiku.rag.client import RebuildMode
    from haiku.rag.client.documents import _store_document_with_chunks
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()

    async with HaikuRAG(temp_db_path, create=True) as rag:
        rag._config.processing.pictures = "image"
        document = Document(content="x", uri="test://doc")
        document.set_docling(docling_doc)
        created = await _store_document_with_chunks(rag, document, [], docling_doc)
        assert created.id is not None
        before = await rag.document_item_repository.get_all_picture_data(created.id)
        assert before.get("#/pictures/0") is not None

        async for _ in rag.rebuild_database(mode=RebuildMode.RECHUNK):
            pass

        after = await rag.document_item_repository.get_all_picture_data(created.id)
        assert after.get("#/pictures/0") == before.get("#/pictures/0")


@pytest.mark.asyncio
async def test_update_clears_picture_data_when_mode_none(temp_db_path):
    """Switching to ``pictures="none"`` and re-running update_document
    drops picture_data — the snapshot/merge gate is what gives users a
    "rebuild reclaims storage" path when they downgrade modes."""
    from haiku.rag.client.documents import (
        _store_document_with_chunks,
        _update_document_with_chunks,
    )
    from haiku.rag.store.models.document import Document
    from tests.store.test_document_items import _docling_doc_with_picture

    docling_doc = _docling_doc_with_picture()

    async with HaikuRAG(temp_db_path, create=True) as rag:
        # Ingest under "image" so picture bytes land in document_items.
        rag._config.processing.pictures = "image"
        document = Document(content="x", uri="test://doc")
        document.set_docling(docling_doc)
        created = await _store_document_with_chunks(rag, document, [], docling_doc)
        assert created.id is not None
        before = await rag.document_item_repository.get_all_picture_data(created.id)
        assert before.get("#/pictures/0") is not None

        # Downgrade to "none" and re-run update with the (already stripped)
        # docling pulled from storage. The snapshot/merge must be skipped so
        # picture_data is cleared on the new items rows.
        rag._config.processing.pictures = "none"
        from_blob = created.get_docling_document()
        assert from_blob is not None
        await _update_document_with_chunks(rag, created, [], from_blob)

        after = await rag.document_item_repository.get_all_picture_data(created.id)
        assert after.get("#/pictures/0") is None


@pytest.mark.asyncio
async def test_expand_context_repopulates_image_data(temp_db_path):
    """expand_context rebuilds SearchResult objects via expand_with_items, so
    it must re-attach picture bytes — otherwise vision flows downstream see
    empty image_data after expansion."""
    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.document_item_repository.create_items(
            "doc-1",
            [
                DocumentItem(
                    document_id="doc-1",
                    position=0,
                    self_ref="#/texts/0",
                    label="section_header",
                    text="Methods",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=1,
                    self_ref="#/texts/1",
                    label="paragraph",
                    text="The figure below shows the architecture.",
                ),
                DocumentItem(
                    document_id="doc-1",
                    position=2,
                    self_ref="#/pictures/0",
                    label="picture",
                    text="",
                    picture_data=PICTURE_BYTES,
                ),
            ],
        )
        seed = SearchResult(
            content="The figure below shows the architecture.",
            score=1.0,
            document_id="doc-1",
            doc_item_refs=["#/texts/1"],
            labels=["paragraph"],
            image_data=None,
        )
        expanded = await rag.expand_context([seed])
        assert len(expanded) == 1
        out = expanded[0]
        assert "#/pictures/0" in out.doc_item_refs
        assert out.image_data == {"#/pictures/0": PICTURE_B64}


@dataclass
class _Deps:
    client: object


@pytest.mark.asyncio
async def test_search_tool_returns_multimodal_when_picture_present():
    """The agent-facing search tool must wrap text + BinaryContent in ToolReturn
    whenever a result carries picture image_data."""
    picture_result = SearchResult(
        content="A diagram of the layout",
        score=1.0,
        chunk_id="chunk-1",
        document_id="doc-1",
        doc_item_refs=["#/pictures/0"],
        labels=["picture"],
        image_data={"#/pictures/0": PICTURE_B64},
    )

    fake_client = AsyncMock()
    fake_client.search = AsyncMock(return_value=[picture_result])
    fake_client.expand_context = AsyncMock(return_value=[picture_result])

    toolset = create_search_toolset(Config, expand_context=False)
    func = toolset.tools["search"].function

    ctx = RunContext(
        deps=_Deps(client=fake_client),  # type: ignore[arg-type]
        model=TestModel(),
        usage=RunUsage(),
        run_id="run-1",
    )
    result = await func(ctx, "anything")

    assert isinstance(result, ToolReturn)
    assert isinstance(result.return_value, str)
    assert "Type: picture" in result.return_value or "rank 1" in result.return_value
    assert result.content is not None
    assert len(result.content) == 1
    part = result.content[0]
    assert isinstance(part, BinaryContent)
    assert part.media_type == "image/png"
    assert part.identifier == "#/pictures/0"
    assert part.data == PICTURE_BYTES


@pytest.mark.asyncio
async def test_search_tool_returns_plain_string_when_no_pictures():
    """When no result carries image_data the tool returns a plain str (no
    ToolReturn wrapper) so non-vision flows are unaffected."""
    text_result = SearchResult(
        content="Some text",
        score=1.0,
        chunk_id="chunk-1",
        document_id="doc-1",
        doc_item_refs=["#/texts/0"],
        labels=["paragraph"],
    )

    fake_client = AsyncMock()
    fake_client.search = AsyncMock(return_value=[text_result])
    fake_client.expand_context = AsyncMock(return_value=[text_result])

    toolset = create_search_toolset(Config, expand_context=False)
    func = toolset.tools["search"].function

    ctx = RunContext(
        deps=_Deps(client=fake_client),  # type: ignore[arg-type]
        model=TestModel(),
        usage=RunUsage(),
        run_id="run-1",
    )
    result = await func(ctx, "anything")

    assert isinstance(result, str)
    assert "rank 1" in result

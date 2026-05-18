"""Tests for skill tool closures from ``haiku.rag.skills._tools.create_skill_tools``.

These cover the vision toggle on the skill ``search`` tool: when the configured
QA model is vision-capable, picture bytes from search results must reach the
sub-agent as ``BinaryContent`` parts (so a vision model can read figures).
When the QA model is not vision-capable, the same search must return plain
text only.
"""

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image as PILImageModule
from pydantic_ai import RunContext
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from haiku.rag.config import AppConfig
from haiku.rag.skills._deps import RAGRunDeps
from haiku.rag.skills._tools import create_skill_tools
from haiku.rag.skills.rag import RAGState
from haiku.rag.store.models.chunk import SearchResult


def _make_png(color: str = "red") -> bytes:
    buf = BytesIO()
    PILImageModule.new("RGB", (4, 4), color).save(buf, "PNG")
    return buf.getvalue()


PICTURE_BYTES = _make_png("red")
PICTURE_B64 = base64.b64encode(PICTURE_BYTES).decode("ascii")


def _picture_result() -> SearchResult:
    return SearchResult(
        content="A diagram of the layout",
        score=1.0,
        chunk_id="chunk-1",
        document_id="doc-1",
        doc_item_refs=["#/pictures/0"],
        labels=["picture"],
        image_data={"#/pictures/0": PICTURE_B64},
    )


def _text_result() -> SearchResult:
    return SearchResult(
        content="Some surrounding paragraph text",
        score=0.9,
        chunk_id="chunk-2",
        document_id="doc-1",
        doc_item_refs=["#/texts/3"],
        labels=["paragraph"],
        image_data=None,
    )


def _make_ctx(rag, state: RAGState) -> RunContext[RAGRunDeps]:
    deps = RAGRunDeps(state=state, rag=rag)
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run-1",
    )


def _build_search_tool(config: AppConfig):
    tools = create_skill_tools(
        db_path=Path("/tmp/unused.lancedb"),
        config=config,
        state_type=RAGState,
        tool_names=["search"],
        model=config.qa.model,
    )
    return tools["search"]


def _fake_rag(results: list[SearchResult]) -> AsyncMock:
    rag = AsyncMock()
    rag.search = AsyncMock(return_value=results)
    rag.expand_context = AsyncMock(return_value=results)
    return rag


@pytest.mark.asyncio
async def test_skill_search_attaches_binary_content_when_vision_capable():
    """vision=True + picture in results → ToolReturn carries BinaryContent."""
    config = AppConfig()
    config.qa.model.vision = True

    search = _build_search_tool(config)
    rag = _fake_rag([_picture_result()])
    ctx = _make_ctx(rag, RAGState())

    result = await search(ctx, "diagram")

    assert isinstance(result, ToolReturn)
    assert isinstance(result.return_value, str)
    assert "rank 1" in result.return_value
    assert result.content is not None
    assert len(result.content) == 1
    part = result.content[0]
    assert isinstance(part, BinaryContent)
    assert part.data == PICTURE_BYTES
    assert part.media_type == "image/png"
    assert part.identifier == "#/pictures/0"


@pytest.mark.asyncio
async def test_skill_search_returns_plain_string_when_not_vision_capable():
    """vision=False (the default) + picture in results → plain text only.
    The picture bytes must not reach a text-only model — providers behave
    inconsistently with image content (Ollama silently accepts and the
    model hallucinates)."""
    config = AppConfig()
    assert config.qa.model.vision is False

    search = _build_search_tool(config)
    rag = _fake_rag([_picture_result()])
    ctx = _make_ctx(rag, RAGState())

    result = await search(ctx, "diagram")

    assert isinstance(result, str)
    assert "rank 1" in result


@pytest.mark.asyncio
async def test_skill_search_returns_plain_string_when_no_pictures():
    """vision=True + no pictures in results → no ToolReturn wrapper, just
    text. The wrapper is only needed when there's actually image content
    to carry."""
    config = AppConfig()
    config.qa.model.vision = True

    search = _build_search_tool(config)
    rag = _fake_rag([_text_result()])
    ctx = _make_ctx(rag, RAGState())

    result = await search(ctx, "paragraph")

    assert isinstance(result, str)
    assert "rank 1" in result


@pytest.mark.asyncio
async def test_skill_search_records_results_into_state():
    """Whether or not the QA model is vision-capable, the SearchResult
    list must land in state.searches[query] so cite/visualize_chunk can
    look chunks up later."""
    config = AppConfig()

    search = _build_search_tool(config)
    rag = _fake_rag([_picture_result(), _text_result()])
    state = RAGState()
    ctx = _make_ctx(rag, state)

    await search(ctx, "anything")

    assert "anything" in state.searches
    assert len(state.searches["anything"]) == 2


@pytest.mark.asyncio
async def test_skill_search_dedups_picture_bytes_by_self_ref():
    """When two search results reference the same picture self_ref (e.g. a
    text chunk and a synthetic picture chunk), the BinaryContent list
    must include that picture exactly once. Otherwise the model receives
    duplicate image content and pays double the image-token cost."""
    config = AppConfig()
    config.qa.model.vision = True

    other = SearchResult(
        content="Surrounding text mentioning the figure",
        score=0.8,
        chunk_id="chunk-3",
        document_id="doc-1",
        doc_item_refs=["#/texts/2", "#/pictures/0"],
        labels=["paragraph", "picture"],
        image_data={"#/pictures/0": PICTURE_B64},
    )
    search = _build_search_tool(config)
    rag = _fake_rag([_picture_result(), other])
    ctx = _make_ctx(rag, RAGState())

    result = await search(ctx, "figure")

    assert isinstance(result, ToolReturn)
    assert result.content is not None
    assert len(result.content) == 1
    assert result.content[0].identifier == "#/pictures/0"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_skill_search_keeps_same_self_ref_from_different_documents():
    """``#/pictures/0`` in document A and ``#/pictures/0`` in document B
    are different figures. Dedup must key on ``(document_id, self_ref)``;
    keying on ``self_ref`` alone would drop document B's bytes."""
    other_bytes = _make_png("blue")
    other_b64 = base64.b64encode(other_bytes).decode("ascii")

    doc_a = SearchResult(
        content="Figure in doc A",
        score=1.0,
        chunk_id="chunk-a",
        document_id="doc-A",
        doc_item_refs=["#/pictures/0"],
        labels=["picture"],
        image_data={"#/pictures/0": PICTURE_B64},
    )
    doc_b = SearchResult(
        content="Figure in doc B",
        score=0.9,
        chunk_id="chunk-b",
        document_id="doc-B",
        doc_item_refs=["#/pictures/0"],
        labels=["picture"],
        image_data={"#/pictures/0": other_b64},
    )

    config = AppConfig()
    config.qa.model.vision = True

    search = _build_search_tool(config)
    rag = _fake_rag([doc_a, doc_b])
    ctx = _make_ctx(rag, RAGState())

    result = await search(ctx, "figure")

    assert isinstance(result, ToolReturn)
    assert result.content is not None
    assert len(result.content) == 2
    payloads = {part.data for part in result.content}  # type: ignore[attr-defined]
    assert PICTURE_BYTES in payloads
    assert other_bytes in payloads

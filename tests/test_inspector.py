import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image as PILImage
from typer.testing import CliRunner

from haiku.rag.cli import _cli as cli
from haiku.rag.store.models import Chunk, Document, SearchResult

runner = CliRunner()


def _png_b64(color: str = "red", size: tuple[int, int] = (8, 8)) -> str:
    img = PILImage.new("RGB", size, color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _make_client(*, vision: bool, image_data: dict[str, str] | None) -> MagicMock:
    """Stub HaikuRAG that returns one expanded SearchResult with the given attachments."""
    from haiku.rag.config import AppConfig

    config = AppConfig()
    config.qa.model.vision = vision

    expanded = SearchResult(
        content="expanded text incl. picture descriptions if any",
        score=0.5,
        chunk_id="chunk-1",
        document_id="doc-1",
        doc_item_refs=["#/texts/0"] + (list(image_data) if image_data else []),
        page_numbers=[1],
        labels=["paragraph"] + (["picture"] if image_data else []),
        image_data=image_data,
    )
    client = MagicMock()
    client._config = config
    client.expand_context = AsyncMock(return_value=[expanded])
    return client


def test_inspect_command():
    """Test inspect command launches inspector TUI."""
    with patch("haiku.rag.inspector.run_inspector") as mock_inspector:
        mock_inspector.return_value = None

        result = runner.invoke(cli, ["inspect"])

        assert result.exit_code == 0
        mock_inspector.assert_called_once()


@pytest.mark.asyncio
async def test_document_list_loads_initial_batch():
    """Test that DocumentList loads only the initial batch on startup."""
    from textual.app import App

    from haiku.rag.inspector.widgets.document_list import DocumentList

    # Create mock documents
    mock_docs = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50)
    ]

    class TestApp(App):
        def compose(self):
            yield DocumentList(id="doc-list")

    app = TestApp()
    async with app.run_test():
        doc_list = app.query_one(DocumentList)

        # Create mock client
        mock_client = AsyncMock()
        mock_client.list_documents = AsyncMock(return_value=mock_docs)

        await doc_list.load_documents(mock_client)

        # Should have called list_documents with a limit (not None)
        mock_client.list_documents.assert_called_once()
        call_kwargs = mock_client.list_documents.call_args
        # The limit should be set (not None) for initial load
        assert call_kwargs.kwargs.get("limit") is not None


@pytest.mark.asyncio
async def test_document_list_load_more():
    """Test that DocumentList can load more documents."""
    from textual.app import App

    from haiku.rag.inspector.widgets.document_list import DocumentList

    # Create mock documents - two batches
    batch1 = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50)
    ]
    batch2 = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50, 100)
    ]

    class TestApp(App):
        def compose(self):
            yield DocumentList(id="doc-list")

    app = TestApp()
    async with app.run_test():
        doc_list = app.query_one(DocumentList)

        mock_client = AsyncMock()
        mock_client.list_documents = AsyncMock(side_effect=[batch1, batch2])

        # Load initial batch
        await doc_list.load_documents(mock_client)
        assert len(doc_list.documents) == 50

        # Load more
        await doc_list.load_more(mock_client)
        assert len(doc_list.documents) == 100

        # Verify offset was used in second call
        second_call = mock_client.list_documents.call_args_list[1]
        assert second_call.kwargs.get("offset") == 50


@pytest.mark.asyncio
async def test_document_list_tracks_has_more():
    """Test that DocumentList tracks whether more documents are available."""
    from textual.app import App

    from haiku.rag.inspector.widgets.document_list import DocumentList

    # First batch returns full page, second returns partial
    batch1 = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50)
    ]
    batch2 = [
        Document(id=f"doc-{i}", content=f"Content {i}", title=f"Doc {i}")
        for i in range(50, 60)
    ]

    class TestApp(App):
        def compose(self):
            yield DocumentList(id="doc-list")

    app = TestApp()
    async with app.run_test():
        doc_list = app.query_one(DocumentList)

        mock_client = AsyncMock()
        mock_client.list_documents = AsyncMock(side_effect=[batch1, batch2])

        await doc_list.load_documents(mock_client)
        # After loading full batch, has_more should be True
        assert doc_list.has_more is True

        await doc_list.load_more(mock_client)
        # After loading partial batch (<50), has_more should be False
        assert doc_list.has_more is False


@pytest.mark.asyncio
async def test_context_modal_renders_pictures_when_vision_enabled():
    """ContextModal must mount one TextualImage per attached picture
    when qa.model.vision is True — that's what the LLM actually sees."""
    from textual.app import App
    from textual_image.widget import Image as TextualImage

    from haiku.rag.inspector.widgets.context_modal import ContextModal

    chunk = Chunk(
        id="chunk-1", document_id="doc-1", content="raw chunk text", metadata={}
    )
    client = _make_client(
        vision=True,
        image_data={"#/pictures/0": _png_b64("red"), "#/pictures/1": _png_b64("blue")},
    )

    class TestApp(App):
        async def on_mount(self) -> None:
            await self.push_screen(ContextModal(chunk=chunk, client=client))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        modal = app.screen
        images = list(modal.query(TextualImage))
        assert len(images) == 2


@pytest.mark.asyncio
async def test_context_modal_suppresses_pictures_when_vision_disabled():
    """ContextModal must NOT mount picture widgets when vision is off,
    even if expansion attached image_data — text-only models would never
    see those bytes, so the inspector shouldn't show them either."""
    from textual.app import App
    from textual_image.widget import Image as TextualImage

    from haiku.rag.inspector.widgets.context_modal import ContextModal

    chunk = Chunk(
        id="chunk-1", document_id="doc-1", content="raw chunk text", metadata={}
    )
    client = _make_client(
        vision=False,
        image_data={"#/pictures/0": _png_b64("red")},
    )

    class TestApp(App):
        async def on_mount(self) -> None:
            await self.push_screen(ContextModal(chunk=chunk, client=client))

    app = TestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        modal = app.screen
        assert list(modal.query(TextualImage)) == []

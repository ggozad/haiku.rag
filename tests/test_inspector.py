from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from haiku.rag.cli import cli
from haiku.rag.store.models import Document

runner = CliRunner()


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

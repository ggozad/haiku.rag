import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haiku.rag.app import HaikuRAGApp
from haiku.rag.store.models.document import Document


@pytest.fixture
def app():
    return HaikuRAGApp(db_path=Path(":memory:"))


@pytest.mark.asyncio
async def test_list_documents(app: HaikuRAGApp, monkeypatch):
    """Test listing documents."""
    mock_docs = [
        Document(id=1, content="doc 1"),
        Document(id=2, content="doc 2"),
    ]
    mock_client = AsyncMock()
    mock_client.list_documents.return_value = mock_docs
    # The async context manager should return the mock client itself
    mock_client.__aenter__.return_value = mock_client

    mock_rich_print = MagicMock()
    mock_console_print = MagicMock()
    monkeypatch.setattr(app, "_rich_print_document", mock_rich_print)
    monkeypatch.setattr(app.console, "print", mock_console_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.list_documents()

    mock_client.list_documents.assert_called_once()
    assert mock_rich_print.call_count == len(mock_docs)
    mock_rich_print.assert_any_call(mock_docs[0], truncate=True)
    mock_rich_print.assert_any_call(mock_docs[1], truncate=True)


@pytest.mark.asyncio
async def test_add_document_from_text(app: HaikuRAGApp, monkeypatch):
    """Test adding a document from text."""
    mock_doc = Document(id=1, content="test document")
    mock_client = AsyncMock()
    mock_client.create_document.return_value = mock_doc
    mock_client.__aenter__.return_value = mock_client

    mock_rich_print = MagicMock()
    mock_print = MagicMock()
    monkeypatch.setattr(app, "_rich_print_document", mock_rich_print)
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.add_document_from_text("test document")

    mock_client.create_document.assert_called_once_with("test document")
    mock_rich_print.assert_called_once_with(mock_doc, truncate=True)
    mock_print.assert_called_once_with(
        "[b]Document with id [cyan]1[/cyan] added successfully.[/b]"
    )


@pytest.mark.asyncio
async def test_add_document_from_source(app: HaikuRAGApp, monkeypatch):
    """Test adding a document from a source path."""
    mock_doc = Document(id=1, content="test document")
    mock_client = AsyncMock()
    mock_client.create_document_from_source.return_value = mock_doc
    mock_client.__aenter__.return_value = mock_client

    mock_rich_print = MagicMock()
    mock_print = MagicMock()
    monkeypatch.setattr(app, "_rich_print_document", mock_rich_print)
    monkeypatch.setattr(app.console, "print", mock_print)

    file_path = "test.txt"
    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.add_document_from_source(file_path)

    mock_client.create_document_from_source.assert_called_once_with(file_path)
    mock_rich_print.assert_called_once_with(mock_doc, truncate=True)
    mock_print.assert_called_once_with(
        "[b]Document with id [cyan]1[/cyan] added successfully.[/b]"
    )


@pytest.mark.asyncio
async def test_get_document(app: HaikuRAGApp, monkeypatch):
    """Test getting a document."""
    mock_doc = Document(id=1, content="test document")
    mock_client = AsyncMock()
    mock_client.get_document_by_id.return_value = mock_doc
    mock_client.__aenter__.return_value = mock_client

    mock_rich_print = MagicMock()
    monkeypatch.setattr(app, "_rich_print_document", mock_rich_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.get_document(1)

    mock_client.get_document_by_id.assert_called_once_with(1)
    mock_rich_print.assert_called_once_with(mock_doc, truncate=False)


@pytest.mark.asyncio
async def test_get_document_not_found(app: HaikuRAGApp, monkeypatch):
    """Test getting a document that does not exist."""
    mock_client = AsyncMock()
    mock_client.get_document_by_id.return_value = None
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.get_document(1)

    mock_client.get_document_by_id.assert_called_once_with(1)
    mock_print.assert_called_once_with("[red]Document with id 1 not found.[/red]")


@pytest.mark.asyncio
async def test_delete_document(app: HaikuRAGApp, monkeypatch):
    """Test deleting a document."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.delete_document(1)

    mock_client.delete_document.assert_called_once_with(1)
    mock_print.assert_called_once_with("[b]Document 1 deleted successfully.[/b]")


@pytest.mark.asyncio
async def test_search(app: HaikuRAGApp, monkeypatch):
    """Test searching for documents."""
    mock_results = [("chunk1", 0.9), ("chunk2", 0.8)]
    mock_client = AsyncMock()
    mock_client.search.return_value = mock_results
    mock_client.__aenter__.return_value = mock_client

    mock_rich_print_search = MagicMock()
    monkeypatch.setattr(app, "_rich_print_search_result", mock_rich_print_search)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.search("query")

    mock_client.search.assert_called_once_with("query", limit=5, k=60)
    assert mock_rich_print_search.call_count == len(mock_results)


@pytest.mark.asyncio
async def test_search_no_results(app: HaikuRAGApp, monkeypatch):
    """Test searching with no results."""
    mock_client = AsyncMock()
    mock_client.search.return_value = []
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.search("query")

    mock_client.search.assert_called_once_with("query", limit=5, k=60)
    mock_print.assert_called_once_with("[red]No results found.[/red]")


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "sse", "http", None])
async def test_serve(app: HaikuRAGApp, monkeypatch, transport):
    """Test the serve method with different transports."""
    mock_server = AsyncMock()
    mock_watcher = MagicMock()
    mock_task = asyncio.create_task(asyncio.sleep(0))
    mock_task.cancel = MagicMock()

    monkeypatch.setattr(
        "haiku.rag.app.create_mcp_server", MagicMock(return_value=mock_server)
    )
    monkeypatch.setattr(
        "haiku.rag.app.FileWatcher", MagicMock(return_value=mock_watcher)
    )
    monkeypatch.setattr("asyncio.create_task", MagicMock(return_value=mock_task))

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        if transport:
            await app.serve(transport=transport)
        else:
            await app.serve()

    if transport == "stdio":
        mock_server.run_stdio_async.assert_called_once()
    elif transport == "sse":
        mock_server.run_sse_async.assert_called_once_with("sse")
    else:
        mock_server.run_http_async.assert_called_once_with("streamable-http")

    mock_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_ask_without_cite(app: HaikuRAGApp, monkeypatch):
    """Test asking a question without citations."""
    mock_answer = "Test answer"
    mock_client = AsyncMock()
    mock_client.ask.return_value = mock_answer
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.ask("test question")

    mock_client.ask.assert_called_once_with("test question", cite=False)


@pytest.mark.asyncio
async def test_ask_with_cite(app: HaikuRAGApp, monkeypatch):
    """Test asking a question with citations."""
    mock_answer = "Test answer with citations"
    mock_client = AsyncMock()
    mock_client.ask.return_value = mock_answer
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.ask("test question", cite=True)

    mock_client.ask.assert_called_once_with("test question", cite=True)

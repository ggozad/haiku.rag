import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haiku.rag.app import HaikuRAGApp
from haiku.rag.store.models.document import Document


@pytest.fixture
def app(tmp_path):
    return HaikuRAGApp(db_path=tmp_path / "test.lancedb")


@pytest.mark.asyncio
async def test_list_documents(app: HaikuRAGApp, monkeypatch):
    """Test listing documents."""
    mock_docs = [
        Document(id="1", content="doc 1"),
        Document(id="2", content="doc 2"),
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
    mock_doc = Document(id="1", content="test document")
    mock_client = AsyncMock()
    mock_client.create_document.return_value = mock_doc
    mock_client.__aenter__.return_value = mock_client

    mock_rich_print = MagicMock()
    mock_print = MagicMock()
    monkeypatch.setattr(app, "_rich_print_document", mock_rich_print)
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.add_document_from_text("test document")

    mock_client.create_document.assert_called_once()
    args, kwargs = mock_client.create_document.call_args
    assert args[0] == "test document"
    assert kwargs.get("metadata") is None
    mock_rich_print.assert_called_once_with(mock_doc, truncate=True)
    mock_print.assert_called_once_with(
        "[bold green]Document 1 added successfully.[/bold green]"
    )


@pytest.mark.asyncio
async def test_add_document_from_source(app: HaikuRAGApp, monkeypatch):
    """Test adding a document from a source path."""
    mock_doc = Document(id="1", content="test document")
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

    mock_client.create_document_from_source.assert_called_once()
    args, kwargs = mock_client.create_document_from_source.call_args
    assert args[0] == file_path
    assert kwargs.get("title") is None
    assert kwargs.get("metadata") is None
    mock_rich_print.assert_called_once_with(mock_doc, truncate=True)
    mock_print.assert_called_once_with(
        "[bold green]Document 1 added successfully.[/bold green]"
    )


@pytest.mark.asyncio
async def test_get_document(app: HaikuRAGApp, monkeypatch):
    """Test getting a document."""
    mock_doc = Document(id="1", content="test document")
    mock_client = AsyncMock()
    mock_client.get_document_by_id.return_value = mock_doc
    mock_client.__aenter__.return_value = mock_client

    mock_rich_print = MagicMock()
    monkeypatch.setattr(app, "_rich_print_document", mock_rich_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.get_document("1")

    mock_client.get_document_by_id.assert_called_once_with("1")
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
        await app.get_document("1")

    mock_client.get_document_by_id.assert_called_once_with("1")
    mock_print.assert_called_once_with("[red]Document with id 1 not found.[/red]")


@pytest.mark.asyncio
async def test_delete_document(app: HaikuRAGApp, monkeypatch):
    """Test deleting a document."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.delete_document("1")

    mock_client.delete_document.assert_called_once_with("1")
    mock_print.assert_called_once_with(
        "[bold green]Document 1 deleted successfully.[/bold green]"
    )


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

    mock_client.search.assert_called_once_with("query", limit=5)
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

    mock_client.search.assert_called_once_with("query", limit=5)
    mock_print.assert_called_once_with("[yellow]No results found.[/yellow]")


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", None])
async def test_serve_mcp_only(app: HaikuRAGApp, monkeypatch, transport):
    """Test the serve method with MCP server only."""
    mock_server = AsyncMock()
    created_tasks = []
    original_create_task = asyncio.create_task

    def track_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        task.cancel()
        return task

    monkeypatch.setattr(
        "haiku.rag.app.create_mcp_server", MagicMock(return_value=mock_server)
    )
    monkeypatch.setattr("haiku.rag.app.asyncio.create_task", track_task)
    monkeypatch.setattr(
        "haiku.rag.app.asyncio.gather", AsyncMock(side_effect=asyncio.CancelledError)
    )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        try:
            await app.serve(
                enable_monitor=False,
                enable_mcp=True,
                mcp_transport=transport,
                enable_a2a=False,
            )
        except asyncio.CancelledError:
            pass

    assert len(created_tasks) == 1


@pytest.mark.asyncio
async def test_serve_monitor_only(app: HaikuRAGApp, monkeypatch):
    """Test the serve method with monitor only."""
    mock_watcher = AsyncMock()
    created_tasks = []
    original_create_task = asyncio.create_task

    def track_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        task.cancel()
        return task

    monkeypatch.setattr(
        "haiku.rag.app.FileWatcher", MagicMock(return_value=mock_watcher)
    )
    monkeypatch.setattr("haiku.rag.app.asyncio.create_task", track_task)
    monkeypatch.setattr(
        "haiku.rag.app.asyncio.gather", AsyncMock(side_effect=asyncio.CancelledError)
    )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        try:
            await app.serve(enable_monitor=True, enable_mcp=False, enable_a2a=False)
        except asyncio.CancelledError:
            pass

    assert len(created_tasks) == 1


@pytest.mark.asyncio
async def test_serve_a2a_only(app: HaikuRAGApp, monkeypatch):
    """Test the serve method with A2A server only."""
    created_tasks = []
    original_create_task = asyncio.create_task

    def track_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        task.cancel()
        return task

    monkeypatch.setattr("haiku.rag.app.asyncio.create_task", track_task)
    monkeypatch.setattr(
        "haiku.rag.app.asyncio.gather", AsyncMock(side_effect=asyncio.CancelledError)
    )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    mock_a2a_app = MagicMock()

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        with patch("haiku.rag.a2a.create_a2a_app", return_value=mock_a2a_app):
            try:
                await app.serve(enable_monitor=False, enable_mcp=False, enable_a2a=True)
            except asyncio.CancelledError:
                pass

    assert len(created_tasks) == 1


@pytest.mark.asyncio
async def test_serve_all_services(app: HaikuRAGApp, monkeypatch):
    """Test the serve method with all services enabled."""
    created_tasks = []
    original_create_task = asyncio.create_task

    def track_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        task.cancel()
        return task

    mock_server = AsyncMock()
    mock_watcher = AsyncMock()
    mock_a2a_app = MagicMock()

    monkeypatch.setattr(
        "haiku.rag.app.create_mcp_server", MagicMock(return_value=mock_server)
    )
    monkeypatch.setattr(
        "haiku.rag.app.FileWatcher", MagicMock(return_value=mock_watcher)
    )
    monkeypatch.setattr("haiku.rag.app.asyncio.create_task", track_task)
    monkeypatch.setattr(
        "haiku.rag.app.asyncio.gather", AsyncMock(side_effect=asyncio.CancelledError)
    )

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        with patch("haiku.rag.a2a.create_a2a_app", return_value=mock_a2a_app):
            try:
                await app.serve(enable_monitor=True, enable_mcp=True, enable_a2a=True)
            except asyncio.CancelledError:
                pass

    assert len(created_tasks) == 3


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


@pytest.mark.asyncio
async def test_ask_with_verbose(app: HaikuRAGApp, monkeypatch):
    """Test asking a question with verbose (should be ignored for non-deep)."""
    mock_answer = "Test answer"
    mock_client = AsyncMock()
    mock_client.ask.return_value = mock_answer
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.ask("test question", verbose=True)

    mock_client.ask.assert_called_once_with("test question", cite=False)


@pytest.mark.asyncio
async def test_ask_with_deep(app: HaikuRAGApp, monkeypatch):
    """Test asking a question with deep QA."""
    from haiku.rag.qa.deep.models import DeepQAAnswer

    mock_output = DeepQAAnswer(answer="Deep QA answer", sources=["test.md"])
    mock_result = MagicMock()
    mock_result.output = mock_output

    mock_graph = AsyncMock()
    mock_graph.run.return_value = mock_result

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        with patch(
            "haiku.rag.qa.deep.graph.build_deep_qa_graph", return_value=mock_graph
        ):
            await app.ask("test question", deep=True)

    mock_graph.run.assert_called_once()
    call_kwargs = mock_graph.run.call_args[1]
    assert call_kwargs["state"].context.original_question == "test question"
    assert call_kwargs["state"].context.use_citations is False


@pytest.mark.asyncio
async def test_ask_with_deep_and_cite(app: HaikuRAGApp, monkeypatch):
    """Test asking a question with deep QA and citations."""
    from haiku.rag.qa.deep.models import DeepQAAnswer

    mock_output = DeepQAAnswer(
        answer="Deep QA answer with citations [test.md]", sources=["test.md"]
    )
    mock_result = MagicMock()
    mock_result.output = mock_output

    mock_graph = AsyncMock()
    mock_graph.run.return_value = mock_result

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        with patch(
            "haiku.rag.qa.deep.graph.build_deep_qa_graph", return_value=mock_graph
        ):
            await app.ask("test question", deep=True, cite=True)

    mock_graph.run.assert_called_once()
    call_kwargs = mock_graph.run.call_args[1]
    assert call_kwargs["state"].context.original_question == "test question"
    assert call_kwargs["state"].context.use_citations is True


@pytest.mark.asyncio
async def test_ask_with_deep_and_verbose(app: HaikuRAGApp, monkeypatch):
    """Test asking a question with deep QA and verbose output."""
    from haiku.rag.qa.deep.models import DeepQAAnswer

    mock_output = DeepQAAnswer(answer="Deep QA answer", sources=["test.md"])
    mock_result = MagicMock()
    mock_result.output = mock_output

    mock_graph = AsyncMock()
    mock_graph.run.return_value = mock_result

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        with patch(
            "haiku.rag.qa.deep.graph.build_deep_qa_graph", return_value=mock_graph
        ):
            await app.ask("test question", deep=True, verbose=True)

    mock_graph.run.assert_called_once()
    call_kwargs = mock_graph.run.call_args[1]
    assert call_kwargs["deps"].console is not None

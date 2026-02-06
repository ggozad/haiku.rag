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

    mock_client.search.assert_called_once_with("query", limit=None, filter=None)
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

    mock_client.search.assert_called_once_with("query", limit=None, filter=None)
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
            await app.serve(enable_monitor=True, enable_mcp=False)
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
        try:
            await app.serve(enable_monitor=True, enable_mcp=True)
        except asyncio.CancelledError:
            pass

    assert len(created_tasks) == 2


@pytest.mark.asyncio
async def test_ask_without_cite(app: HaikuRAGApp, monkeypatch):
    """Test asking a question without citations."""
    mock_answer = "Test answer"
    mock_citations = []
    mock_client = AsyncMock()
    mock_client.ask.return_value = (mock_answer, mock_citations)
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.ask("test question")

    mock_client.ask.assert_called_once_with("test question", filter=None)


@pytest.mark.asyncio
async def test_ask_with_cite(app: HaikuRAGApp, monkeypatch):
    """Test asking a question with citations."""
    from haiku.rag.agents.research.models import Citation

    mock_answer = "Test answer with citations"
    mock_citations = [
        Citation(
            document_id="doc-123",
            chunk_id="chunk-456",
            document_uri="test.md",
            document_title="Test Document",
            page_numbers=[1],
            content="Test content",
        )
    ]
    mock_client = AsyncMock()
    mock_client.ask.return_value = (mock_answer, mock_citations)
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.ask("test question", cite=True)

    mock_client.ask.assert_called_once_with("test question", filter=None)
    # Verify print was called (once for answer, once for citations)
    assert mock_print.call_count >= 1


@pytest.mark.asyncio
async def test_ask_with_deep(app: HaikuRAGApp, monkeypatch):
    """Test asking a question with deep mode uses research graph."""
    import haiku.rag.app as app_module
    from haiku.rag.agents.research.models import ResearchReport

    mock_output = ResearchReport(
        title="Test",
        executive_summary="Deep research answer",
        main_findings=["Finding 1"],
        conclusions=["Conclusion 1"],
        sources_summary="Sources",
    )

    mock_graph = AsyncMock()
    mock_graph.run.return_value = mock_output

    mock_client = AsyncMock()

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)
    monkeypatch.setattr(app_module, "build_research_graph", lambda **kwargs: mock_graph)

    with patch("haiku.rag.app.HaikuRAG") as mock_rag_class:
        mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)
        await app.ask("test question", deep=True)

    # Check if there was an error printed
    print_calls = [str(c) for c in mock_print.call_args_list]
    error_calls = [c for c in print_calls if "Error" in c]
    assert not error_calls, f"Error was printed: {error_calls}"

    mock_graph.run.assert_called_once()
    call_kwargs = mock_graph.run.call_args[1]
    assert call_kwargs["state"].context.original_question == "test question"


@pytest.mark.asyncio
async def test_ask_with_deep_and_cite(app: HaikuRAGApp, monkeypatch):
    """Test asking a question with deep mode (cite is ignored for research graph)."""
    import haiku.rag.app as app_module
    from haiku.rag.agents.research.models import ResearchReport

    mock_output = ResearchReport(
        title="Test",
        executive_summary="Deep research answer",
        main_findings=["Finding 1"],
        conclusions=["Conclusion 1"],
        sources_summary="Sources",
    )

    mock_graph = AsyncMock()
    mock_graph.run.return_value = mock_output

    mock_client = AsyncMock()

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)
    monkeypatch.setattr(app_module, "build_research_graph", lambda **kwargs: mock_graph)

    with patch("haiku.rag.app.HaikuRAG") as mock_rag_class:
        mock_rag_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_rag_class.return_value.__aexit__ = AsyncMock(return_value=None)
        await app.ask("test question", deep=True, cite=True)

    mock_graph.run.assert_called_once()
    call_kwargs = mock_graph.run.call_args[1]
    assert call_kwargs["state"].context.original_question == "test question"


@pytest.mark.asyncio
async def test_history_all_tables(tmp_path, monkeypatch):
    """Test history command shows version history for all tables."""
    from haiku.rag.store.engine import Store

    # Create a real database with some data
    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    await app.history()

    # Should print header and at least one version for each table
    calls = [str(c) for c in mock_print.call_args_list]
    assert any("Version History" in c for c in calls)
    assert any("documents" in c for c in calls)
    assert any("chunks" in c for c in calls)
    assert any("settings" in c for c in calls)


@pytest.mark.asyncio
async def test_history_specific_table(tmp_path, monkeypatch):
    """Test history command for a specific table."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    await app.history(table="documents")

    calls = [str(c) for c in mock_print.call_args_list]
    assert any("Version History" in c for c in calls)
    assert any("documents" in c for c in calls)
    # Should not show other tables
    assert not any("chunks" in c and "documents" not in c for c in calls)


@pytest.mark.asyncio
async def test_history_invalid_table(tmp_path, monkeypatch):
    """Test history command with invalid table name."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    await app.history(table="invalid_table")

    calls = [str(c) for c in mock_print.call_args_list]
    assert any("Unknown table" in c for c in calls)


@pytest.mark.asyncio
async def test_history_with_limit(tmp_path, monkeypatch):
    """Test history command with limit."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    await app.history(limit=1)

    # Should still work with limit
    calls = [str(c) for c in mock_print.call_args_list]
    assert any("Version History" in c for c in calls)


@pytest.mark.asyncio
async def test_history_nonexistent_db(tmp_path, monkeypatch):
    """Test history command when database doesn't exist."""
    db_path = tmp_path / "nonexistent.lancedb"
    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    await app.history()

    calls = [str(c) for c in mock_print.call_args_list]
    assert any("does not exist" in c for c in calls)


@pytest.mark.asyncio
async def test_init_creates_database(tmp_path, monkeypatch):
    """Test init creates a new database."""
    db_path = tmp_path / "new.lancedb"
    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    assert not db_path.exists()
    await app.init()

    assert db_path.exists()
    calls = [str(c) for c in mock_print.call_args_list]
    assert any("initialized" in c for c in calls)


@pytest.mark.asyncio
async def test_init_existing_database(tmp_path, monkeypatch):
    """Test init with existing database shows warning."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "existing.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    await app.init()

    calls = [str(c) for c in mock_print.call_args_list]
    assert any("already exists" in c for c in calls)


@pytest.mark.asyncio
async def test_vacuum(tmp_path, monkeypatch):
    """Test vacuum operation."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    await app.vacuum()

    calls = [str(c) for c in mock_print.call_args_list]
    assert any("Vacuum completed" in c for c in calls)


@pytest.mark.asyncio
async def test_create_index_insufficient_chunks(tmp_path, monkeypatch):
    """Test create_index with insufficient chunks shows warning."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    await app.create_index()

    calls = [str(c) for c in mock_print.call_args_list]
    assert any("Need at least 256 chunks" in c for c in calls)


@pytest.mark.asyncio
async def test_rebuild_empty_database(tmp_path, monkeypatch):
    """Test rebuild with empty database shows warning."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)
    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    await app.rebuild()

    calls = [str(c) for c in mock_print.call_args_list]
    assert any("No documents found" in c for c in calls)


def test_migrate_with_pending_migrations(tmp_path):
    """Test migrate method when migrations are applied."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)

    with patch("haiku.rag.store.engine.Store") as mock_store_class:
        mock_store = MagicMock()
        mock_store.migrate.return_value = ["Migration 1", "Migration 2"]
        mock_store_class.return_value = mock_store

        result = app.migrate()

        mock_store_class.assert_called_once_with(
            db_path,
            config=app.config,
            skip_validation=True,
            skip_migration_check=True,
        )
        mock_store.migrate.assert_called_once()
        mock_store.close.assert_called_once()
        assert result == ["Migration 1", "Migration 2"]


def test_migrate_no_pending_migrations(tmp_path):
    """Test migrate method when no migrations are pending."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)

    with patch("haiku.rag.store.engine.Store") as mock_store_class:
        mock_store = MagicMock()
        mock_store.migrate.return_value = []
        mock_store_class.return_value = mock_store

        result = app.migrate()

        mock_store.migrate.assert_called_once()
        mock_store.close.assert_called_once()
        assert result == []


def test_migrate_closes_store_on_exception(tmp_path):
    """Test migrate method closes store even if migration fails."""
    from haiku.rag.store.engine import Store

    db_path = tmp_path / "test.lancedb"
    store = Store(db_path, create=True)
    store.close()

    app = HaikuRAGApp(db_path=db_path)

    with patch("haiku.rag.store.engine.Store") as mock_store_class:
        mock_store = MagicMock()
        mock_store.migrate.side_effect = Exception("Migration error")
        mock_store_class.return_value = mock_store

        with pytest.raises(Exception, match="Migration error"):
            app.migrate()

        mock_store.close.assert_called_once()


@pytest.mark.asyncio
async def test_rlm(app: HaikuRAGApp, monkeypatch):
    """Test rlm method calls client.rlm and prints results."""
    from haiku.rag.agents.rlm.models import RLMResult

    mock_result = RLMResult(
        answer="The total is 42.",
        program="result = sum(values)\nprint(result)",
    )

    mock_client = AsyncMock()
    mock_client.rlm = AsyncMock(return_value=mock_result)
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.rlm("What is the total?")

    mock_client.rlm.assert_called_once_with(
        "What is the total?", documents=None, filter=None
    )
    calls = [str(c) for c in mock_print.call_args_list]
    assert any("Question" in c for c in calls)
    assert any("Program" in c for c in calls)
    assert any("Answer" in c for c in calls)


@pytest.mark.asyncio
async def test_rlm_with_document_and_filter(app: HaikuRAGApp, monkeypatch):
    """Test rlm method passes document and filter to client."""
    from haiku.rag.agents.rlm.models import RLMResult

    mock_result = RLMResult(
        answer="Answer with filter",
        program="print('filtered')",
    )

    mock_client = AsyncMock()
    mock_client.rlm = AsyncMock(return_value=mock_result)
    mock_client.__aenter__.return_value = mock_client

    mock_print = MagicMock()
    monkeypatch.setattr(app.console, "print", mock_print)

    with patch("haiku.rag.app.HaikuRAG", return_value=mock_client):
        await app.rlm("What is it?", document="doc-123", filter="uri LIKE '%test%'")

    mock_client.rlm.assert_called_once_with(
        "What is it?", documents=["doc-123"], filter="uri LIKE '%test%'"
    )

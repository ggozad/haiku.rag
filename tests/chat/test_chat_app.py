from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from haiku.rag.cli import _cli as cli

runner = CliRunner()


def test_chat_command():
    """Test chat command launches chat TUI."""
    with patch("haiku.rag.chat.run_chat") as mock_chat:
        mock_chat.return_value = None

        result = runner.invoke(cli, ["chat"])

        assert result.exit_code == 0
        mock_chat.assert_called_once()


def _make_mock_client():
    """Create a mock HaikuRAG client."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def _make_app(db_path: Path, mock_client: AsyncMock | None = None):
    """Create a ChatApp with mocked HaikuRAG."""
    from haiku.rag.chat.app import ChatApp

    if mock_client is None:
        mock_client = _make_mock_client()

    skill = MagicMock()
    skill.state_type = None
    skill.state_namespace = None
    skill.tools = []
    skill.toolsets = []
    skill.resources = []
    skill.metadata = MagicMock()
    skill.metadata.name = "rag"
    skill.metadata.description = "RAG skill"

    return ChatApp(
        db_path=db_path,
        skill=skill,
        read_only=True,
    ), mock_client


@pytest.mark.asyncio
async def test_chat_app_has_required_widgets(temp_db_path: Path):
    """Test that ChatApp has the required widgets: ChatHistory, Input."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            chat_history = app.query_one(ChatHistory)
            assert chat_history is not None

            from textual.widgets import Input

            chat_input = app.query_one(Input)
            assert chat_input is not None


@pytest.mark.asyncio
async def test_chat_app_quit_binding(temp_db_path: Path):
    """Test that pressing ctrl+q quits the app."""
    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test() as pilot:
            assert app.is_running
            await pilot.press("ctrl+q")
            assert not app.is_running


@pytest.mark.asyncio
async def test_chat_history_can_add_message(temp_db_path: Path):
    """Test that ChatHistory can display messages."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            chat_history = app.query_one(ChatHistory)

            await chat_history.add_message("user", "Hello, how are you?")
            assert len(chat_history.messages) == 1
            assert chat_history.messages[0] == ("user", "Hello, how are you?")

            await chat_history.add_message("assistant", "I'm doing well, thank you!")
            assert len(chat_history.messages) == 2


@pytest.mark.asyncio
async def test_chat_history_can_add_tool_calls(temp_db_path: Path):
    """Test that ChatHistory can display inline tool calls."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory, ToolCallWidget

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            chat_history = app.query_one(ChatHistory)

            tool_widget = await chat_history.add_tool_call(
                "tool-1", "search", {"query": "test"}
            )
            assert isinstance(tool_widget, ToolCallWidget)
            assert tool_widget._completed is False

            chat_history.mark_tool_complete("tool-1")
            assert tool_widget._completed is True


@pytest.mark.asyncio
async def test_chat_history_can_add_citations(temp_db_path: Path):
    """Test that ChatHistory can display inline citations."""
    from haiku.rag.agents.research.models import Citation
    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test():
            chat_history = app.query_one(ChatHistory)

            test_citations = [
                Citation(
                    index=1,
                    document_id="doc1",
                    chunk_id="chunk1",
                    document_uri="file:///test/doc1.pdf",
                    document_title="Test Document 1",
                    page_numbers=[1, 2],
                    headings=["Section 1"],
                    content="This is some test content from doc 1",
                ),
                Citation(
                    index=2,
                    document_id="doc2",
                    chunk_id="chunk2",
                    document_uri="file:///test/doc2.pdf",
                    document_title="Test Document 2",
                    page_numbers=[5],
                    headings=["Section 2", "Subsection"],
                    content="This is test content from doc 2",
                ),
            ]

            await chat_history.add_citations(test_citations)

            citation_widgets = chat_history.query(CitationWidget)
            assert len(list(citation_widgets)) == 2


@pytest.mark.asyncio
async def test_chat_history_thinking_indicator(temp_db_path: Path):
    """Test that ChatHistory can show and hide thinking indicator."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory, ThinkingWidget

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test() as pilot:
            chat_history = app.query_one(ChatHistory)

            await chat_history.show_thinking()
            thinking = chat_history.query(ThinkingWidget)
            assert len(list(thinking)) == 1

            chat_history.hide_thinking()
            await pilot.pause()
            thinking = chat_history.query(ThinkingWidget)
            assert len(list(thinking)) == 0


@pytest.mark.asyncio
async def test_clear_chat_resets_state(temp_db_path: Path):
    """Test that clearing chat resets state and messages."""
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test() as pilot:
            chat_history = app.query_one(ChatHistory)

            await chat_history.add_message("user", "Hello")
            await chat_history.add_message("assistant", "Hi there")
            assert len(chat_history.messages) == 2

            await app.action_clear_chat()
            await pilot.pause()

            assert len(chat_history.messages) == 0


@pytest.mark.asyncio
async def test_citation_expand_collapse_with_enter(temp_db_path: Path):
    """Test that pressing Enter on a focused citation toggles expand/collapse."""
    from haiku.rag.agents.research.models import Citation
    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget

    app, mock_client = _make_app(temp_db_path)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        async with app.run_test() as pilot:
            chat_history = app.query_one(ChatHistory)

            test_citation = Citation(
                index=1,
                document_id="doc1",
                chunk_id="chunk1",
                document_uri="file:///test/doc1.pdf",
                document_title="Test Document",
                page_numbers=[1],
                content="Test content",
            )
            await chat_history.add_citations([test_citation])

            citation_widget = chat_history.query_one(CitationWidget)
            assert citation_widget.collapsed is True

            citation_widget.focus()
            await pilot.pause()

            await pilot.press("enter")
            await pilot.pause()
            assert citation_widget.collapsed is False

            await pilot.press("enter")
            await pilot.pause()
            assert citation_widget.collapsed is True

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


@pytest.mark.asyncio
async def test_chat_app_has_required_widgets(temp_db_path: Path):
    """Test that ChatApp has the required widgets: ChatHistory, Input."""
    from haiku.rag.chat.app import ChatApp
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test():
            chat_history = app.query_one(ChatHistory)
            assert chat_history is not None

            from textual.widgets import Input

            chat_input = app.query_one(Input)
            assert chat_input is not None


@pytest.mark.asyncio
async def test_chat_app_quit_binding(temp_db_path: Path):
    """Test that pressing ctrl+q quits the app."""
    from haiku.rag.chat.app import ChatApp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test() as pilot:
            # App should be running
            assert app.is_running

            # Press ctrl+q to quit
            await pilot.press("ctrl+q")

            # App should have exited
            assert not app.is_running


@pytest.mark.asyncio
async def test_chat_history_can_add_message(temp_db_path: Path):
    """Test that ChatHistory can display messages."""
    from haiku.rag.chat.app import ChatApp
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test():
            chat_history = app.query_one(ChatHistory)

            # Add a user message
            await chat_history.add_message("user", "Hello, how are you?")
            assert len(chat_history.messages) == 1
            assert chat_history.messages[0] == ("user", "Hello, how are you?")

            # Add an assistant message
            await chat_history.add_message("assistant", "I'm doing well, thank you!")
            assert len(chat_history.messages) == 2


@pytest.mark.asyncio
async def test_chat_app_calls_agent_on_submit(temp_db_path: Path):
    """Test that submitting a message triggers agent invocation."""
    from contextlib import asynccontextmanager

    from haiku.rag.chat.app import ChatApp
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Create a mock stream result
    mock_result = MagicMock()
    mock_result.output = "This is the agent's response."

    # Mock stream that returns the result
    mock_stream = MagicMock()
    mock_stream.get_result = AsyncMock(return_value=mock_result)

    async def mock_stream_text():
        yield "This is the agent's response."

    mock_stream.stream_text = mock_stream_text

    @asynccontextmanager
    async def mock_run_stream(*args, **kwargs):
        yield mock_stream

    # Create a mock agent that returns a fixed response
    mock_agent = MagicMock()
    mock_agent.run_stream = mock_run_stream

    with (
        patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client),
        patch("haiku.rag.chat.app.create_chat_agent", return_value=mock_agent),
    ):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test() as pilot:
            # Type a message in the input
            await pilot.press("H", "e", "l", "l", "o")
            await pilot.press("enter")

            # Give the app time to process
            await pilot.pause()

            # Verify the response was added to chat history
            chat_history = app.query_one(ChatHistory)
            assert len(chat_history.messages) >= 2  # User message + agent response


@pytest.mark.asyncio
async def test_chat_history_can_add_tool_calls(temp_db_path: Path):
    """Test that ChatHistory can display inline tool calls."""
    from haiku.rag.chat.app import ChatApp
    from haiku.rag.chat.widgets.chat_history import ChatHistory, ToolCallWidget

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test():
            chat_history = app.query_one(ChatHistory)

            # Add a tool call
            tool_widget = await chat_history.add_tool_call("search", {"query": "test"})
            assert isinstance(tool_widget, ToolCallWidget)
            assert tool_widget._complete is False

            # Mark it complete
            chat_history.mark_tool_complete(tool_widget)
            assert tool_widget._complete is True


@pytest.mark.asyncio
async def test_chat_history_can_add_citations(temp_db_path: Path):
    """Test that ChatHistory can display inline citations."""
    from haiku.rag.agents.chat.state import CitationInfo
    from haiku.rag.chat.app import ChatApp
    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test():
            chat_history = app.query_one(ChatHistory)

            test_citations = [
                CitationInfo(
                    index=1,
                    document_id="doc1",
                    chunk_id="chunk1",
                    document_uri="file:///test/doc1.pdf",
                    document_title="Test Document 1",
                    page_numbers=[1, 2],
                    headings=["Section 1"],
                    content="This is some test content from doc 1",
                ),
                CitationInfo(
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

            # Verify citation widgets were added
            citation_widgets = chat_history.query(CitationWidget)
            assert len(list(citation_widgets)) == 2


@pytest.mark.asyncio
async def test_chat_history_thinking_indicator(temp_db_path: Path):
    """Test that ChatHistory can show and hide thinking indicator."""
    from haiku.rag.chat.app import ChatApp
    from haiku.rag.chat.widgets.chat_history import ChatHistory, ThinkingWidget

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test() as pilot:
            chat_history = app.query_one(ChatHistory)

            # Show thinking indicator
            await chat_history.show_thinking()
            thinking = chat_history.query(ThinkingWidget)
            assert len(list(thinking)) == 1

            # Hide thinking indicator (remove is deferred, so pause)
            chat_history.hide_thinking()
            await pilot.pause()
            thinking = chat_history.query(ThinkingWidget)
            assert len(list(thinking)) == 0


@pytest.mark.asyncio
async def test_clear_chat_resets_session(temp_db_path: Path):
    """Test that clearing chat resets the session state."""
    from haiku.rag.chat.app import ChatApp
    from haiku.rag.chat.widgets.chat_history import ChatHistory

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test() as pilot:
            chat_history = app.query_one(ChatHistory)

            # Add some messages
            await chat_history.add_message("user", "Hello")
            await chat_history.add_message("assistant", "Hi there")
            assert len(chat_history.messages) == 2

            # Store original session ID
            assert app.session_state is not None
            original_session_id = app.session_state.session_id

            # Clear chat
            await pilot.press("ctrl+l")
            await pilot.pause()

            # Verify messages cleared
            assert len(chat_history.messages) == 0

            # Verify session state reset (new session ID)
            assert app.session_state is not None
            assert app.session_state.session_id != original_session_id


@pytest.mark.asyncio
async def test_citation_expand_collapse_with_enter(temp_db_path: Path):
    """Test that pressing Enter on a focused citation toggles expand/collapse."""
    from haiku.rag.agents.chat.state import CitationInfo
    from haiku.rag.chat.app import ChatApp
    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test() as pilot:
            chat_history = app.query_one(ChatHistory)

            # Add a citation
            test_citation = CitationInfo(
                index=1,
                document_id="doc1",
                chunk_id="chunk1",
                document_uri="file:///test/doc1.pdf",
                document_title="Test Document",
                page_numbers=[1],
                content="Test content",
            )
            await chat_history.add_citations([test_citation])

            # Get the citation widget
            citation_widget = chat_history.query_one(CitationWidget)
            assert citation_widget.collapsed is True

            # Focus the citation
            citation_widget.focus()
            await pilot.pause()

            # Press Enter to expand
            await pilot.press("enter")
            await pilot.pause()
            assert citation_widget.collapsed is False

            # Press Enter again to collapse
            await pilot.press("enter")
            await pilot.pause()
            assert citation_widget.collapsed is True

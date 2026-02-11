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
    from haiku.rag.agents.research.models import Citation
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

            # Clear chat via action (available through command palette)
            await app.action_clear_chat()
            await pilot.pause()

            # Verify messages cleared
            assert len(chat_history.messages) == 0

            # Verify session state reset
            assert app.session_state is not None
            assert app.session_state.qa_history == []
            assert app.session_state.citations == []


@pytest.mark.asyncio
async def test_handle_stream_event_extracts_citations_from_state_snapshot(
    temp_db_path: Path,
):
    """Test that _handle_stream_event extracts citations from STATE_SNAPSHOT events."""
    from ag_ui.core import EventType, StateSnapshotEvent
    from pydantic_ai import FunctionToolResultEvent
    from pydantic_ai.messages import ToolReturnPart

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY
    from haiku.rag.chat.app import ChatApp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test():
            # Create a STATE_SNAPSHOT event with citations
            snapshot_event = StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot={
                    AGUI_STATE_KEY: {
                        "citations": [
                            {
                                "index": 1,
                                "document_id": "doc1",
                                "chunk_id": "chunk1",
                                "document_uri": "test.pdf",
                                "document_title": "Test Doc",
                                "content": "Test content",
                            }
                        ],
                        "qa_history": [],
                        "citation_registry": {"chunk1": 1},
                    }
                },
            )

            # Create a tool result with the snapshot in metadata
            tool_return = ToolReturnPart(
                tool_name="search",
                content="Found results",
                tool_call_id="test-call-1",
                metadata=[snapshot_event],
            )

            event = FunctionToolResultEvent(result=tool_return)

            # Handle the event
            await app._handle_stream_event(event)

            # Session state should be synced
            assert len(app.session_state.citations) == 1
            assert app.session_state.citations[0].chunk_id == "chunk1"
            assert app.session_state.citation_registry == {"chunk1": 1}


@pytest.mark.asyncio
async def test_handle_stream_event_extracts_citations_from_state_delta(
    temp_db_path: Path,
):
    """Test that _handle_stream_event extracts citations from STATE_DELTA events.

    This test demonstrates that state deltas need to be applied to extract citations.
    After the first STATE_SNAPSHOT, subsequent tool calls emit STATE_DELTA events
    containing JSON Patch operations.
    """
    from ag_ui.core import EventType, StateDeltaEvent, StateSnapshotEvent
    from pydantic_ai import FunctionToolResultEvent
    from pydantic_ai.messages import ToolReturnPart

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY
    from haiku.rag.chat.app import ChatApp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test():
            # First, handle a STATE_SNAPSHOT to establish initial state
            initial_snapshot = StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot={
                    AGUI_STATE_KEY: {
                        "citations": [],
                        "qa_history": [],
                        "citation_registry": {},
                    }
                },
            )
            tool_return1 = ToolReturnPart(
                tool_name="search",
                content="Initial search",
                tool_call_id="test-call-1",
                metadata=[initial_snapshot],
            )
            event1 = FunctionToolResultEvent(result=tool_return1)
            await app._handle_stream_event(event1)
            assert len(app.session_state.citations) == 0

            # Now handle a STATE_DELTA event that adds citations
            delta_event = StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "replace",
                        "path": f"/{AGUI_STATE_KEY}/citations",
                        "value": [
                            {
                                "index": 1,
                                "document_id": "doc1",
                                "chunk_id": "chunk1",
                                "document_uri": "test.pdf",
                                "document_title": "Test Doc",
                                "content": "Test content from delta",
                            }
                        ],
                    },
                    {
                        "op": "add",
                        "path": f"/{AGUI_STATE_KEY}/citation_registry/chunk1",
                        "value": 1,
                    },
                ],
            )
            tool_return2 = ToolReturnPart(
                tool_name="ask",
                content="Answer with citations",
                tool_call_id="test-call-2",
                metadata=[delta_event],
            )
            event2 = FunctionToolResultEvent(result=tool_return2)

            # Handle the delta event
            await app._handle_stream_event(event2)

            # Session state should be synced
            assert len(app.session_state.citations) == 1
            assert app.session_state.citations[0].chunk_id == "chunk1"
            assert app.session_state.citations[0].content == "Test content from delta"


@pytest.mark.asyncio
async def test_handle_stream_event_delta_with_preinitialized_state(
    temp_db_path: Path,
):
    """Test delta handling when _agui_state_snapshot is pre-initialized.

    This is the actual TUI scenario: session_state exists from the start,
    so the agent emits deltas (not snapshots) even on the first tool call.
    The TUI pre-initializes _agui_state_snapshot from session_state.
    """
    from ag_ui.core import EventType, StateDeltaEvent
    from pydantic_ai import FunctionToolResultEvent
    from pydantic_ai.messages import ToolReturnPart

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY
    from haiku.rag.chat.app import ChatApp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test():
            # Pre-initialize _agui_state_snapshot (simulating what _run_agent does)
            app._agui_state_snapshot = {
                AGUI_STATE_KEY: {
                    "citations": [],
                    "qa_history": [],
                    "citation_registry": {},
                    "document_filter": [],
                    "initial_context": None,
                    "session_context": None,
                }
            }

            # Now handle a STATE_DELTA event directly (no prior snapshot event)
            delta_event = StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "replace",
                        "path": f"/{AGUI_STATE_KEY}/citations",
                        "value": [
                            {
                                "index": 1,
                                "document_id": "doc1",
                                "chunk_id": "chunk1",
                                "document_uri": "test.pdf",
                                "document_title": "Test Doc",
                                "content": "Content from first delta",
                            }
                        ],
                    },
                    {
                        "op": "add",
                        "path": f"/{AGUI_STATE_KEY}/citation_registry/chunk1",
                        "value": 1,
                    },
                ],
            )
            tool_return = ToolReturnPart(
                tool_name="ask",
                content="Answer with citations",
                tool_call_id="test-call-1",
                metadata=[delta_event],
            )
            event = FunctionToolResultEvent(result=tool_return)

            # Handle the delta event
            await app._handle_stream_event(event)

            # Session state should be synced
            assert len(app.session_state.citations) == 1
            assert app.session_state.citations[0].chunk_id == "chunk1"
            assert app.session_state.citations[0].content == "Content from first delta"


@pytest.mark.asyncio
async def test_handle_stream_event_syncs_session_context(temp_db_path: Path):
    """Test that _handle_stream_event syncs session_context to session_state."""
    from ag_ui.core import EventType, StateDeltaEvent
    from pydantic_ai import FunctionToolResultEvent
    from pydantic_ai.messages import ToolReturnPart

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY
    from haiku.rag.chat.app import ChatApp

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("haiku.rag.chat.app.HaikuRAG", return_value=mock_client):
        app = ChatApp(temp_db_path, read_only=True)

        async with app.run_test():
            # Pre-initialize state
            app._agui_state_snapshot = {
                AGUI_STATE_KEY: {
                    "citations": [],
                    "qa_history": [],
                    "citation_registry": {},
                    "document_filter": [],
                    "initial_context": None,
                    "session_context": None,
                }
            }

            # Verify session_context starts as None
            assert app.session_state.session_context is None

            # Handle a delta that adds session_context
            delta_event = StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "replace",
                        "path": f"/{AGUI_STATE_KEY}/session_context",
                        "value": {
                            "summary": "User asked about Python async patterns.",
                            "last_updated": "2025-01-15T10:30:00",
                        },
                    },
                ],
            )
            tool_return = ToolReturnPart(
                tool_name="ask",
                content="Answer about async",
                tool_call_id="test-call-1",
                metadata=[delta_event],
            )
            event = FunctionToolResultEvent(result=tool_return)

            await app._handle_stream_event(event)

            # Session context should be synced to session_state
            assert app.session_state.session_context is not None
            assert (
                app.session_state.session_context.summary
                == "User asked about Python async patterns."
            )


@pytest.mark.asyncio
async def test_citation_expand_collapse_with_enter(temp_db_path: Path):
    """Test that pressing Enter on a focused citation toggles expand/collapse."""
    from haiku.rag.agents.research.models import Citation
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

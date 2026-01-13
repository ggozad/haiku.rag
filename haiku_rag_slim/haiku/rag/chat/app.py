# pyright: reportPossiblyUnboundVariable=false
import asyncio
import uuid
from collections.abc import AsyncIterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ag_ui.core import EventType
from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RunContext,
)
from pydantic_ai.messages import ModelMessage

from haiku.rag.agents.chat.agent import create_chat_agent
from haiku.rag.agents.chat.state import ChatDeps, ChatSessionState, CitationInfo
from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config

if TYPE_CHECKING:
    from textual.app import ComposeResult

try:
    import logfire

    logfire.configure(console=False)
except ImportError:
    pass

try:
    import textual_image.widget  # noqa: F401 - import early for renderer detection
    from textual.app import App
    from textual.binding import Binding
    from textual.widgets import Footer, Header, Input
    from textual.worker import Worker

    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = object  # type: ignore


class ChatApp(App):  # type: ignore[misc]
    """Textual TUI for conversational RAG."""

    TITLE = "haiku.rag Chat"

    CSS = """
    Screen {
        layout: grid;
        grid-size: 1 2;
        grid-rows: 1fr auto;
        background: $surface;
    }

    #chat-history {
        height: 100%;
    }

    Header {
        background: $primary;
    }

    Footer {
        background: $surface-darken-1;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("ctrl+g", "show_visual", "Visual", show=True),
        Binding("ctrl+i", "show_info", "Info", show=True),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    def __init__(
        self, db_path: Path, read_only: bool = False, before: datetime | None = None
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self.read_only = read_only
        self.before = before
        self.client: HaikuRAG | None = None
        self.config = get_config()
        self.agent: Agent[ChatDeps, str] | None = None
        self.session_state: ChatSessionState | None = None
        self._is_processing = False
        self._tool_call_widgets: dict[str, Any] = {}
        self._last_citations: list[CitationInfo] = []
        self._selected_citation_idx: int | None = None
        self._current_worker: Worker[None] | None = None
        self._message_history: list[ModelMessage] = []

    def compose(self) -> "ComposeResult":
        """Compose the UI layout."""
        yield Header()
        yield ChatHistory(id="chat-history")
        yield Input(placeholder="Ask a question...", id="chat-input")
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.client = HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        )
        await self.client.__aenter__()

        # Create agent and session state
        self.agent = create_chat_agent(self.config)
        self.session_state = ChatSessionState(session_id=str(uuid.uuid4()))

        # Focus the input field
        self.query_one(Input).focus()

    async def on_unmount(self) -> None:
        """Clean up when unmounting."""
        if self.client:
            await self.client.__aexit__(None, None, None)

    async def _handle_stream_event(self, event: AgentStreamEvent) -> None:
        """Handle streaming events from the agent."""
        chat_history = self.query_one(ChatHistory)

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            tool_call_id = event.part.tool_call_id or str(uuid.uuid4())
            args = event.part.args_as_dict()
            widget = await chat_history.add_tool_call(tool_name, args)
            self._tool_call_widgets[tool_call_id] = widget

        elif isinstance(event, FunctionToolResultEvent):
            tool_call_id = event.tool_call_id
            if tool_call_id and tool_call_id in self._tool_call_widgets:
                widget = self._tool_call_widgets[tool_call_id]
                chat_history.mark_tool_complete(widget)

            # Extract citations from StateSnapshotEvent in tool metadata
            result = getattr(event, "result", None)
            metadata = getattr(result, "metadata", None) if result else None
            if metadata:
                for meta_event in metadata:
                    if (
                        hasattr(meta_event, "type")
                        and meta_event.type == EventType.STATE_SNAPSHOT
                    ):
                        snapshot = getattr(meta_event, "snapshot", {})
                        if "citations" in snapshot:
                            self._last_citations = [
                                CitationInfo(**c) for c in snapshot["citations"]
                            ]

    async def _event_stream_handler(
        self,
        _ctx: RunContext[ChatDeps],
        event_stream: AsyncIterable[AgentStreamEvent],
    ) -> None:
        """Handle streaming events from the agent."""
        async for event in event_stream:
            await self._handle_stream_event(event)
            # Yield to event loop to keep UI responsive
            await asyncio.sleep(0)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_message = event.value.strip()
        if not user_message or self._is_processing:
            return

        if not self.client or not self.agent:
            return

        # Clear the input
        event.input.clear()

        # Add user message to history
        chat_history = self.query_one(ChatHistory)
        await chat_history.add_message("user", user_message)

        # Clear for new query
        self._tool_call_widgets.clear()
        self._last_citations.clear()
        self._selected_citation_idx = None

        # Run agent in a worker to keep UI responsive
        self._is_processing = True
        self.query_one(Input).disabled = True
        self._current_worker = self.run_worker(
            self._run_agent(user_message), exclusive=True
        )

    async def _run_agent(self, user_message: str) -> None:
        """Run the agent in a background worker."""
        if not self.client or not self.agent:
            return

        chat_history = self.query_one(ChatHistory)

        # Show thinking indicator
        await chat_history.show_thinking()

        try:
            deps = ChatDeps(
                client=self.client,
                config=self.config,
                session_state=self.session_state,
            )

            async with self.agent.run_stream(
                user_message,
                deps=deps,
                message_history=self._message_history,
                event_stream_handler=self._event_stream_handler,
            ) as stream:
                # Hide thinking when we start getting content
                chat_history.hide_thinking()

                # Create assistant message for streaming
                assistant_msg = await chat_history.add_message("assistant", "")

                # Stream text updates
                async for text in stream.stream_text():
                    assistant_msg.update_content(text)
                    chat_history.scroll_end(animate=False)
                    # Yield to event loop to keep UI responsive
                    await asyncio.sleep(0)

                # Update message history with this conversation
                self._message_history = stream.all_messages()

                # Add citations captured from tool metadata
                if self._last_citations:
                    await chat_history.add_citations(self._last_citations)

        except asyncio.CancelledError:
            chat_history.hide_thinking()
            await chat_history.add_message("assistant", "*Cancelled*")
        except Exception as e:
            chat_history.hide_thinking()
            await chat_history.add_message("assistant", f"Error: {e}")
        finally:
            self._is_processing = False
            self._current_worker = None
            chat_input = self.query_one(Input)
            chat_input.disabled = False
            chat_input.focus()

    async def action_clear_chat(self) -> None:
        """Clear the chat history and reset session."""
        chat_history = self.query_one(ChatHistory)
        await chat_history.clear_messages()
        self._last_citations.clear()
        self._selected_citation_idx = None
        self._message_history.clear()
        # Reset session state for fresh conversation
        self.session_state = ChatSessionState(session_id=str(uuid.uuid4()))

    def action_focus_input(self) -> None:
        """Focus the input field, or cancel if processing."""
        if self._is_processing and self._current_worker:
            self._current_worker.cancel()
        self.query_one(Input).focus()

    def _clear_citation_selection(self) -> None:
        """Clear citation selection."""
        chat_history = self.query_one(ChatHistory)
        for widget in chat_history.query(CitationWidget):
            widget.remove_class("selected")
        self._selected_citation_idx = None

    def on_descendant_focus(self, _event: object) -> None:
        """Clear citation selection when input is focused."""
        if isinstance(self.focused, Input):
            self._clear_citation_selection()

    async def action_show_visual(self) -> None:
        """Show visual grounding for the selected citation."""
        if not self.client or not self._last_citations:
            return

        idx = (
            self._selected_citation_idx
            if self._selected_citation_idx is not None
            else 0
        )
        citation = self._last_citations[idx]
        chunk = await self.client.chunk_repository.get_by_id(citation.chunk_id)
        if not chunk:
            return

        from haiku.rag.inspector.widgets.visual_modal import VisualGroundingModal

        await self.push_screen(VisualGroundingModal(chunk=chunk, client=self.client))

    async def action_show_info(self) -> None:
        """Show database info modal."""
        if not self.client:
            return

        from haiku.rag.inspector.widgets.info_modal import InfoModal

        await self.push_screen(InfoModal(self.client, self.db_path))

    def on_citation_widget_selected(self, event: CitationWidget.Selected) -> None:
        """Handle citation selection."""
        chat_history = self.query_one(ChatHistory)

        # Remove selected class from all citations
        for widget in chat_history.query(CitationWidget):
            widget.remove_class("selected")

        # Add selected class to the newly selected citation
        citation_widgets = list(chat_history.query(CitationWidget))
        if 0 <= event.citation_index < len(citation_widgets):
            citation_widgets[event.citation_index].add_class("selected")
            self._selected_citation_idx = event.citation_index

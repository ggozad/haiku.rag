# pyright: reportPossiblyUnboundVariable=false
import asyncio
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.skills.rag import AGENT_PREAMBLE, RAGState
from haiku.skills.agent import SkillToolset
from haiku.skills.models import Skill

if TYPE_CHECKING:
    from textual.app import ComposeResult

try:
    import logfire

    logfire.configure(send_to_logfire="if-token-present", console=False)
    logfire.instrument_pydantic_ai()
except ImportError:
    pass

try:
    import textual_image.widget  # noqa: F401 - import early for renderer detection
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
        EventType,
        RunAgentInput,
        StateDeltaEvent,
        TextMessageContentEvent,
        ToolCallEndEvent,
        ToolCallStartEvent,
        UserMessage,
    )
    from jsonpatch import JsonPatch
    from pydantic_ai import Agent
    from pydantic_ai.ag_ui import AGUIAdapter
    from textual.app import App, SystemCommand
    from textual.binding import Binding
    from textual.widgets import Footer, Header, Input
    from textual.worker import Worker

    from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = object  # type: ignore
    SystemCommand = object  # type: ignore


RAG_STATE_NAMESPACE = "rag"


class ChatApp(App):
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
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    def __init__(
        self,
        db_path: Path,
        skill: Skill,
        read_only: bool = False,
        before: datetime | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self._skill = skill
        self.read_only = read_only
        self.before = before
        self._model = model or "openai:gpt-4o"
        self.client: HaikuRAG | None = None
        self.config = get_config()
        self._toolset: SkillToolset | None = None
        self._agent: Agent[None, str] | None = None
        self._messages: list[Any] = []
        self._state: dict[str, Any] = {}
        self._is_processing = False
        self._current_worker: Worker[None] | None = None
        self._document_filter: list[str] = []

    def compose(self) -> "ComposeResult":
        """Compose the UI layout."""
        yield Header()
        yield ChatHistory(id="chat-history")
        yield Input(placeholder="Ask a question...", id="chat-input")
        yield Footer()

    def get_system_commands(self, screen: Any) -> Iterable[SystemCommand]:
        """Add commands to the command palette."""
        yield from super().get_system_commands(screen)
        yield SystemCommand(
            "Clear chat",
            "Clear the chat history and reset session",
            self.action_clear_chat,
        )
        yield SystemCommand(
            "Filter documents",
            "Select documents to filter searches",
            self.action_show_filter,
        )
        yield SystemCommand(
            "Show visual grounding",
            "Show visual grounding for selected citation",
            self.action_show_visual,
        )
        yield SystemCommand(
            "Database info",
            "Show database information",
            self.action_show_info,
        )
        yield SystemCommand(
            "View state",
            "Show the current session state",
            self.action_view_state,
        )

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.client = HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        )
        await self.client.__aenter__()

        self._toolset = SkillToolset(skills=[self._skill])
        self._agent = Agent(
            self._model,
            instructions=AGENT_PREAMBLE + self._toolset.system_prompt,
            toolsets=[self._toolset],
        )
        self._state = self._toolset.build_state_snapshot()

        self.query_one(Input).focus()

    async def on_unmount(self) -> None:
        """Clean up when unmounting."""
        if self.client:
            await self.client.__aexit__(None, None, None)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_message = event.value.strip()
        if not user_message or self._is_processing:
            return

        event.input.clear()

        chat_history = self.query_one(ChatHistory)
        await chat_history.add_message("user", user_message)

        self._messages.append(
            UserMessage(
                id=str(uuid.uuid4()),
                role="user",
                content=user_message,
            )
        )

        self._is_processing = True
        self.query_one(Input).disabled = True
        self._current_worker = self.run_worker(
            self._run_agent(user_message), exclusive=True
        )

    async def _run_agent(self, user_message: str) -> None:
        """Run the agent in a background worker."""
        if not self._agent or not self._toolset:
            return

        chat_history = self.query_one(ChatHistory)
        await chat_history.show_thinking()

        run_input = RunAgentInput(
            thread_id="tui",
            run_id=str(uuid.uuid4()),
            messages=self._messages,
            state=self._state,
            tools=[],
            context=[],
            forwarded_props={},
        )

        adapter = AGUIAdapter(agent=self._agent, run_input=run_input)

        message = None
        accumulated_text = ""

        try:
            async for event in adapter.run_stream():
                if not isinstance(event, BaseEvent):
                    continue
                if event.type == EventType.TEXT_MESSAGE_START:
                    chat_history.hide_thinking()
                    message = await chat_history.add_message("assistant")
                    accumulated_text = ""
                elif event.type == EventType.TEXT_MESSAGE_CONTENT:
                    assert isinstance(event, TextMessageContentEvent)
                    accumulated_text += event.delta
                    if message:
                        message.update_content(accumulated_text)
                        chat_history.scroll_end(animate=False)
                elif event.type == EventType.TEXT_MESSAGE_END:
                    self._messages.append(
                        AssistantMessage(
                            id=str(uuid.uuid4()),
                            role="assistant",
                            content=accumulated_text,
                        )
                    )
                    # Show citations from RAG state
                    await self._show_citations(chat_history)
                elif event.type == EventType.TOOL_CALL_START:
                    assert isinstance(event, ToolCallStartEvent)
                    chat_history.hide_thinking()
                    await chat_history.add_tool_call(
                        event.tool_call_id, event.tool_call_name
                    )
                    await chat_history.show_thinking("Executing tasks...")
                elif event.type == EventType.TOOL_CALL_END:
                    assert isinstance(event, ToolCallEndEvent)
                    chat_history.mark_tool_complete(event.tool_call_id)
                elif event.type == EventType.STATE_DELTA:
                    assert isinstance(event, StateDeltaEvent)
                    patch = JsonPatch(event.delta)
                    self._state = patch.apply(self._state)
                    self._toolset.restore_state_snapshot(self._state)
                elif event.type == EventType.STATE_SNAPSHOT:
                    self._state = getattr(event, "snapshot", self._state)
                    self._toolset.restore_state_snapshot(self._state)
                elif event.type == EventType.RUN_FINISHED:
                    chat_history.hide_thinking()
                elif event.type == EventType.RUN_ERROR:
                    chat_history.hide_thinking()
                    error_msg = getattr(event, "message", "Unknown error")
                    await chat_history.add_message("assistant", f"Error: {error_msg}")

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

    async def _show_citations(self, chat_history: "ChatHistory") -> None:
        """Show citations from the RAG state after an agent response."""
        if not self._toolset:
            return
        rag_state = self._toolset.get_namespace(RAG_STATE_NAMESPACE)
        if rag_state is None:
            return
        citations = getattr(rag_state, "citations", [])
        if citations:
            # Show only new citations (since last response)
            await chat_history.add_citations(citations)

    async def action_clear_chat(self) -> None:
        """Clear the chat history and reset session."""
        chat_history = self.query_one(ChatHistory)
        await chat_history.clear_messages()
        self._messages.clear()
        # Reset state
        if self._toolset:
            self._state = self._toolset.build_state_snapshot()

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

    def on_descendant_focus(self, _event: object) -> None:
        """Clear citation selection when chat input is focused."""
        if isinstance(self.focused, Input) and self.focused.id == "chat-input":
            self._clear_citation_selection()

    async def action_show_visual(self) -> None:
        """Show visual grounding for the selected citation."""
        if not self.client:
            return

        chat_history = self.query_one(ChatHistory)
        selected_widgets = list(chat_history.query(CitationWidget).filter(".selected"))
        if not selected_widgets:
            return

        citation = selected_widgets[0].citation
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

    def action_view_state(self) -> None:
        """Show the current session state."""
        from haiku.skills.chat.app import StateScreen

        self.push_screen(StateScreen(self._state))

    def on_citation_widget_selected(self, event: CitationWidget.Selected) -> None:
        """Handle citation selection."""
        chat_history = self.query_one(ChatHistory)

        for widget in chat_history.query(CitationWidget):
            widget.remove_class("selected")

        event.widget.add_class("selected")

    async def action_show_filter(self) -> None:
        """Show document filter modal."""
        if not self.client:
            return

        from haiku.rag.chat.widgets.document_filter_modal import DocumentFilterModal

        await self.push_screen(
            DocumentFilterModal(
                client=self.client,
                selected=self._document_filter,
            )
        )

    def on_document_filter_modal_filter_changed(self, event: Any) -> None:
        """Handle document filter changes from modal."""
        from haiku.rag.tools.filters import build_multi_document_filter

        self._document_filter = event.selected

        if self._toolset:
            rag_state = self._toolset.get_namespace(RAG_STATE_NAMESPACE)
            if isinstance(rag_state, RAGState):
                rag_state.document_filter = build_multi_document_filter(
                    self._document_filter
                )
                self._state = self._toolset.build_state_snapshot()

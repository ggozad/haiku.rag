import asyncio
import uuid
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import textual_image.widget  # noqa: F401 - import early for renderer detection
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.run import AgentRunResultEvent
from textual.app import App, SystemCommand
from textual.binding import Binding
from textual.widgets import Footer, Header
from textual.worker import Worker

from haiku.rag.capabilities._base import RAGCapabilityBase
from haiku.rag.capabilities.analysis import AnalysisState
from haiku.rag.capabilities.compaction import create_capability as create_compaction
from haiku.rag.capabilities.rag import AGENT_PREAMBLE, RAGState
from haiku.rag.chat.widgets.chat_history import ChatHistory, CitationWidget
from haiku.rag.chat.widgets.image_select import ImageAdded
from haiku.rag.chat.widgets.prompt import (
    FlexibleInput,
    PostableTextArea,
    build_user_prompt,
)
from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config
from haiku.rag.store.models.chunk import qualified_id
from haiku.rag.telemetry import configure as configure_telemetry

configure_telemetry(service_name="haiku-rag")

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from haiku.rag.client.scope import DatabaseScope


RAG_STATE_NAMESPACE = "rag"
ANALYSIS_STATE_NAMESPACE = "analysis"


@dataclass
class ChatDeps:
    state: dict[str, Any] = field(default_factory=dict)


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
        capabilities: Sequence[RAGCapabilityBase[Any]],
        scope: "DatabaseScope",
        read_only: bool = False,
        model: str | None = None,
    ) -> None:
        super().__init__()
        self.scope = scope
        self._capabilities = capabilities
        self.read_only = read_only
        self._model = model
        self.client: HaikuRAG | None = None
        self.config = get_config()
        self._agent: Agent[ChatDeps, str] | None = None
        self._messages: list[Any] = []
        self._state: dict[str, Any] = {}
        self._is_processing = False
        self._current_worker: Worker[None] | None = None
        self._document_filter: list[tuple[str | None, str]] = []
        self._images: list[bytes] = []
        # Stable per-launch id for multi-turn model and telemetry correlation.
        self._conversation_id = str(uuid.uuid4())

    def compose(self) -> "ComposeResult":
        """Compose the UI layout."""
        yield Header()
        yield ChatHistory(id="chat-history")
        yield FlexibleInput(id="chat-input")
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

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        client = HaikuRAG._covering(self.scope, self.config, read_only=self.read_only)
        # Assign only after a successful open: on_unmount must not tear down
        # a client whose __aenter__ failed.
        await client.__aenter__()
        self.client = client
        # Lent to the capabilities, with the scope it covers: one connection
        # per database however many capabilities read it, and the analysis
        # sandbox is built over the same selection.
        for capability in self._capabilities:
            capability.borrowed_rag = client
            capability.scope = self.scope

        self._agent = Agent(
            self._model,
            deps_type=ChatDeps,
            instructions=AGENT_PREAMBLE,
            # A chat is multi-turn by definition, so earlier questions are reduced
            # to the evidence they cited rather than carried whole.
            capabilities=[*self._capabilities, create_compaction()],
        )
        self._state = {}
        for capability in self._capabilities:
            self._state[capability.state_namespace] = (
                capability.state_type().model_dump(mode="json")
            )

        self.query_one(FlexibleInput).focus()

    async def on_unmount(self) -> None:
        """Clean up when unmounting."""
        if self.client:
            await self.client.__aexit__(None, None, None)

    async def on_flexible_input_submitted(self, event: FlexibleInput.Submitted) -> None:
        """Handle user input submission."""
        user_message = event.value.strip()
        if not user_message or self._is_processing:
            return

        event.input.clear()

        chat_history = self.query_one(ChatHistory)
        await chat_history.add_message("user", user_message)

        user_prompt = build_user_prompt(user_message, self._images)
        self._images = []

        self._is_processing = True
        self.query_one(FlexibleInput).disabled = True
        self._current_worker = self.run_worker(
            self._run_agent(user_prompt), exclusive=True
        )

    def on_image_added(self, event: ImageAdded) -> None:
        """Attach a picked image and insert its token into the prompt."""
        self._images.append(event.data)
        prompt = self.query_one(FlexibleInput)
        prompt.insert_at_cursor(f"[Image #{len(self._images)}]")
        prompt.focus()
        self.notify(f"Attached {event.path.name}")

    async def _run_agent(self, user_prompt: str | list[str | BinaryContent]) -> None:
        """Run the agent in a background worker."""
        if not self._agent:
            return

        chat_history = self.query_one(ChatHistory)
        await chat_history.show_thinking()

        message = None
        # The run gets a copy: state and message history have to advance together.
        # A cancelled or failed run discards its messages, and state that advanced
        # anyway would leave the next question deriving its identity from a shorter
        # history than the evidence already recorded — refused as non-append-only,
        # with the conversation stuck until it is cleared.
        deps = ChatDeps(state=deepcopy(self._state))

        try:
            async with self._agent.run_stream_events(
                user_prompt,
                message_history=self._messages,
                conversation_id=self._conversation_id,
                deps=deps,
            ) as stream:
                async for event in stream:
                    if isinstance(event, PartStartEvent) and isinstance(
                        event.part, TextPart
                    ):
                        chat_history.hide_thinking()
                        message = await chat_history.add_message("assistant")
                        if event.part.content:
                            await message.append_delta(event.part.content)
                    elif isinstance(event, PartDeltaEvent) and isinstance(
                        event.delta, TextPartDelta
                    ):
                        if message:
                            await message.append_delta(event.delta.content_delta)
                            chat_history.scroll_end(animate=False)
                    elif isinstance(event, PartEndEvent) and isinstance(
                        event.part, TextPart
                    ):
                        if message:
                            await message.finish_stream()
                    elif isinstance(event, FunctionToolCallEvent):
                        part = event.part
                        chat_history.hide_thinking()
                        await chat_history.add_tool_call(
                            part.tool_call_id, part.tool_name
                        )
                        chat_history.update_tool_args(
                            part.tool_call_id, part.args_as_dict()
                        )
                        await chat_history.show_thinking("Executing tasks...")
                    elif isinstance(event, FunctionToolResultEvent):
                        chat_history.mark_tool_complete(event.part.tool_call_id)
                    elif isinstance(event, AgentRunResultEvent):
                        self._messages = event.result.all_messages()
                        self._state = deps.state
                        chat_history.hide_thinking()
                        await self._show_citations_and_programs(chat_history)

        except asyncio.CancelledError:
            chat_history.hide_thinking()
            if message:
                await message.finish_stream()
            await chat_history.add_message("assistant", "*Cancelled*")
        except Exception as e:
            chat_history.hide_thinking()
            if message:
                await message.finish_stream()
            await chat_history.add_message("assistant", f"Error: {e}")
        finally:
            self._is_processing = False
            self._current_worker = None
            chat_input = self.query_one(FlexibleInput)
            chat_input.disabled = False
            chat_input.focus()

    async def _show_citations_and_programs(self, chat_history: "ChatHistory") -> None:
        """Show citations and programs from capability states after a response."""
        citations = []
        for namespace in (RAG_STATE_NAMESPACE, ANALYSIS_STATE_NAMESPACE):
            state_data = self._state.get(namespace)
            if not state_data:
                continue
            state_type = RAGState if namespace == RAG_STATE_NAMESPACE else AnalysisState
            state = state_type.model_validate(state_data)
            for cid in state.citations:
                if cid in state.citation_index:
                    citations.append(state.citation_index[cid])
        if not citations:
            return

        picture_bytes: dict[tuple[str | None, str | None], list[bytes]] = {}
        if self.client is not None:
            for citation in citations:
                refs = list(citation.picture_refs or [])
                if not refs:
                    continue
                # A sourceless citation across a set has no owner to fetch
                # pictures from: it renders with its figure markers alone.
                owner = await self.client.reader_for(citation.source)
                if owner is None:
                    continue
                blobs: list[bytes] = []
                for ref in refs:
                    data = await owner.get_picture_bytes(citation.document_id, ref)
                    if data:
                        blobs.append(data)
                if blobs:
                    picture_bytes[qualified_id(citation.source, citation.chunk_id)] = (
                        blobs
                    )

        await chat_history.add_citations(
            citations,
            picture_bytes=picture_bytes,
            include_collection=self.client is not None and self.client.covers_multiple,
        )

        if analysis_data := self._state.get(ANALYSIS_STATE_NAMESPACE):
            analysis_state = AnalysisState.model_validate(analysis_data)
            successful = [e for e in analysis_state.executions if e.success]
            if successful:
                await chat_history.add_program(successful[-1].code)

    async def action_clear_chat(self) -> None:
        """Clear the chat history and reset session."""
        chat_history = self.query_one(ChatHistory)
        await chat_history.clear_messages()
        self._messages.clear()
        self._state = {
            capability.state_namespace: capability.state_type().model_dump(mode="json")
            for capability in self._capabilities
        }
        # Cleared chat starts a fresh Logfire conversation.
        self._conversation_id = str(uuid.uuid4())

    def action_focus_input(self) -> None:
        """Focus the input field, or cancel if processing."""
        if self._is_processing and self._current_worker:
            self._current_worker.cancel()
        self.query_one(FlexibleInput).focus()

    def _clear_citation_selection(self) -> None:
        """Clear citation selection."""
        chat_history = self.query_one(ChatHistory)
        for widget in chat_history.query(CitationWidget):
            widget.remove_class("selected")

    def on_descendant_focus(self, _event: object) -> None:
        """Clear citation selection when chat input is focused."""
        if isinstance(self.focused, PostableTextArea):
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
        # Chunks, pages and bounding boxes all come from the database holding the
        # cited chunk, which a client covering a set has to be asked for.
        client = await self.client.reader_for(citation.source)
        if client is None:
            return
        chunk_ids = citation.chunk_ids or [citation.chunk_id]
        chunks = []
        for cid in chunk_ids:
            chunk = await client.get_chunk_by_id(cid)
            if chunk:
                chunks.append(chunk)
        if not chunks:
            return

        from haiku.rag.inspector.widgets.visual_modal import VisualGroundingModal

        await self.push_screen(
            VisualGroundingModal(
                chunk=chunks,
                client=client,
                refs=citation.doc_item_refs or None,
            )
        )

    async def action_show_info(self) -> None:
        """Show database info modal."""
        if not self.client:
            return

        from haiku.rag.inspector.widgets.info_modal import InfoModal

        await self.push_screen(InfoModal(self.client))

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
        """Scope the conversation to the selection: the filter carries the ids,
        and `sources` restricts the search to the databases the selection names.
        """
        from haiku.rag.tools.filters import build_document_id_filter

        self._document_filter = event.selected

        doc_filter = build_document_id_filter(
            sorted({doc_id for _, doc_id in event.selected})
        )
        selected_sources = {source for source, _ in event.selected}
        sources: list[str] | None = None
        if selected_sources and None not in selected_sources:
            sources = sorted(s for s in selected_sources if s is not None)
        for namespace, state_type in (
            (RAG_STATE_NAMESPACE, RAGState),
            (ANALYSIS_STATE_NAMESPACE, AnalysisState),
        ):
            if namespace in self._state:
                state = state_type.model_validate(self._state[namespace])
                state.document_filter = doc_filter
                state.sources = sources
                self._state[namespace] = state.model_dump(mode="json")

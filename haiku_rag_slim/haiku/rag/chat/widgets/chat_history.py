from typing import TYPE_CHECKING

from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, LoadingIndicator, Markdown, Static

from haiku.rag.agents.chat.state import CitationInfo

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Key


class ChatMessage(Static):
    """A single chat message with role styling."""

    def __init__(self, role: str, content: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def compose(self) -> "ComposeResult":
        prefix = "**You:**" if self.role == "user" else "**Assistant:**"
        yield Markdown(f"{prefix}\n\n{self.content}", id="message-content")

    def update_content(self, content: str) -> None:
        """Update the message content (for streaming)."""
        self.content = content
        prefix = "**You:**" if self.role == "user" else "**Assistant:**"
        markdown = self.query_one("#message-content", Markdown)
        markdown.update(f"{prefix}\n\n{content}")


class ToolCallWidget(Static):
    """Styled inline display of a tool call."""

    TOOL_LABELS = {
        "search": "Searching",
        "ask": "Asking",
        "get_document": "Fetching",
    }

    def __init__(self, tool_name: str, args: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.args = args or {}
        self._complete = False

    def compose(self) -> "ComposeResult":
        label = self.TOOL_LABELS.get(self.tool_name, self.tool_name)

        # Build description based on tool type
        if self.tool_name == "search":
            query = self.args.get("query", "...")
            doc = self.args.get("document_name")
            desc = f'"{query}"'
            if doc:
                desc += f" in {doc}"
        elif self.tool_name == "ask":
            question = self.args.get("question", "...")
            doc = self.args.get("document_name")
            desc = f'"{question}"'
            if doc:
                desc += f" from {doc}"
        elif self.tool_name == "get_document":
            query = self.args.get("query", "...")
            desc = f'"{query}"'
        else:
            desc = str(self.args) if self.args else ""

        with Horizontal(classes="tool-row"):
            if self._complete:
                yield Static("✓", classes="tool-status")
            else:
                yield LoadingIndicator(classes="tool-spinner")
            yield Static(label, classes="tool-badge")
            yield Static(desc, classes="tool-desc")

    def mark_complete(self) -> None:
        self._complete = True
        self.refresh(recompose=True)


class CitationWidget(Collapsible):
    """Inline expandable citation."""

    can_focus = True
    can_focus_children = False

    class Selected(Message):
        """Message sent when a citation is selected."""

        def __init__(self, citation_index: int) -> None:
            super().__init__()
            self.citation_index = citation_index

    def __init__(self, citation: CitationInfo, **kwargs) -> None:
        title = f"[{citation.index}] {citation.document_title or citation.document_uri}"
        if citation.page_numbers:
            pages = ", ".join(map(str, citation.page_numbers[:3]))
            if len(citation.page_numbers) > 3:
                pages += "..."
            title += f" (p.{pages})"

        # Build content widgets
        content = citation.content
        if len(content) > 500:
            content = content[:500] + "..."

        children: list[Markdown | Static] = [Markdown(content)]
        if citation.headings:
            headings = " > ".join(citation.headings[:3])
            children.append(Static(f"Section: {headings}", classes="citation-metadata"))
        children.append(
            Static(f"Source: {citation.document_uri}", classes="citation-metadata")
        )

        super().__init__(*children, title=title, collapsed=True, **kwargs)
        self.citation = citation

    def on_focus(self) -> None:
        """When focused, mark as selected."""
        self.post_message(self.Selected(self.citation.index - 1))

    def on_key(self, event: "Key") -> None:
        """Handle Enter to toggle expand/collapse."""
        if event.key == "enter":
            self.collapsed = not self.collapsed
            event.stop()


class ThinkingWidget(Static):
    """Thinking indicator shown while agent is processing."""

    def compose(self) -> "ComposeResult":
        with Horizontal(classes="thinking-row"):
            yield LoadingIndicator(classes="thinking-spinner")
            yield Static("Thinking...", classes="thinking-text")


class SourcesHeader(Static):
    """Header for the citations section."""

    def __init__(self, count: int, **kwargs) -> None:
        super().__init__(f"Sources ({count})", **kwargs)


class ChatHistory(VerticalScroll):
    """Scrollable container for chat messages, tool calls, and citations."""

    can_focus = True

    DEFAULT_CSS = """
    ChatHistory {
        height: 100%;
        background: $surface;
        padding: 1 2;
    }

    /* Messages */
    ChatMessage {
        margin: 1 0;
        padding: 1 2;
        background: $panel;
    }

    ChatMessage.user {
        background: $primary 15%;
        border-left: thick $primary;
        margin-right: 4;
    }

    ChatMessage.assistant {
        background: $success 15%;
        border-left: thick $success;
        margin-left: 4;
    }

    ChatMessage Markdown {
        margin: 0;
        padding: 0;
    }

    /* Tool calls */
    ToolCallWidget {
        margin: 0 0 0 4;
        padding: 0 1;
        height: auto;
        background: $surface;
        border-left: thick $warning;
    }

    ToolCallWidget.complete {
        border-left: thick $success;
    }

    .tool-row {
        height: auto;
        width: 100%;
    }

    .tool-spinner {
        width: 2;
        height: 1;
        color: $warning;
    }

    .tool-status {
        width: 2;
        color: $success;
    }

    .tool-badge {
        width: auto;
        color: $text;
        text-style: bold;
        padding-right: 1;
    }

    .tool-desc {
        width: 1fr;
        color: $text-muted;
    }

    /* Sources section */
    SourcesHeader {
        margin: 2 0 1 0;
        padding: 0 1;
        text-style: bold;
        color: $text;
        background: $primary 15%;
        border-left: thick $primary;
    }

    /* Citations */
    CitationWidget {
        margin: 0 0 0 2;
        background: $surface;
    }

    CitationWidget > CollapsibleTitle {
        padding: 0 1;
        color: $text-muted;
    }

    CitationWidget:focus {
        background: $accent 15%;
        border-left: thick $accent;
    }

    CitationWidget:focus > CollapsibleTitle {
        color: $text;
        text-style: bold;
    }

    CitationWidget.selected {
        background: $accent 15%;
        border-left: thick $accent;
    }

    CitationWidget.selected > CollapsibleTitle {
        color: $text;
        text-style: bold;
    }

    CitationWidget Contents {
        padding: 1 2;
        background: $panel;
    }

    CitationWidget .citation-metadata {
        margin-top: 1;
        color: $text-muted;
        text-style: italic;
    }

    /* Thinking indicator */
    ThinkingWidget {
        margin: 1 0 0 4;
        padding: 0 1;
        height: auto;
        background: $surface;
        border-left: thick $primary;
    }

    .thinking-row {
        height: auto;
        width: 100%;
    }

    .thinking-spinner {
        width: 2;
        height: 1;
        color: $primary;
    }

    .thinking-text {
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.messages: list[tuple[str, str]] = []

    async def add_message(self, role: str, content: str = "") -> ChatMessage:
        """Add a message to the chat history."""
        self.messages.append((role, content))
        message_widget = ChatMessage(role, content, classes=role)
        await self.mount(message_widget)
        self.scroll_end(animate=False)
        return message_widget

    async def add_tool_call(
        self, tool_name: str, args: dict | None = None
    ) -> ToolCallWidget:
        """Add an inline tool call indicator."""
        widget = ToolCallWidget(tool_name, args)
        await self.mount(widget)
        self.scroll_end(animate=False)
        return widget

    def mark_tool_complete(self, widget: ToolCallWidget) -> None:
        """Mark a tool call as complete."""
        widget.mark_complete()
        widget.add_class("complete")

    async def add_citations(self, citations: list[CitationInfo]) -> None:
        """Add citations inline after a response."""
        if not citations:
            return
        await self.mount(SourcesHeader(len(citations)))
        for citation in citations:
            widget = CitationWidget(citation)
            await self.mount(widget)
        self.scroll_end(animate=False)

    async def show_thinking(self) -> None:
        """Show the thinking indicator."""
        await self.mount(ThinkingWidget(id="thinking"))
        self.scroll_end(animate=False)

    def hide_thinking(self) -> None:
        """Hide the thinking indicator."""
        try:
            self.query_one("#thinking", ThinkingWidget).remove()
        except Exception:
            pass

    async def clear_messages(self) -> None:
        """Clear all messages from the chat history."""
        self.messages.clear()
        await self.remove_children()

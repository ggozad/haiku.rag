from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown, Static, TextArea

if TYPE_CHECKING:
    from haiku.rag.tools.session import SessionContext


class ContextModal(ModalScreen):  # pragma: no cover
    """Modal screen for viewing/editing context.

    Before first message (not locked, no session context): Edit initial context
    After first message (locked or has session context): View session context
    """

    BINDINGS = [
        Binding("escape", "cancel", "Close", show=False),
        Binding("ctrl+o", "cancel", "Close", show=False),
    ]

    CSS = """
    ContextModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.5);
    }

    #context-container {
        width: 70;
        height: auto;
        max-height: 32;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }

    #context-header {
        height: auto;
        margin-bottom: 1;
    }

    #context-description {
        height: auto;
        margin-bottom: 1;
        color: $text-muted;
    }

    #context-editor {
        height: 12;
        min-height: 8;
        max-height: 16;
    }

    #context-content {
        height: 1fr;
        max-height: 16;
        scrollbar-gutter: stable;
    }

    #button-row {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #button-row Button {
        margin-left: 1;
        min-width: 10;
    }
    """

    class ContextUpdated(Message):
        """Emitted when the context is saved."""

        def __init__(self, context: str) -> None:
            super().__init__()
            self.context = context

    def __init__(
        self,
        initial_context: str | None = None,
        session_context: "SessionContext | None" = None,
        is_locked: bool = False,
    ) -> None:
        super().__init__()
        self._initial_context = initial_context
        self._session_context = session_context
        self._is_locked = is_locked

    @property
    def _is_edit_mode(self) -> bool:
        """Edit mode when not locked and no session context yet."""
        has_session_context = self._session_context and self._session_context.summary
        return not self._is_locked and not has_session_context

    def compose(self) -> ComposeResult:
        with Vertical(id="context-container"):
            if self._is_edit_mode:
                yield Static("[bold]Initial Context[/bold]", id="context-header")
                yield Static(
                    "Set background context to guide the conversation. "
                    "This will be locked after you send your first message.",
                    id="context-description",
                )
                initial_value = self._initial_context or ""
                yield TextArea(initial_value, id="context-editor")
                with Horizontal(id="button-row"):
                    yield Button("Cancel", id="cancel-btn", variant="default")
                    yield Button("Save", id="save-btn", variant="primary")
            else:
                yield Static("[bold]Session Context[/bold]", id="context-header")
                yield Static(
                    "What the assistant has learned from your conversation.",
                    id="context-description",
                )
                with VerticalScroll(id="context-content"):
                    yield Markdown(self._get_session_content())
                with Horizontal(id="button-row"):
                    yield Button("Close", id="cancel-btn", variant="primary")

    def _get_session_content(self) -> str:
        if not self._session_context:
            return "*No session context yet. Ask a question first.*"

        ctx = self._session_context
        updated = (
            ctx.last_updated.strftime("%Y-%m-%d %H:%M:%S")
            if ctx.last_updated
            else "unknown"
        )

        return f"**Last updated:** {updated}\n\n---\n\n{ctx.summary}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "save-btn":
            self.action_save()

    def action_cancel(self) -> None:
        """Cancel and close without saving."""
        self.app.pop_screen()

    def action_save(self) -> None:
        """Save context and close."""
        editor = self.query_one("#context-editor", TextArea)
        self.post_message(self.ContextUpdated(editor.text))
        self.app.pop_screen()

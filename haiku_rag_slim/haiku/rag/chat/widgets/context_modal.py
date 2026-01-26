from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

if TYPE_CHECKING:
    from haiku.rag.agents.chat.state import ChatSessionState


class ContextModal(ModalScreen):  # pragma: no cover
    """Modal screen for displaying session context."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("ctrl+o", "dismiss", "Close", show=True),
    ]

    CSS = """
    ContextModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.5);
    }

    #context-container {
        width: auto;
        min-width: 40;
        max-width: 80;
        height: auto;
        max-height: 20;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }

    #context-header {
        height: auto;
        margin-bottom: 1;
    }

    #context-content {
        height: 1fr;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self, session_state: "ChatSessionState | None"):
        super().__init__()
        self.session_state = session_state

    def compose(self) -> ComposeResult:
        with Vertical(id="context-container"):
            yield Static("[bold]Session Context[/bold]", id="context-header")
            with VerticalScroll(id="context-content"):
                yield Markdown(self._get_content())

    def _get_content(self) -> str:
        if not self.session_state:
            return "*No session state.*"

        if not self.session_state.session_context:
            return "*No session context yet. Ask a question first.*"

        ctx = self.session_state.session_context
        updated = (
            ctx.last_updated.strftime("%Y-%m-%d %H:%M:%S")
            if ctx.last_updated
            else "unknown"
        )

        return f"**Last updated:** {updated}\n\n---\n\n{ctx.summary}"

    async def action_dismiss(self, result=None) -> None:
        self.app.pop_screen()

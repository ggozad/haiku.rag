from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown, Static


class ContextModal(ModalScreen):  # pragma: no cover
    """Modal screen for viewing session Q&A history."""

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

    def __init__(self, qa_history: list | None = None) -> None:
        super().__init__()
        self._qa_history = qa_history or []

    def compose(self) -> ComposeResult:
        with Vertical(id="context-container"):
            yield Static("[bold]Session Context[/bold]", id="context-header")
            yield Static(
                "Questions and answers from this session.",
                id="context-description",
            )
            with VerticalScroll(id="context-content"):
                yield Markdown(self._get_content())
            with Horizontal(id="button-row"):
                yield Button("Close", id="cancel-btn", variant="primary")

    def _get_content(self) -> str:
        if not self._qa_history:
            return "*No questions asked yet.*"

        parts = []
        for entry in self._qa_history:
            q = getattr(entry, "question", str(entry))
            a = getattr(entry, "answer", "")
            parts.append(f"**Q:** {q}\n\n**A:** {a}")

        return "\n\n---\n\n".join(parts)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.action_cancel()

    def action_cancel(self) -> None:
        """Cancel and close."""
        self.app.pop_screen()

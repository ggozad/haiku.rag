from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Static

from haiku.rag.client import HaikuRAG


class DocumentFilterModal(ModalScreen):  # pragma: no cover
    """Modal screen for selecting documents to filter searches."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    CSS = """
    DocumentFilterModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.5);
    }

    #filter-container {
        width: 60;
        height: auto;
        max-height: 28;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }

    #filter-header {
        height: auto;
        margin-bottom: 1;
    }

    #filter-search {
        margin-bottom: 1;
    }

    #filter-list {
        height: 1fr;
        min-height: 8;
        max-height: 16;
        scrollbar-gutter: stable;
    }

    #filter-footer {
        height: auto;
        margin-top: 1;
        color: $text-muted;
    }

    #button-row {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #button-row Button {
        margin-left: 1;
    }

    .doc-checkbox {
        height: auto;
        padding: 0 1;
    }

    .doc-checkbox:hover {
        background: $surface-lighten-1;
    }
    """

    class FilterChanged(Message):
        """Emitted when the document filter selection changes."""

        def __init__(self, selected: list[str]) -> None:
            super().__init__()
            self.selected = selected

    def __init__(
        self,
        client: HaikuRAG,
        selected: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.initial_selected = selected or []
        self._selected: set[str] = set(self.initial_selected)

    def compose(self) -> ComposeResult:
        with Vertical(id="filter-container"):
            yield Static("[bold]Filter Documents[/bold]", id="filter-header")
            yield Input(placeholder="Search documents...", id="filter-search")
            with VerticalScroll(id="filter-list"):
                yield Static("Loading...", id="loading-indicator")
            yield Static("", id="filter-footer")
            with Horizontal(id="button-row"):
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button("Apply", id="apply-btn", variant="primary")

    async def on_mount(self) -> None:
        """Load documents when mounted."""
        await self._load_documents()

    async def _load_documents(self) -> None:
        """Load all documents from the client."""
        docs = await self.client.list_documents()

        # Remove loading indicator
        loading = self.query_one("#loading-indicator", Static)
        loading.remove()

        # Add checkboxes for all documents
        filter_list = self.query_one("#filter-list", VerticalScroll)
        for doc in docs:
            display_name = doc.title or doc.uri or str(doc.id)
            checkbox = Checkbox(
                display_name,
                value=display_name in self._selected,
                id=f"doc-{hash(display_name)}",
                classes="doc-checkbox",
            )
            checkbox._doc_id = display_name  # type: ignore[attr-defined]
            await filter_list.mount(checkbox)

        self._update_footer()

    def _update_footer(self) -> None:
        """Update the footer with selection count."""
        footer = self.query_one("#filter-footer", Static)
        count = len(self._selected)
        if count == 0:
            footer.update("[dim]No filter (all documents)[/dim]")
        else:
            footer.update(f"[bold]{count}[/bold] document(s) selected")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state changes."""
        checkbox = event.checkbox
        doc_id = getattr(checkbox, "_doc_id", None)
        if doc_id is None:
            return

        if event.value:
            self._selected.add(doc_id)
        else:
            self._selected.discard(doc_id)

        self._update_footer()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter document list based on search input."""
        search_term = event.value.lower().strip()
        filter_list = self.query_one("#filter-list", VerticalScroll)

        for checkbox in filter_list.query(Checkbox):
            doc_id = getattr(checkbox, "_doc_id", "")
            if search_term == "" or search_term in doc_id.lower():
                checkbox.display = True
            else:
                checkbox.display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.action_cancel()
        elif event.button.id == "apply-btn":
            self.action_confirm()

    def action_cancel(self) -> None:
        """Cancel and close without saving."""
        self.app.pop_screen()

    def action_confirm(self) -> None:
        """Confirm selection and close."""
        self.post_message(self.FilterChanged(list(self._selected)))
        self.app.pop_screen()

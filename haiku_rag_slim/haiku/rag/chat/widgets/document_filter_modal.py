from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.utils import escape_sql_string

# Documents listed at once. Mounting a checkbox per document wedges the modal on
# a large corpus, so the rest is reached through the search box.
DOCUMENT_PAGE = 200


class DocumentCheckbox(Checkbox):
    def __init__(self, label: str, doc_id: str, *, value: bool) -> None:
        super().__init__(label, value=value, classes="doc-checkbox")
        self.doc_id = doc_id
        # `label` is a reactive Text; narrowing the page wants the plain string.
        self.search_text = label


def search_filter(term: str) -> str | None:
    """A document filter matching `term` in a title or URI, or None for no term.

    The term is text to find, not a pattern: `LIKE` is case-sensitive and reads
    `%` and `_` as wildcards, so both sides are lowered and the term's own
    wildcards are escaped.
    """
    term = term.strip()
    if not term:
        return None
    literal = term.lower()
    for wildcard in ("\\", "%", "_"):
        literal = literal.replace(wildcard, f"\\{wildcard}")
    escaped = escape_sql_string(f"%{literal}%")
    return (
        f"LOWER(title) LIKE '{escaped}' ESCAPE '\\' "
        f"OR LOWER(uri) LIKE '{escaped}' ESCAPE '\\'"
    )


class DocumentFilterModal(ModalScreen):
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
        self._shown = 0
        self._matching = 0

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
        """Load the first page of documents when mounted."""
        await self._load_documents()

    async def _load_documents(self, search: str = "") -> None:
        """Show one page of documents, narrowed by `search` when given."""
        document_filter = search_filter(search)
        docs = await self.client.list_documents(
            limit=DOCUMENT_PAGE, filter=document_filter
        )
        self._matching = await self.client.count_documents(filter=document_filter)

        filter_list = self.query_one("#filter-list", VerticalScroll)
        await filter_list.remove_children()

        # Sort the interleaved page and label each document's database, which
        # a title alone does not say.
        labelled = sorted(
            (
                (
                    f"{doc.title or doc.uri or doc.id}"
                    + (f"  ({doc.source})" if doc.source else ""),
                    doc.id,
                )
                for doc in docs
                if doc.id is not None
            ),
            key=lambda pair: pair[0],
        )

        boxes = [
            DocumentCheckbox(label, doc_id, value=doc_id in self._selected)
            for label, doc_id in labelled
        ]
        if boxes:
            await filter_list.mount_all(boxes)

        self._shown = len(boxes)
        self._update_footer()

    def _update_footer(self) -> None:
        """Update the footer with the selection and how much of the corpus is shown."""
        footer = self.query_one("#filter-footer", Static)
        count = len(self._selected)
        if count == 0:
            state = "[dim]No filter (all documents)[/dim]"
        else:
            state = f"[bold]{count}[/bold] document(s) selected"
        if self._matching > self._shown:
            state += (
                f" [dim]— showing {self._shown} of {self._matching};"
                " type and press enter to search[/dim]"
            )
        footer.update(state)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state changes."""
        checkbox = event.checkbox
        if not isinstance(checkbox, DocumentCheckbox):
            return

        if event.value:
            self._selected.add(checkbox.doc_id)
        else:
            self._selected.discard(checkbox.doc_id)

        self._update_footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Ask the database for documents matching the search."""
        await self._load_documents(event.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Narrow the page already shown, for feedback while typing."""
        search_term = event.value.lower().strip()
        filter_list = self.query_one("#filter-list", VerticalScroll)

        for checkbox in filter_list.query(DocumentCheckbox):
            checkbox.display = (
                search_term == "" or search_term in checkbox.search_text.lower()
            )

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

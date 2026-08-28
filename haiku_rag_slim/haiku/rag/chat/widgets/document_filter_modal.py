from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.tools.filters import build_document_id_filter
from haiku.rag.utils import escape_sql_string

# Documents listed at once. Mounting a checkbox per document wedges the modal on
# a large corpus, so the rest is reached through the search box.
DOCUMENT_PAGE = 200


class DocumentCheckbox(Checkbox):
    def __init__(self, label: str, doc_id: str, *, value: bool) -> None:
        super().__init__(label, value=value, classes="doc-checkbox")
        self.doc_id = doc_id


def _labelled(docs) -> list[tuple[str, str]]:
    """Each document's label and id, sorted. The database is named alongside the
    title, which a title alone does not say. Labels are escaped: titles and
    database names are data, not Textual markup."""
    return sorted(
        (
            escape(
                f"{doc.title or doc.uri or doc.id}"
                + (f"  ({doc.source})" if doc.source else "")
            ),
            doc.id,
        )
        for doc in docs
        if doc.id is not None
    )


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
        self._matching = 0
        self._search = ""
        self._page = 0
        # Selections outside the page stay applied, and a checkbox is the only
        # way to remove one, so their own listing keeps them reachable while
        # this one holds the page bound.
        self._listing_selected = False

    def compose(self) -> ComposeResult:
        with Vertical(id="filter-container"):
            yield Static("[bold]Filter Documents[/bold]", id="filter-header")
            yield Input(
                placeholder="Search documents; press Enter to search...",
                id="filter-search",
            )
            with VerticalScroll(id="filter-list"):
                yield Static("Loading...", id="loading-indicator")
            yield Static("", id="filter-footer")
            with Horizontal(id="button-row"):
                yield Button("Selected", id="selected-btn", variant="default")
                yield Button("Prev", id="prev-btn", variant="default")
                yield Button("Next", id="next-btn", variant="default")
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button("Apply", id="apply-btn", variant="primary")

    async def on_mount(self) -> None:
        """Load the first page of documents when mounted."""
        await self._load_documents()

    async def _load_documents(self, search: str | None = None) -> None:
        """Show one page of whichever listing is on screen.

        Either the documents matching the search, or the selected ones. Both
        page at `DOCUMENT_PAGE`, so the mounted widgets stay bounded whichever
        is showing and every selection stays reachable.
        """
        if search is not None:
            self._search = search.strip()
            self._page = 0
            self._listing_selected = False

        if self._listing_selected:
            ids = sorted(self._selected)
            self._matching = len(ids)
            page = ids[self._page * DOCUMENT_PAGE : (self._page + 1) * DOCUMENT_PAGE]
            docs = (
                list(
                    await self.client.list_documents(
                        filter=build_document_id_filter(page)
                    )
                )
                if page
                else []
            )
        else:
            document_filter = search_filter(self._search)
            docs = list(
                await self.client.list_documents(
                    limit=DOCUMENT_PAGE,
                    offset=self._page * DOCUMENT_PAGE,
                    filter=document_filter,
                )
            )
            self._matching = await self.client.count_documents(filter=document_filter)

        filter_list = self.query_one("#filter-list", VerticalScroll)
        await filter_list.remove_children()

        boxes = [
            DocumentCheckbox(label, doc_id, value=doc_id in self._selected)
            for label, doc_id in _labelled(docs)
        ]
        if boxes:
            await filter_list.mount_all(boxes)
        else:
            # Otherwise the list is an empty box, indistinguishable from one
            # still loading.
            empty = (
                "Nothing selected." if self._listing_selected else "No documents match."
            )
            await filter_list.mount(Static(empty, id="filter-empty"))

        self._update_footer()

    @property
    def _pages(self) -> int:
        """Pages the current listing spans, at least one."""
        return max(1, -(-self._matching // DOCUMENT_PAGE))

    def _update_footer(self) -> None:
        """Report the selection, where in the listing this page sits, and
        whether the search box holds a term the listing was not built from."""
        footer = self.query_one("#filter-footer", Static)
        count = len(self._selected)
        if count == 0:
            state = "[dim]No filter (all documents)[/dim]"
        else:
            state = f"[bold]{count}[/bold] document(s) selected"
        if self._listing_selected:
            state += " [dim]— listing the selected[/dim]"
        if self._pages > 1:
            state += (
                f" [dim]— page {self._page + 1} of {self._pages}"
                f" ({self._matching} total)[/dim]"
            )
        if self.query_one("#filter-search", Input).value.strip() != self._search:
            state += " [dim]— press enter to search[/dim]"
        footer.update(state)

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state changes."""
        checkbox = event.checkbox
        if not isinstance(checkbox, DocumentCheckbox):
            return

        if event.value:
            self._selected.add(checkbox.doc_id)
        else:
            self._selected.discard(checkbox.doc_id)

        if self._listing_selected:
            # This listing is the selection, so removing one changes both what
            # it holds and how far it runs. The last page can stop existing.
            pages = max(1, -(-len(self._selected) // DOCUMENT_PAGE))
            self._page = min(self._page, pages - 1)
            await self._load_documents()
            return

        self._update_footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Ask the database for documents matching the search."""
        await self._load_documents(event.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Prompt when the search term has not been submitted."""
        self._update_footer()

    async def _turn_to(self, page: int) -> None:
        """Show `page` of the current listing, if there is one."""
        if 0 <= page < self._pages:
            self._page = page
            await self._load_documents()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "selected-btn":
            self._listing_selected = not self._listing_selected
            self._page = 0
            await self._load_documents()
            return
        if event.button.id == "prev-btn":
            await self._turn_to(self._page - 1)
            return
        if event.button.id == "next-btn":
            await self._turn_to(self._page + 1)
            return
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

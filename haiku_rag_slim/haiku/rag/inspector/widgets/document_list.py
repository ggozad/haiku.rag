from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import ListItem, ListView, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Document


class DocumentList(VerticalScroll):  # pragma: no cover
    """Widget for displaying and browsing documents."""

    can_focus = False

    class DocumentSelected(Message):
        """Message sent when a document is selected."""

        def __init__(self, document: Document) -> None:
            super().__init__()
            self.document = document

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.documents: list[Document] = []
        self.list_view = ListView()

    def compose(self) -> ComposeResult:
        """Compose the document list."""
        yield Static("[bold]Documents[/bold]", classes="title")
        yield self.list_view

    async def load_documents(self, client: HaikuRAG) -> None:
        """Load all documents from the database."""
        self.documents = await client.list_documents(limit=None)
        await self.list_view.clear()
        for doc in self.documents:
            title = doc.title or doc.uri or doc.id
            await self.list_view.append(ListItem(Static(f"{title}")))

    @on(ListView.Highlighted)
    @on(ListView.Selected)
    async def handle_document_selection(
        self, event: ListView.Highlighted | ListView.Selected
    ) -> None:
        """Handle document selection (arrow keys or Enter)."""
        if event.list_view != self.list_view:
            return
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self.documents):
            self.post_message(self.DocumentSelected(self.documents[idx]))

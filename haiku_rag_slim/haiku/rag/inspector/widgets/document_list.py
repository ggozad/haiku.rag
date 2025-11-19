from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import ListItem, ListView, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Document


class DocumentList(VerticalScroll):
    """Widget for displaying and browsing documents."""

    class DocumentSelected(Message):
        """Message sent when a document is selected."""

        def __init__(self, document: Document) -> None:
            super().__init__()
            self.document = document

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.documents: list[Document] = []
        self.list_view: ListView | None = None

    def compose(self) -> ComposeResult:
        """Compose the document list."""
        yield Static("[bold]Documents[/bold]", classes="title")
        self.list_view = ListView()
        yield self.list_view

    async def load_documents(
        self, client: HaikuRAG, limit: int = 100, offset: int = 0
    ) -> None:
        """Load documents from the database.

        Args:
            client: HaikuRAG client instance
            limit: Maximum number of documents to load
            offset: Offset for pagination
        """
        if self.list_view is None:
            return

        self.documents = await client.list_documents(limit=limit, offset=offset)

        # Clear existing items
        await self.list_view.clear()

        # Add document items
        for doc in self.documents:
            title = doc.title or doc.uri or doc.id or "Untitled"
            item = ListItem(Static(f"{title}"))
            await self.list_view.append(item)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle document selection."""
        if event.list_view == self.list_view:
            idx = event.list_view.index
            if idx is not None and 0 <= idx < len(self.documents):
                document = self.documents[idx]
                self.post_message(self.DocumentSelected(document))

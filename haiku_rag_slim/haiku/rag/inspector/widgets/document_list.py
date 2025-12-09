from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import ListItem, ListView, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Document

BATCH_SIZE = 50


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
        self.has_more: bool = True
        self._client: HaikuRAG | None = None
        self._loading: bool = False

    def compose(self) -> ComposeResult:
        """Compose the document list."""
        yield Static("[bold]Documents[/bold]", classes="title")
        yield self.list_view

    async def load_documents(self, client: HaikuRAG) -> None:
        """Load initial batch of documents from the database."""
        self._client = client
        self.documents = await client.list_documents(limit=BATCH_SIZE, offset=0)
        self.has_more = len(self.documents) >= BATCH_SIZE
        await self.list_view.clear()
        for doc in self.documents:
            title = doc.title or doc.uri or doc.id
            await self.list_view.append(ListItem(Static(f"{title}")))

    async def load_more(self, client: HaikuRAG) -> None:
        """Load the next batch of documents."""
        if not self.has_more or self._loading:
            return
        self._loading = True
        offset = len(self.documents)
        new_docs = await client.list_documents(limit=BATCH_SIZE, offset=offset)
        self.has_more = len(new_docs) >= BATCH_SIZE
        self.documents.extend(new_docs)
        for doc in new_docs:
            title = doc.title or doc.uri or doc.id
            await self.list_view.append(ListItem(Static(f"{title}")))
        self._loading = False

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
            # Infinite scroll: load more when near the end
            if self._client and self.has_more and idx >= len(self.documents) - 10:
                await self.load_more(self._client)

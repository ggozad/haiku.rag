from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import ListItem, ListView, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Chunk


class ChunkList(VerticalScroll):
    """Widget for displaying and browsing chunks."""

    can_focus = False

    class ChunkSelected(Message):
        """Message sent when a chunk is selected."""

        def __init__(self, chunk: Chunk) -> None:
            super().__init__()
            self.chunk = chunk

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chunks: list[Chunk] = []
        self.list_view: ListView | None = None

    def compose(self) -> ComposeResult:
        """Compose the chunk list."""
        yield Static("[bold]Chunks[/bold]", classes="title")
        self.list_view = ListView()
        yield self.list_view

    async def load_chunks_for_document(
        self, client: HaikuRAG, document_id: str
    ) -> None:
        """Load chunks for a specific document.

        Args:
            client: HaikuRAG client instance
            document_id: ID of the document to load chunks for
        """
        if self.list_view is None:
            return

        self.chunks = await client.chunk_repository.get_by_document_id(document_id)

        # Clear existing items
        await self.list_view.clear()

        # Add chunk items
        for chunk in self.chunks:
            first_line = chunk.content.split("\n")[0]
            item = ListItem(Static(f"[{chunk.order}] {first_line}"))
            await self.list_view.append(item)

    async def load_chunks_from_search(
        self, client: HaikuRAG, query: str, limit: int = 20
    ) -> None:
        """Load chunks from search results.

        Args:
            client: HaikuRAG client instance
            query: Search query
            limit: Maximum number of results
        """
        if self.list_view is None:
            return

        results = await client.chunk_repository.search(
            query=query, limit=limit, search_type="hybrid"
        )
        self.chunks = [chunk for chunk, _score in results]

        # Clear existing items
        await self.list_view.clear()

        # Add chunk items with scores
        for chunk, score in results:
            first_line = chunk.content.split("\n")[0]
            item = ListItem(Static(f"[{score:.2f}] {first_line}"))
            await self.list_view.append(item)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle chunk selection."""
        if event.list_view == self.list_view:
            idx = event.list_view.index
            if idx is not None and 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                self.post_message(self.ChunkSelected(chunk))

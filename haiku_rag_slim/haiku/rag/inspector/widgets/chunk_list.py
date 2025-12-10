from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import ListItem, ListView, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Chunk


class ChunkList(VerticalScroll):  # pragma: no cover
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
        self.list_view = ListView()

    def compose(self) -> ComposeResult:
        """Compose the chunk list."""
        yield Static("[bold]Chunks[/bold]", classes="title")
        yield self.list_view

    async def load_chunks_for_document(
        self, client: HaikuRAG, document_id: str
    ) -> None:
        """Load chunks for a specific document."""
        self.chunks = await client.chunk_repository.get_by_document_id(document_id)
        await self.list_view.clear()
        for chunk in self.chunks:
            first_line = chunk.content.split("\n")[0]
            await self.list_view.append(
                ListItem(Static(f"[{chunk.order}] {first_line}"))
            )

    @on(ListView.Highlighted)
    @on(ListView.Selected)
    async def handle_chunk_selection(
        self, event: ListView.Highlighted | ListView.Selected
    ) -> None:
        """Handle chunk selection (arrow keys or Enter)."""
        if event.list_view != self.list_view:
            return
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self.chunks):
            self.post_message(self.ChunkSelected(self.chunks[idx]))

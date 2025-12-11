from textual import on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import ListItem, ListView, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.store.models import Chunk

BATCH_SIZE = 50


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
        self.has_more: bool = False
        self._client: HaikuRAG | None = None
        self._document_id: str | None = None
        self._loading: bool = False
        self._total_chunks: int = 0

    def compose(self) -> ComposeResult:
        """Compose the chunk list."""
        yield Static("[bold]Chunks[/bold]", classes="title")
        yield self.list_view

    async def load_chunks_for_document(
        self, client: HaikuRAG, document_id: str
    ) -> None:
        """Load initial batch of chunks for a specific document."""
        self._client = client
        self._document_id = document_id
        self._total_chunks = await client.chunk_repository.count_by_document_id(
            document_id
        )

        self.chunks = await client.chunk_repository.get_by_document_id(
            document_id, limit=BATCH_SIZE, offset=0
        )
        self.has_more = len(self.chunks) < self._total_chunks

        await self.list_view.clear()
        for chunk in self.chunks:
            first_line = chunk.content.split("\n")[0]
            await self.list_view.append(
                ListItem(Static(f"[{chunk.order}] {first_line}"))
            )

    async def load_more(self) -> None:
        """Load the next batch of chunks."""
        if (
            not self.has_more
            or self._loading
            or not self._client
            or not self._document_id
        ):
            return

        self._loading = True
        offset = len(self.chunks)
        new_chunks = await self._client.chunk_repository.get_by_document_id(
            self._document_id, limit=BATCH_SIZE, offset=offset
        )
        self.has_more = (offset + len(new_chunks)) < self._total_chunks
        self.chunks.extend(new_chunks)

        for chunk in new_chunks:
            first_line = chunk.content.split("\n")[0]
            await self.list_view.append(
                ListItem(Static(f"[{chunk.order}] {first_line}"))
            )
        self._loading = False

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
            # Infinite scroll: load more when near the end
            if self.has_more and idx >= len(self.chunks) - 10:
                await self.load_more()

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Input, ListItem, ListView, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.inspector.widgets.detail_view import DetailView
from haiku.rag.store.models import Chunk


class SearchModal(Screen):
    """Screen for searching chunks."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
    ]

    CSS = """
    SearchModal {
        background: $surface;
        layout: vertical;
    }

    #search-header {
        dock: top;
        height: auto;
    }

    #search-content {
        height: 1fr;
        width: 100%;
    }

    #search-results-container {
        width: 1fr;
        border: solid $primary;
    }

    #search-detail {
        width: 2fr;
        border: solid $accent;
    }

    ListItem {
        overflow: hidden;
    }

    ListItem Static {
        overflow: hidden;
        text-overflow: ellipsis;
    }
    """

    class ChunkSelected(Message):
        """Message sent when a chunk is selected from search results."""

        def __init__(self, chunk: Chunk) -> None:
            super().__init__()
            self.chunk = chunk

    def __init__(self, client: HaikuRAG):
        super().__init__()
        self.client = client
        self.chunks: list[Chunk] = []

    def compose(self) -> ComposeResult:
        """Compose the search screen."""
        with Vertical(id="search-header"):
            yield Static("[bold]Search Chunks[/bold]")
            yield Input(placeholder="Enter search query...", id="search-input")
            yield Static("", id="status-label")
        with Horizontal(id="search-content"):
            with VerticalScroll(id="search-results-container"):
                yield ListView(id="search-results")
            yield DetailView(id="search-detail")

    async def on_mount(self) -> None:
        """Focus the search input when mounted."""
        status_label = self.query_one("#status-label", Static)
        status_label.update("Type query and press Enter to search")
        search_input = self.query_one("#search-input", Input)
        search_input.focus()

    @on(Input.Submitted, "#search-input")
    async def search_submitted(self, event: Input.Submitted) -> None:
        """Handle search query submission."""
        query = event.value.strip()
        if query:
            await self.run_search(query)

    async def run_search(self, query: str) -> None:
        """Perform the search."""
        status_label = self.query_one("#status-label", Static)
        list_view = self.query_one("#search-results", ListView)

        status_label.update("Searching...")

        try:
            # Perform search
            results = await self.client.chunk_repository.search(
                query=query, limit=50, search_type="hybrid"
            )

            self.chunks = [chunk for chunk, _score in results]

            # Clear and populate results
            await list_view.clear()
            for chunk, score in results:
                first_line = chunk.content.split("\n")[0]
                score_str = f"{score:.2f}" if score else "N/A"
                item = ListItem(Static(f"[{score_str}] {first_line}"))
                await list_view.append(item)

            # Update status
            status_label.update(f"Found {len(self.chunks)} results")

            # Select first result, show in detail view, and focus list
            if self.chunks:
                list_view.index = 0
                detail_view = self.query_one("#search-detail", DetailView)
                await detail_view.show_chunk(self.chunks[0])
                list_view.focus()
        except Exception as e:
            status_label.update(f"Error: {str(e)}")

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle chunk navigation (arrow keys)."""
        list_view = self.query_one("#search-results", ListView)
        if event.list_view == list_view and event.item is not None:
            idx = event.list_view.index
            if idx is not None and 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                detail_view = self.query_one("#search-detail", DetailView)
                await detail_view.show_chunk(chunk)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle chunk selection (Enter key)."""
        list_view = self.query_one("#search-results", ListView)
        if event.list_view == list_view:
            idx = event.list_view.index
            if idx is not None and 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                self.post_message(self.ChunkSelected(chunk))
                self.app.pop_screen()

    async def action_dismiss(self, result=None) -> None:
        """Close the search screen."""
        self.app.pop_screen()

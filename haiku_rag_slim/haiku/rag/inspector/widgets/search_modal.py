from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Input, ListItem, ListView, Static

from haiku.rag.client import HaikuRAG
from haiku.rag.inspector.widgets.detail_view import DetailView
from haiku.rag.store.models import Chunk, SearchResult


class SearchModal(Screen):  # pragma: no cover
    """Screen for searching chunks."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("v", "show_visual", "Visual", show=True),
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
        self.search_results: list[SearchResult] = []

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
            # Perform search using client API
            self.search_results = await self.client.search(query=query, limit=50)

            # Get chunks for the results
            self.chunks = []
            for result in self.search_results:
                if result.chunk_id:
                    chunk = await self.client.chunk_repository.get_by_id(
                        result.chunk_id
                    )
                    if chunk:
                        self.chunks.append(chunk)

            # Clear and populate results
            await list_view.clear()
            for result in self.search_results:
                first_line = result.content.split("\n")[0][:60]
                score_str = f"{result.score:.2f}"
                # Add page info if available
                page_info = ""
                if result.page_numbers:
                    pages = ", ".join(str(p) for p in result.page_numbers[:3])
                    page_info = f" (p.{pages})"
                item = ListItem(Static(f"[{score_str}]{page_info} {first_line}"))
                await list_view.append(item)

            # Update status
            status_label.update(f"Found {len(self.chunks)} results")

            # Select first result, show in detail view, and focus list
            if self.chunks and self.search_results:
                list_view.index = 0
                detail_view = self.query_one("#search-detail", DetailView)
                await detail_view.show_search_result(
                    self.chunks[0], self.search_results[0]
                )
                list_view.focus()
        except Exception as e:
            status_label.update(f"Error: {str(e)}")

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Handle chunk navigation (arrow keys)."""
        list_view = self.query_one("#search-results", ListView)
        if event.list_view != list_view or event.item is None:
            return
        idx = event.list_view.index
        detail_view = self.query_one("#search-detail", DetailView)
        await detail_view.show_search_result(self.chunks[idx], self.search_results[idx])  # type: ignore[index]

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle chunk selection (Enter key)."""
        list_view = self.query_one("#search-results", ListView)
        if event.list_view != list_view:
            return
        idx = event.list_view.index
        self.post_message(self.ChunkSelected(self.chunks[idx]))  # type: ignore[index]
        self.app.pop_screen()

    async def action_dismiss(self, result=None) -> None:
        self.app.pop_screen()

    async def action_show_visual(self) -> None:
        """Show visual grounding for the current chunk."""
        list_view = self.query_one("#search-results", ListView)
        idx = list_view.index
        if idx is None or not self.chunks:
            return

        chunk = self.chunks[idx]

        from haiku.rag.inspector.widgets.visual_modal import VisualGroundingModal

        # Use app's _switch_modal to close this modal before opening visual
        await self.app._switch_modal(  # type: ignore[attr-defined]
            VisualGroundingModal(
                chunk=chunk,
                client=self.client,
                document_uri=chunk.document_uri,
            )
        )

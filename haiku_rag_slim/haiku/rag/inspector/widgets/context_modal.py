from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Markdown, Static

from haiku.rag.store.models import SearchResult

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.store.models import Chunk


class ContextModal(Screen):  # pragma: no cover
    """Modal screen for displaying how a chunk appears to agents."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("c", "dismiss", "Close", show=True),
    ]

    CSS = """
    ContextModal {
        background: $surface;
        layout: vertical;
    }

    #context-header {
        dock: top;
        height: auto;
        padding: 1;
    }

    #context-content {
        height: 1fr;
        width: 100%;
        padding: 1;
    }

    #context-content Markdown {
        width: 100%;
    }
    """

    def __init__(self, chunk: "Chunk", client: "HaikuRAG"):
        super().__init__()
        self.chunk = chunk
        self.client = client
        self._content_widget = Markdown("Loading...")

    def compose(self) -> ComposeResult:
        yield Static("[bold]Agent Context Format[/bold]", id="context-header")
        with VerticalScroll(id="context-content"):
            yield self._content_widget

    async def on_mount(self) -> None:
        """Load and display the expanded context."""
        # Create a SearchResult from the chunk
        chunk_meta = self.chunk.get_chunk_metadata()
        search_result = SearchResult(
            content=self.chunk.content,
            score=0.0,
            chunk_id=self.chunk.id,
            document_id=self.chunk.document_id,
            document_uri=self.chunk.document_uri,
            document_title=self.chunk.document_title,
            doc_item_refs=chunk_meta.doc_item_refs,
            page_numbers=chunk_meta.page_numbers,
            headings=chunk_meta.headings,
            labels=chunk_meta.labels,
        )

        # Expand context using the client (this is what agents actually receive)
        expanded_results = await self.client.expand_context([search_result])
        expanded = expanded_results[0] if expanded_results else search_result

        formatted = expanded.format_for_agent()

        content = (
            "*This is how the chunk appears to agents after context expansion:*\n\n---\n\n"
            f"{formatted}"
        )

        await self._content_widget.update(content)

    async def action_dismiss(self, result=None) -> None:
        self.app.pop_screen()

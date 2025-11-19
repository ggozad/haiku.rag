# pyright: reportPossiblyUnboundVariable=false
from pathlib import Path
from typing import TYPE_CHECKING

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config

if TYPE_CHECKING:
    from textual.app import ComposeResult

try:
    from textual.app import App
    from textual.binding import Binding
    from textual.widgets import Footer, Header

    from haiku.rag.inspector.widgets.chunk_list import ChunkList
    from haiku.rag.inspector.widgets.detail_view import DetailView
    from haiku.rag.inspector.widgets.document_list import DocumentList

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = object  # type: ignore


class InspectorApp(App):  # type: ignore[misc]
    """Textual TUI for inspecting LanceDB data."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 1fr 2fr;
        grid-rows: 1fr 1fr;
    }

    #document-list {
        column-span: 1;
        row-span: 2;
        border: solid $primary;
    }

    #chunk-list {
        column-span: 1;
        row-span: 1;
        border: solid $secondary;
    }

    #detail-view {
        column-span: 1;
        row-span: 1;
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

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "focus_documents", "Documents", show=True),
        Binding("c", "focus_chunks", "Chunks", show=True),
        Binding("v", "focus_detail", "Detail", show=True),
        Binding("/", "search", "Search", show=True),
    ]

    def __init__(self, db_path: Path):
        super().__init__()
        self.db_path = db_path
        self.client: HaikuRAG | None = None

    def compose(self) -> "ComposeResult":
        """Compose the UI layout."""
        yield Header()
        yield DocumentList(id="document-list")
        yield ChunkList(id="chunk-list")
        yield DetailView(id="detail-view")
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app when mounted."""
        config = get_config()
        self.client = HaikuRAG(db_path=self.db_path, config=config, allow_create=False)
        await self.client.__aenter__()

        # Load initial documents
        doc_list = self.query_one(DocumentList)
        await doc_list.load_documents(self.client)

    async def on_unmount(self) -> None:
        """Clean up when unmounting."""
        if self.client:
            await self.client.__aexit__(None, None, None)

    def action_focus_documents(self) -> None:
        """Focus the documents list."""
        self.query_one(DocumentList).focus()

    def action_focus_chunks(self) -> None:
        """Focus the chunks list."""
        self.query_one(ChunkList).focus()

    def action_focus_detail(self) -> None:
        """Focus the detail view."""
        self.query_one(DetailView).focus()

    def action_search(self) -> None:
        """Open search dialog."""
        # TODO: Implement search dialog
        pass

    async def on_document_list_document_selected(
        self, message: DocumentList.DocumentSelected
    ) -> None:
        """Handle document selection from document list.

        Args:
            message: Message containing selected document
        """
        if not self.client:
            return

        # Show document details
        detail_view = self.query_one(DetailView)
        await detail_view.show_document(message.document)

        # Load chunks for this document
        if message.document.id:
            chunk_list = self.query_one(ChunkList)
            await chunk_list.load_chunks_for_document(self.client, message.document.id)

    async def on_chunk_list_chunk_selected(
        self, message: ChunkList.ChunkSelected
    ) -> None:
        """Handle chunk selection from chunk list.

        Args:
            message: Message containing selected chunk
        """
        # Show chunk details
        detail_view = self.query_one(DetailView)
        await detail_view.show_chunk(message.chunk)


def run_inspector(db_path: Path | None = None) -> None:
    """Run the inspector TUI.

    Args:
        db_path: Path to the LanceDB database. If None, uses default from config.
    """
    if not TEXTUAL_AVAILABLE:
        raise ImportError(
            "Textual is not installed. Install it with: pip install 'haiku.rag-slim[inspector]'"
        )

    config = get_config()
    if db_path is None:
        db_path = config.storage.data_dir / "haiku.rag.lancedb"

    app = InspectorApp(db_path)
    app.run()

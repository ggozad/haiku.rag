# pyright: reportPossiblyUnboundVariable=false
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from haiku.rag.client import HaikuRAG
from haiku.rag.config import get_config

if TYPE_CHECKING:
    from textual.app import ComposeResult

try:
    from textual.app import App
    from textual.binding import Binding
    from textual.screen import Screen
    from textual.widgets import Footer, Header

    from haiku.rag.inspector.widgets.chunk_list import ChunkList
    from haiku.rag.inspector.widgets.detail_view import DetailView
    from haiku.rag.inspector.widgets.document_list import DocumentList
    from haiku.rag.inspector.widgets.search_modal import SearchModal

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = object  # type: ignore


class InspectorApp(App):  # type: ignore[misc]  # pragma: no cover
    """Textual TUI for inspecting LanceDB data."""

    TITLE = "haiku.rag DB Inspector"

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
        Binding("/", "search", "Search", show=True),
        Binding("i", "show_info", "Info", show=True),
        Binding("v", "show_visual", "Visual", show=True),
        Binding("c", "show_context", "Context", show=True),
    ]

    def __init__(
        self, db_path: Path, read_only: bool = False, before: datetime | None = None
    ):
        super().__init__()
        self.db_path = db_path
        self.read_only = read_only
        self.before = before
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
        self.client = HaikuRAG(
            db_path=self.db_path,
            config=config,
            read_only=self.read_only,
            before=self.before,
        )
        await self.client.__aenter__()

        # Load initial documents
        doc_list = self.query_one(DocumentList)
        await doc_list.load_documents(self.client)

        doc_list.list_view.focus()

    async def on_unmount(self) -> None:
        """Clean up when unmounting."""
        if self.client:
            await self.client.__aexit__(None, None, None)

    def _select_chunk(self, chunk_list: ChunkList, chunk_id: str) -> None:
        """Helper to select a chunk after refresh."""
        for idx, c in enumerate(chunk_list.chunks):
            if c.id == chunk_id:
                chunk_list.list_view.index = idx
                chunk_list.list_view.focus()
                break

    async def _dismiss_modals(self) -> None:
        """Dismiss all modal screens, returning to the main screen."""
        while len(self.screen_stack) > 1:
            self.pop_screen()

    async def _switch_modal(self, screen: Screen) -> None:
        """Switch to a new modal, dismissing any existing modals first."""
        await self._dismiss_modals()
        await self.push_screen(screen)

    async def action_search(self) -> None:
        """Open search modal."""
        if self.client:
            await self._switch_modal(SearchModal(self.client))

    async def action_show_info(self) -> None:
        """Show database info modal."""
        if self.client:
            from haiku.rag.inspector.widgets.info_modal import InfoModal

            await self._switch_modal(InfoModal(self.client, self.db_path))

    async def on_search_modal_chunk_selected(
        self, message: SearchModal.ChunkSelected
    ) -> None:
        """Handle chunk selection from search modal."""
        if not self.client:
            return

        chunk = message.chunk

        # Navigate to the document containing this chunk
        if chunk.document_id:
            doc = await self.client.document_repository.get_by_id(chunk.document_id)
            if doc:
                doc_list = self.query_one(DocumentList)
                chunk_list = self.query_one(ChunkList)

                # Find and select the document
                for idx, d in enumerate(doc_list.documents):
                    if d.id == chunk.document_id:
                        doc_list.list_view.index = idx
                        break

                # Load chunks for this document
                await chunk_list.load_chunks_for_document(
                    self.client, chunk.document_id
                )

                # Wait a tick for the ListView to process the new items
                self.call_after_refresh(self._select_chunk, chunk_list, chunk.id)

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

    async def action_show_visual(self) -> None:
        """Show visual grounding for the currently selected chunk."""
        if not self.client:
            return

        chunk_list = self.query_one(ChunkList)
        idx = chunk_list.list_view.index
        if idx is None or idx >= len(chunk_list.chunks):
            return

        chunk = chunk_list.chunks[idx]

        from haiku.rag.inspector.widgets.visual_modal import VisualGroundingModal

        await self._switch_modal(VisualGroundingModal(chunk=chunk, client=self.client))

    async def action_show_context(self) -> None:
        """Show how the currently selected chunk would be formatted for agents."""
        if not self.client:
            return

        chunk_list = self.query_one(ChunkList)
        idx = chunk_list.list_view.index
        if idx is None or idx >= len(chunk_list.chunks):
            return

        chunk = chunk_list.chunks[idx]

        from haiku.rag.inspector.widgets.context_modal import ContextModal

        await self._switch_modal(ContextModal(chunk=chunk, client=self.client))


def run_inspector(
    db_path: Path | None = None,
    read_only: bool = False,
    before: datetime | None = None,
) -> None:  # pragma: no cover
    """Run the inspector TUI.

    Args:
        db_path: Path to the LanceDB database. If None, uses default from config.
        read_only: Whether to open the database in read-only mode.
        before: Query database as it existed before this datetime.
    """
    config = get_config()
    if db_path is None:
        db_path = config.storage.data_dir / "haiku.rag.lancedb"

    app = InspectorApp(db_path, read_only=read_only, before=before)
    app.run()

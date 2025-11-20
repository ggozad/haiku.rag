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
    from textual.containers import Container
    from textual.widgets import Footer, Header, Input, ListItem, Static

    from haiku.rag.inspector.widgets.chunk_list import ChunkList
    from haiku.rag.inspector.widgets.detail_view import DetailView
    from haiku.rag.inspector.widgets.document_list import DocumentList

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False
    App = object  # type: ignore


class InspectorApp(App):  # type: ignore[misc]
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

    #search-container {
        dock: top;
        height: 3;
        background: $panel;
        display: none;
    }

    #search-container.visible {
        display: block;
    }

    #search-input {
        width: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("/", "search", "Search", show=True),
    ]

    def __init__(self, db_path: Path):
        super().__init__()
        self.db_path = db_path
        self.client: HaikuRAG | None = None
        self.search_visible = False
        self.search_active = False
        # Track current selection for easy restoration
        self.current_document_id: str | None = None
        self.current_chunk_id: str | None = None

    def compose(self) -> "ComposeResult":
        """Compose the UI layout."""
        yield Header()
        with Container(id="search-container"):
            yield Input(placeholder="Search chunks...", id="search-input")
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

        # Select first document and load its chunks
        if doc_list.documents and doc_list.list_view:
            doc_list.list_view.index = 0
            first_doc = doc_list.documents[0]
            self.current_document_id = first_doc.id

            if first_doc.id:
                chunk_list = self.query_one(ChunkList)
                await chunk_list.load_chunks_for_document(self.client, first_doc.id)

                # Select first chunk
                if chunk_list.chunks and chunk_list.list_view:
                    chunk_list.list_view.index = 0
                    self.current_chunk_id = chunk_list.chunks[0].id

        # Focus the document list view
        if doc_list.list_view:
            doc_list.list_view.focus()

    async def on_unmount(self) -> None:
        """Clean up when unmounting."""
        if self.client:
            await self.client.__aexit__(None, None, None)

    async def action_search(self) -> None:
        """Toggle search input visibility."""
        search_container = self.query_one("#search-container", Container)
        search_input = self.query_one("#search-input", Input)

        if self.search_visible:
            search_container.remove_class("visible")
            search_input.value = ""
            self.search_visible = False
            self.search_active = False

            # Restore full document list and reload chunks for selected document
            if self.client:
                doc_list = self.query_one(DocumentList)
                chunk_list = self.query_one(ChunkList)

                # Reload all documents
                await doc_list.load_documents(self.client)

                # Restore document and chunk selection
                if (
                    self.current_document_id
                    and doc_list.list_view
                    and doc_list.documents
                ):
                    # Find and select the document
                    doc_found = False
                    for idx, doc in enumerate(doc_list.documents):
                        if doc.id == self.current_document_id:
                            doc_list.list_view.index = idx
                            doc_found = True
                            break

                    # Reload chunks for this document (without scores)
                    if doc_found:
                        await chunk_list.load_chunks_for_document(
                            self.client, self.current_document_id
                        )

                        # Restore chunk selection and show it in detail view
                        if (
                            self.current_chunk_id
                            and chunk_list.list_view
                            and chunk_list.chunks
                        ):
                            for idx, chunk in enumerate(chunk_list.chunks):
                                if chunk.id == self.current_chunk_id:
                                    chunk_list.list_view.index = idx
                                    # Show the chunk in detail view
                                    detail_view = self.query_one(DetailView)
                                    await detail_view.show_chunk(chunk)
                                    break

                        # Focus back to document list to show selection
                        doc_list.list_view.focus()
        else:
            search_container.add_class("visible")
            search_input.focus()
            self.search_visible = True

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        if event.input.id == "search-input" and self.client:
            query = event.value.strip()
            if query:
                self.search_active = True
                chunk_list = self.query_one(ChunkList)
                await chunk_list.load_chunks_from_search(self.client, query)

                # Get unique document IDs from the search results
                doc_ids = list(
                    {
                        chunk.document_id
                        for chunk in chunk_list.chunks
                        if chunk.document_id
                    }
                )

                # Update document list to show only documents with matching chunks
                if doc_ids:
                    doc_list = self.query_one(DocumentList)
                    documents = []
                    for doc_id in doc_ids:
                        doc = await self.client.document_repository.get_by_id(doc_id)
                        if doc:
                            documents.append(doc)

                    # Update the document list with filtered documents
                    doc_list.documents = documents
                    if doc_list.list_view:
                        await doc_list.list_view.clear()
                        for doc in documents:
                            title = doc.title or doc.uri or doc.id or "Untitled"
                            item = ListItem(Static(f"{title}"))
                            await doc_list.list_view.append(item)

                # Select first chunk and focus the chunk list
                if chunk_list.list_view:
                    if chunk_list.chunks:
                        chunk_list.list_view.index = 0
                    chunk_list.list_view.focus()

    async def on_document_list_document_selected(
        self, message: DocumentList.DocumentSelected
    ) -> None:
        """Handle document selection from document list.

        Args:
            message: Message containing selected document
        """
        if not self.client:
            return

        # Always track current document (even during search)
        self.current_document_id = message.document.id

        # Show document details
        detail_view = self.query_one(DetailView)
        await detail_view.show_document(message.document)

        # Load chunks for this document (but not during search - preserve search results)
        if message.document.id and not self.search_active:
            chunk_list = self.query_one(ChunkList)
            await chunk_list.load_chunks_for_document(self.client, message.document.id)

    async def on_chunk_list_chunk_selected(
        self, message: ChunkList.ChunkSelected
    ) -> None:
        """Handle chunk selection from chunk list.

        Args:
            message: Message containing selected chunk
        """
        # Always track current chunk (even during search)
        self.current_chunk_id = message.chunk.id

        # Show chunk details
        detail_view = self.query_one(DetailView)
        await detail_view.show_chunk(message.chunk)

        # Track the document this chunk belongs to
        if message.chunk.document_id:
            self.current_document_id = message.chunk.document_id


def run_inspector(db_path: Path | None = None) -> None:
    """Run the inspector TUI.

    Args:
        db_path: Path to the LanceDB database. If None, uses default from config.
    """
    config = get_config()
    if db_path is None:
        db_path = config.storage.data_dir / "haiku.rag.lancedb"

    app = InspectorApp(db_path)
    app.run()

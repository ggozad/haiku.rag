from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static
from textual_image.widget import Image as TextualImage

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from haiku.rag.client import HaikuRAG
    from haiku.rag.store.models import Chunk


class VisualGroundingModal(Screen):  # pragma: no cover
    """Modal screen for displaying visual grounding with bounding boxes."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("v", "dismiss", "Close", show=True),
        Binding("left", "prev_page", "Previous Page"),
        Binding("right", "next_page", "Next Page"),
    ]

    CSS = """
    VisualGroundingModal {
        background: $surface;
        layout: vertical;
    }

    #visual-header {
        dock: top;
        height: auto;
        padding: 1;
    }

    #visual-content {
        height: 1fr;
        width: 100%;
        align: center middle;
    }

    #visual-content Image {
        width: auto;
        height: 100%;
    }

    #page-nav {
        dock: bottom;
        height: auto;
        padding: 1;
    }
    """

    def __init__(
        self,
        chunk: "Chunk",
        client: "HaikuRAG",
        document_uri: str | None = None,
    ):
        super().__init__()
        self.chunk = chunk
        self.client = client
        self.document_uri = document_uri or chunk.document_uri
        self.images: list[PILImage] = []
        self.current_page_idx = 0
        self._image_widget: Widget = Static("Loading...", id="image-display")
        self._page_info = Static("", id="page-info")

    def compose(self) -> ComposeResult:
        uri_display = self.document_uri or "Document"
        with Vertical(id="visual-header"):
            yield Static(f"[bold]Visual Grounding[/bold] - {uri_display}")
        with Horizontal(id="visual-content"):
            yield self._image_widget
        with Horizontal(id="page-nav"):
            yield self._page_info

    async def on_mount(self) -> None:
        """Load images and display the first page."""
        self.images = await self.client.visualize_chunk(self.chunk)
        await self._render_current_page()

    async def _render_current_page(self) -> None:
        """Render the current page."""
        if not self.images:
            if isinstance(self._image_widget, Static):
                self._image_widget.update(
                    "[yellow]No page images available[/yellow]\n"
                    "This document was converted without page images."
                )
            self._page_info.update("")
            return

        self._page_info.update(
            f"Page {self.current_page_idx + 1}/{len(self.images)} - Use ←/→ to navigate"
        )

        try:
            image = self.images[self.current_page_idx]
            new_widget = TextualImage(image, id="rendered-image")
            await self._image_widget.remove()
            content = self.query_one("#visual-content", Horizontal)
            await content.mount(new_widget)
            self._image_widget = new_widget
        except Exception as e:
            if isinstance(self._image_widget, Static):
                self._image_widget.update(f"[red]Error: {e}[/red]")

    async def action_dismiss(self, result=None) -> None:
        self.app.pop_screen()

    async def action_prev_page(self) -> None:
        """Navigate to the previous page."""
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            await self._render_current_page()

    async def action_next_page(self) -> None:
        """Navigate to the next page."""
        if self.current_page_idx < len(self.images) - 1:
            self.current_page_idx += 1
            await self._render_current_page()

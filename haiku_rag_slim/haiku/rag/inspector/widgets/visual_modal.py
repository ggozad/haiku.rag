from copy import deepcopy
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static
from textual_image.widget import Image as TextualImage

from haiku.rag.store.models import BoundingBox

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument
    from PIL.Image import Image as PILImage


class VisualGroundingModal(Screen):
    """Modal screen for displaying visual grounding with bounding boxes."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
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
        docling_document: "DoclingDocument",
        bounding_boxes: list[BoundingBox],
        page_numbers: list[int],
        document_uri: str | None = None,
    ):
        super().__init__()
        self.docling_document = docling_document
        self.document_uri = document_uri
        self.bounding_boxes = bounding_boxes
        self.page_numbers = sorted(set(page_numbers)) if page_numbers else []
        self.current_page_idx = 0
        self._image_widget: Widget = Static("Loading...", id="image-display")
        self._page_info = Static("", id="page-info")

    def compose(self) -> ComposeResult:
        uri_display = self.document_uri or self.docling_document.name or "Document"
        with Vertical(id="visual-header"):
            yield Static(f"[bold]Visual Grounding[/bold] - {uri_display}")
        with Horizontal(id="visual-content"):
            yield self._image_widget
        with Horizontal(id="page-nav"):
            yield self._page_info

    async def on_mount(self) -> None:
        """Load and display the first page."""
        await self._render_current_page()

    async def _render_current_page(self) -> None:
        """Render the current page with bounding boxes."""
        if not self.page_numbers:
            if isinstance(self._image_widget, Static):
                self._image_widget.update("[red]No page information available[/red]")
            self._page_info.update("")
            return

        current_page = self.page_numbers[self.current_page_idx]
        self._page_info.update(
            f"Page {current_page} "
            f"({self.current_page_idx + 1}/{len(self.page_numbers)}) "
            f"- Use ←/→ to navigate"
        )

        try:
            image = self._render_page_with_boxes(current_page)
            if image:
                new_widget = TextualImage(image, id="rendered-image")
                await self._image_widget.remove()
                content = self.query_one("#visual-content", Horizontal)
                await content.mount(new_widget)
                self._image_widget = new_widget
            elif isinstance(self._image_widget, Static):
                self._image_widget.update(
                    "[yellow]No page image available[/yellow]\n"
                    "This document was converted without page images."
                )
        except Exception as e:
            if isinstance(self._image_widget, Static):
                self._image_widget.update(f"[red]Error: {e}[/red]")

    def _render_page_with_boxes(self, page_no: int) -> "PILImage | None":
        """Render a page from DoclingDocument with bounding boxes."""
        from PIL import ImageDraw

        # Get the page from DoclingDocument
        if page_no not in self.docling_document.pages:
            return None

        page = self.docling_document.pages[page_no]
        if page.image is None:
            return None

        pil_image = page.image.pil_image
        if pil_image is None:
            return None

        # Get page dimensions
        page_height = page.size.height

        # Calculate scale factor (image pixels vs document coordinates)
        scale_x = pil_image.width / page.size.width
        scale_y = pil_image.height / page.size.height

        # Get bounding boxes for this page
        page_boxes = [bb for bb in self.bounding_boxes if bb.page_no == page_no]

        if page_boxes:
            # Draw bounding boxes
            image = deepcopy(pil_image)
            draw = ImageDraw.Draw(image, "RGBA")

            for bbox in page_boxes:
                # Convert from document coordinates to image coordinates
                # Document coords are typically bottom-left origin
                # PIL uses top-left origin
                x0 = bbox.left * scale_x
                y0 = (page_height - bbox.top) * scale_y  # Flip Y
                x1 = bbox.right * scale_x
                y1 = (page_height - bbox.bottom) * scale_y  # Flip Y

                # Ensure proper ordering (y0 should be less than y1 for PIL)
                if y0 > y1:
                    y0, y1 = y1, y0

                # Draw filled rectangle with transparency
                fill_color = (255, 255, 0, 80)  # Yellow with transparency
                outline_color = (255, 165, 0, 255)  # Orange outline

                draw.rectangle([(x0, y0), (x1, y1)], fill=fill_color, outline=None)
                draw.rectangle([(x0, y0), (x1, y1)], outline=outline_color, width=3)

            return image

        return pil_image

    async def action_dismiss(self, result=None) -> None:
        self.app.pop_screen()

    async def action_prev_page(self) -> None:
        """Navigate to the previous page."""
        if self.current_page_idx > 0:
            self.current_page_idx -= 1
            await self._render_current_page()

    async def action_next_page(self) -> None:
        """Navigate to the next page."""
        if self.current_page_idx < len(self.page_numbers) - 1:
            self.current_page_idx += 1
            await self._render_current_page()

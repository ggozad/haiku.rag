from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static
from textual_image.widget import Image as TextualImage


class DocIndexModal(Screen):  # pragma: no cover
    """Whole-document overlay: all chunk boxes + all mm_asset boxes per page."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("d", "dismiss", "Close", show=True),
        Binding("left", "prev_page", "Previous Page"),
        Binding("right", "next_page", "Next Page"),
        Binding("o", "open_image", "Open image", show=True),
    ]

    CSS = """
    DocIndexModal {
        background: $surface;
        layout: vertical;
    }

    #doc-index-header {
        dock: top;
        height: auto;
        padding: 1;
    }

    #doc-index-content {
        height: 1fr;
        width: 100%;
        align: center middle;
    }

    #doc-index-content Image {
        width: auto;
        height: 100%;
    }

    #doc-index-footer {
        dock: bottom;
        height: auto;
        padding: 1;
    }
    """

    def __init__(
        self,
        *,
        client,
        document_id: str,
        document_uri: str | None = None,
        document_title: str | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.document_id = document_id
        self.document_uri = document_uri
        self.document_title = document_title
        self._image_widget: Widget = Static("Loading...", id="image-display")
        self._page_info = Static("", id="page-info")
        self._out_dir: Path = Path(tempfile.mkdtemp(prefix="haiku-inspector-doc-index-"))
        self._pages: list[tuple[int, Path]] = []  # (page_no, png_path)
        self._page_idx: int = 0

    def compose(self) -> ComposeResult:
        title = self.document_title or self.document_uri or self.document_id
        with Vertical(id="doc-index-header"):
            yield Static(f"[bold]Doc Index Overlay[/bold] - {title}")
            yield Static(f"[dim]saved:[/dim] {self._out_dir}")
        with Horizontal(id="doc-index-content"):
            yield self._image_widget
        with Horizontal(id="doc-index-footer"):
            yield self._page_info

    async def on_mount(self) -> None:
        # Build overlay images once.
        try:
            from PIL import ImageDraw
        except Exception as e:
            if isinstance(self._image_widget, Static):
                self._image_widget.update(f"[red]Pillow required: {e}[/red]")
            return

        doc = await self.client.document_repository.get_by_id(self.document_id)
        if not doc:
            if isinstance(self._image_widget, Static):
                self._image_widget.update("[red]Document not found[/red]")
            return

        docling_doc = doc.get_docling_document()
        if not docling_doc:
            if isinstance(self._image_widget, Static):
                self._image_widget.update("[yellow]No DoclingDocument stored.[/yellow]")
            return

        # Gather chunk bounding boxes grouped by page.
        boxes_by_page_chunks: dict[int, list[tuple[float, float, float, float]]] = {}
        # Load all chunks (paged).
        chunks: list = []
        offset = 0
        batch = 200
        while True:
            part = await self.client.chunk_repository.get_by_document_id(
                self.document_id, limit=batch, offset=offset
            )
            if not part:
                break
            chunks.extend(part)
            offset += len(part)
            if len(part) < batch:
                break

        for ch in chunks:
            meta = ch.get_chunk_metadata()
            for bb in meta.resolve_bounding_boxes(docling_doc):
                boxes_by_page_chunks.setdefault(int(bb.page_no), []).append(
                    (float(bb.left), float(bb.top), float(bb.right), float(bb.bottom))
                )

        # Gather mm_asset bboxes grouped by page.
        boxes_by_page_assets: dict[int, list[tuple[float, float, float, float]]] = {}
        if self.client.store.mm_assets_table is not None:
            rows = (
                self.client.store.mm_assets_table.search()
                .where(f"document_id = '{self.document_id}'")
                .to_list()
            )
            for r in rows:
                page_no = r.get("page_no")
                bbox = r.get("bbox")
                if page_no is None or not bbox:
                    continue
                # bbox is stored as a dict or JSON string
                if isinstance(bbox, str):
                    import json as _json

                    try:
                        bbox = _json.loads(bbox)
                    except Exception:
                        continue
                if not isinstance(bbox, dict):
                    continue
                if not all(k in bbox for k in ("left", "top", "right", "bottom")):
                    continue
                boxes_by_page_assets.setdefault(int(page_no), []).append(
                    (
                        float(bbox["left"]),
                        float(bbox["top"]),
                        float(bbox["right"]),
                        float(bbox["bottom"]),
                    )
                )

        # Render all pages with available images.
        for page_no, page in sorted(docling_doc.pages.items(), key=lambda kv: kv[0]):
            if page.image is None or page.image.pil_image is None or page.size is None:
                continue
            pil = page.image.pil_image
            page_h = float(page.size.height)
            scale_x = pil.width / float(page.size.width)
            scale_y = pil.height / float(page.size.height)

            img = deepcopy(pil)
            draw = ImageDraw.Draw(img, "RGBA")

            # Chunks: yellow/orange
            for left, top, right, bottom in boxes_by_page_chunks.get(int(page_no), []):
                x0 = left * scale_x
                y0 = (page_h - top) * scale_y
                x1 = right * scale_x
                y1 = (page_h - bottom) * scale_y
                if y0 > y1:
                    y0, y1 = y1, y0
                draw.rectangle([(x0, y0), (x1, y1)], fill=(255, 255, 0, 40), outline=None)
                draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 165, 0, 255), width=2)

            # Assets: cyan/blue
            for left, top, right, bottom in boxes_by_page_assets.get(int(page_no), []):
                x0 = left * scale_x
                y0 = (page_h - top) * scale_y
                x1 = right * scale_x
                y1 = (page_h - bottom) * scale_y
                if y0 > y1:
                    y0, y1 = y1, y0
                draw.rectangle([(x0, y0), (x1, y1)], fill=(0, 255, 255, 30), outline=None)
                draw.rectangle([(x0, y0), (x1, y1)], outline=(0, 180, 255, 255), width=2)

            out = self._out_dir / f"page_{int(page_no)}.png"
            try:
                img.save(out, format="PNG")
            except Exception:
                continue
            self._pages.append((int(page_no), out))

        await self._render_current_page()

    async def _render_current_page(self) -> None:
        if not self._pages:
            if isinstance(self._image_widget, Static):
                self._image_widget.update(
                    "[yellow]No page images available.[/yellow]\n"
                    "This document was converted without page images."
                )
            self._page_info.update("")
            return

        page_no, path = self._pages[self._page_idx]
        self._page_info.update(
            f"Page {page_no} ({self._page_idx+1}/{len(self._pages)})  |  "
            f"←/→ navigate  |  o open"
        )

        try:
            from PIL import Image

            pil = Image.open(path)
            new_widget = TextualImage(pil, id="rendered-image")
            await self._image_widget.remove()
            content = self.query_one("#doc-index-content", Horizontal)
            await content.mount(new_widget)
            self._image_widget = new_widget
        except Exception as e:
            if isinstance(self._image_widget, Static):
                self._image_widget.update(f"[red]Error: {e}[/red]")

    async def action_prev_page(self) -> None:
        if self._pages and self._page_idx > 0:
            self._page_idx -= 1
            await self._render_current_page()

    async def action_next_page(self) -> None:
        if self._pages and self._page_idx < len(self._pages) - 1:
            self._page_idx += 1
            await self._render_current_page()

    async def action_open_image(self) -> None:
        if not self._pages:
            return
        _, path = self._pages[self._page_idx]
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception:
            return

    async def action_dismiss(self, result=None) -> None:
        # Best-effort cleanup of temporary images.
        try:
            shutil.rmtree(self._out_dir, ignore_errors=True)
        except Exception:
            pass
        self.app.pop_screen()


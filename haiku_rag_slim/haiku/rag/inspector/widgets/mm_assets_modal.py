from __future__ import annotations

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import ListItem, ListView, Markdown, Static


class MMAssetsModal(Screen):  # pragma: no cover
    """Browse multimodal assets (mm_assets) for a document."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("m", "dismiss", "Close", show=True),
        Binding("o", "open", "Open image", show=True),
        Binding("p", "toggle_mode", "Toggle page/crop", show=True),
    ]

    CSS = """
    MMAssetsModal {
        background: $surface;
        layout: vertical;
    }

    #mm-header {
        dock: top;
        height: auto;
        padding: 1;
    }

    #mm-content {
        height: 1fr;
        width: 100%;
    }

    #mm-list {
        width: 1fr;
        border: solid $primary;
    }

    #mm-detail {
        width: 2fr;
        border: solid $accent;
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
        self.mode: str = "crop"
        self.rows: list[dict] = []
        self._out_dir: Path = Path(tempfile.mkdtemp(prefix="haiku-inspector-mm-assets-"))

    def compose(self) -> ComposeResult:
        title = self.document_title or self.document_uri or self.document_id
        with Vertical(id="mm-header"):
            yield Static(f"[bold]MM Assets[/bold] - {title}")
            yield Static(
                f"[dim]mode:[/dim] {self.mode}  |  "
                f"press [bold]p[/bold] to toggle, [bold]o[/bold] to open image",
                id="mm-status",
            )
            yield Static(f"[dim]saved:[/dim] {self._out_dir}")
        with Horizontal(id="mm-content"):
            with VerticalScroll(id="mm-list"):
                yield ListView(id="mm-assets-list")
            with VerticalScroll(id="mm-detail"):
                yield Markdown("Select an asset…", id="mm-detail-md")

    async def on_mount(self) -> None:
        list_view = self.query_one("#mm-assets-list", ListView)
        await list_view.clear()

        if self.client.store.mm_assets_table is None:
            await self.query_one("#mm-detail-md", Markdown).update(
                "[yellow]mm_assets table is not available.[/yellow]\n\n"
                "Enable multimodal indexing and ingest documents with pictures first."
            )
            return

        # Load rows for this document (simple scan via query).
        self.rows = (
            self.client.store.mm_assets_table.search()
            .where(f"document_id = '{self.document_id}'")
            .to_list()
        )

        if not self.rows:
            await self.query_one("#mm-detail-md", Markdown).update(
                "[yellow]No mm_assets for this document.[/yellow]"
            )
            return

        for r in self.rows:
            asset_id = r.get("id")
            page = r.get("page_no")
            ref = r.get("doc_item_ref")
            label = f"p.{page} {ref} {asset_id}"
            await list_view.append(ListItem(Static(label)))

        list_view.index = 0
        await self._render_detail()
        list_view.focus()

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "mm-assets-list":
            return
        await self._render_detail()

    async def _render_detail(self) -> None:
        list_view = self.query_one("#mm-assets-list", ListView)
        idx = list_view.index
        if idx is None or idx >= len(self.rows):
            return
        r = self.rows[idx]
        asset_id = r.get("id")
        page_no = r.get("page_no")
        ref = r.get("doc_item_ref")
        caption = r.get("caption")
        description = r.get("description")
        md = f"""\
**asset_id**: `{asset_id}`
**page**: {page_no}
**doc_item_ref**: `{ref}`

**caption**:
{caption or "*none*"}

**description**:
{description or "*none*"}

**saved_dir**:
`{self._out_dir}`
"""
        await self.query_one("#mm-detail-md", Markdown).update(md)

    async def action_toggle_mode(self) -> None:
        self.mode = "page" if self.mode == "crop" else "crop"
        self.query_one("#mm-status", Static).update(
            f"[dim]mode:[/dim] {self.mode}  |  "
            f"press [bold]p[/bold] to toggle, [bold]o[/bold] to open image"
        )

    async def action_open(self) -> None:
        list_view = self.query_one("#mm-assets-list", ListView)
        idx = list_view.index
        if idx is None or idx >= len(self.rows):
            return
        asset_id = self.rows[idx].get("id")
        if not asset_id:
            return

        images = await self.client.visualize_mm_asset(asset_id=str(asset_id), mode=self.mode)
        if not images:
            return

        # Save first image and open via OS viewer.
        path = self._out_dir / f"{asset_id}_{self.mode}.png"
        try:
            images[0].save(path, format="PNG")
        except Exception:
            return

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


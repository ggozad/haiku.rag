import json
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from haiku.rag.utils import format_bytes

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG


class InfoModal(ModalScreen):  # pragma: no cover
    """Modal screen for displaying database information."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("i", "dismiss", "Close", show=True),
    ]

    CSS = """
    InfoModal {
        align: center middle;
    }

    #info-container {
        width: 80;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    #info-header {
        height: auto;
        margin-bottom: 1;
    }

    #info-content {
        height: auto;
        max-height: 100%;
    }
    """

    def __init__(self, client: "HaikuRAG", db_path: Path):
        super().__init__()
        self.client = client
        self.db_path = db_path
        self._content_widget = Static("Loading...")

    def compose(self) -> ComposeResult:
        with Vertical(id="info-container"):
            yield Static("[bold]Database Info[/bold]", id="info-header")
            with VerticalScroll(id="info-content"):
                yield self._content_widget

    async def on_mount(self) -> None:
        """Load and display database info."""
        import lancedb

        lines: list[str] = []

        # Path
        lines.append(f"[bold $accent]path[/bold $accent]: {self.db_path}")

        if not self.db_path.exists():
            lines.append("[red]Database path does not exist.[/red]")
            self._content_widget.update("\n".join(lines))
            return

        # Connect to get table info
        try:
            db = lancedb.connect(self.db_path)
            table_names = set(db.table_names())
        except Exception as e:
            lines.append(f"[red]Failed to open database: {e}[/red]")
            self._content_widget.update("\n".join(lines))
            return

        # Get versions
        try:
            ldb_version = pkg_version("lancedb")
        except Exception:
            ldb_version = "unknown"
        try:
            hr_version = pkg_version("haiku.rag-slim")
        except Exception:
            hr_version = "unknown"
        try:
            docling_version = pkg_version("docling")
        except Exception:
            docling_version = "unknown"

        # Get stats from store
        table_stats = self.client.store.get_stats()

        # Read settings
        stored_version = "unknown"
        embed_provider: str | None = None
        embed_model: str | None = None
        vector_dim: int | None = None

        if "settings" in table_names:
            settings_tbl = db.open_table("settings")
            arrow = settings_tbl.search().where("id = 'settings'").limit(1).to_arrow()
            rows = arrow.to_pylist() if arrow is not None else []
            if rows:
                raw = rows[0].get("settings") or "{}"
                data = json.loads(raw) if isinstance(raw, str) else (raw or {})
                stored_version = str(data.get("version", stored_version))
                embeddings = data.get("embeddings", {})
                embed_model_obj = embeddings.get("model", {})
                embed_provider = embed_model_obj.get("provider")
                embed_model = embed_model_obj.get("name")
                vector_dim = embed_model_obj.get("vector_dim")

        num_docs = table_stats["documents"].get("num_rows", 0)
        doc_bytes = table_stats["documents"].get("total_bytes", 0)

        num_chunks = table_stats["chunks"].get("num_rows", 0)
        chunk_bytes = table_stats["chunks"].get("total_bytes", 0)

        has_vector_index = table_stats["chunks"].get("has_vector_index", False)
        num_indexed_rows = table_stats["chunks"].get("num_indexed_rows", 0)
        num_unindexed_rows = table_stats["chunks"].get("num_unindexed_rows", 0)

        # Table versions
        doc_versions = (
            len(list(db.open_table("documents").list_versions()))
            if "documents" in table_names
            else 0
        )
        chunk_versions = (
            len(list(db.open_table("chunks").list_versions()))
            if "chunks" in table_names
            else 0
        )

        # Build output
        lines.append(
            f"[bold $accent]haiku.rag version (db)[/bold $accent]: {stored_version}"
        )

        if embed_provider or embed_model or vector_dim:
            provider_part = embed_provider or "unknown"
            model_part = embed_model or "unknown"
            dim_part = f"{vector_dim}" if vector_dim is not None else "unknown"
            lines.append(
                f"[bold $accent]embeddings[/bold $accent]: "
                f"{provider_part}/{model_part} (dim: {dim_part})"
            )
        else:
            lines.append("[bold $accent]embeddings[/bold $accent]: unknown")

        lines.append(
            f"[bold $accent]documents[/bold $accent]: {num_docs} ({format_bytes(doc_bytes)})"
        )
        lines.append(
            f"[bold $accent]chunks[/bold $accent]: {num_chunks} ({format_bytes(chunk_bytes)})"
        )

        # Vector index info
        if has_vector_index:
            lines.append("[bold $accent]vector index[/bold $accent]: ✓ exists")
            lines.append(
                f"[bold $accent]indexed chunks[/bold $accent]: {num_indexed_rows}"
            )
            if num_unindexed_rows > 0:
                lines.append(
                    f"[bold $accent]unindexed chunks[/bold $accent]: [yellow]{num_unindexed_rows}[/yellow]"
                )
            else:
                lines.append(
                    f"[bold $accent]unindexed chunks[/bold $accent]: {num_unindexed_rows}"
                )
        else:
            if num_chunks >= 256:
                lines.append(
                    "[bold $accent]vector index[/bold $accent]: [yellow]✗ not created[/yellow]"
                )
            else:
                lines.append(
                    f"[bold $accent]vector index[/bold $accent]: ✗ not created "
                    f"(need {256 - num_chunks} more chunks)"
                )

        lines.append(
            f"[bold $accent]versions (documents)[/bold $accent]: {doc_versions}"
        )
        lines.append(
            f"[bold $accent]versions (chunks)[/bold $accent]: {chunk_versions}"
        )

        lines.append("")
        lines.append("[bold]Versions[/bold]")
        lines.append(f"[bold $accent]haiku.rag[/bold $accent]: {hr_version}")
        lines.append(f"[bold $accent]lancedb[/bold $accent]: {ldb_version}")
        lines.append(f"[bold $accent]docling[/bold $accent]: {docling_version}")

        self._content_widget.update("\n".join(lines))

    async def action_dismiss(self, result=None) -> None:
        self.app.pop_screen()

import asyncio
from typing import TYPE_CHECKING

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from haiku.rag.utils import format_bytes, get_package_versions

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG


async def database_lines(client: "HaikuRAG") -> list[str]:
    """What one database reports about itself, without naming its location.

    Reported through the connection the client already holds. A failure becomes
    a line of the report, and the other databases still report.
    """
    from haiku.rag.store.engine import ConnectionMode
    from haiku.rag.store.info import get_database_stats

    lines: list[str] = []
    db_path = client.store.db_path

    if client.store._connection_mode == ConnectionMode.LOCAL and not db_path.exists():
        return ["[red]Database path does not exist.[/red]"]

    try:
        db = client.store.db
        stats = await get_database_stats(db)
    except Exception as e:
        return [f"[red]Failed to open database: {escape(str(e))}[/red]"]

    # The store read and parsed these on open.
    settings = client.store.stored_settings
    stored_version = escape(str(settings.get("version", "unknown")))
    embed_model_obj = settings.get("embeddings", {}).get("model", {})
    embed_provider = embed_model_obj.get("provider")
    embed_model = embed_model_obj.get("name")
    vector_dim = embed_model_obj.get("vector_dim")

    num_docs = stats["documents"].get("num_rows", 0)
    num_chunks = stats["chunks"].get("num_rows", 0)
    has_vector_index = stats["chunks"].get("has_vector_index", False)
    num_unindexed_rows = stats["chunks"].get("num_unindexed_rows", 0)

    lines.append(
        f"[bold $accent]haiku.rag version (db)[/bold $accent]: {stored_version}"
    )

    if embed_provider or embed_model or vector_dim:
        provider_part = escape(str(embed_provider or "unknown"))
        model_part = escape(str(embed_model or "unknown"))
        dim_part = escape(str(vector_dim)) if vector_dim is not None else "unknown"
        lines.append(
            f"[bold $accent]embeddings[/bold $accent]: "
            f"{provider_part}/{model_part} (dim: {dim_part})"
        )
    else:
        lines.append("[bold $accent]embeddings[/bold $accent]: unknown")

    lines.append(
        f"[bold $accent]documents[/bold $accent]: {num_docs} "
        f"({format_bytes(stats['documents'].get('total_bytes', 0))})"
    )
    lines.append(
        f"[bold $accent]document_meta[/bold $accent]: "
        f"{stats['document_meta'].get('num_rows', 0)} "
        f"({format_bytes(stats['document_meta'].get('total_bytes', 0))})"
    )
    lines.append(
        f"[bold $accent]chunks[/bold $accent]: {num_chunks} "
        f"({format_bytes(stats['chunks'].get('total_bytes', 0))})"
    )

    if has_vector_index:
        lines.append("[bold $accent]vector index[/bold $accent]: ✓ exists")
        lines.append(
            f"[bold $accent]indexed chunks[/bold $accent]: "
            f"{stats['chunks'].get('num_indexed_rows', 0)}"
        )
        colour = "[yellow]" if num_unindexed_rows > 0 else ""
        close = "[/yellow]" if num_unindexed_rows > 0 else ""
        lines.append(
            f"[bold $accent]unindexed chunks[/bold $accent]: "
            f"{colour}{num_unindexed_rows}{close}"
        )
    elif num_chunks >= 256:
        lines.append(
            "[bold $accent]vector index[/bold $accent]: [yellow]✗ not created[/yellow]"
        )
    else:
        lines.append(
            f"[bold $accent]vector index[/bold $accent]: ✗ not created "
            f"(need {256 - num_chunks} more chunks)"
        )

    for table in ("documents", "document_meta", "chunks"):
        lines.append(
            f"[bold $accent]versions ({table})[/bold $accent]: "
            f"{stats[table].get('num_versions', 0)}"
        )
    lines.append("")
    return lines


class InfoModal(ModalScreen):
    """Modal screen for displaying database information."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("i", "dismiss", "Close", show=True),
    ]

    CSS = """
    InfoModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.5);
    }

    #info-container {
        width: auto;
        min-width: 40;
        max-width: 80;
        height: auto;
        max-height: 20;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
    }

    #info-header {
        height: auto;
        margin-bottom: 1;
    }

    #info-content {
        height: 1fr;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self, client: "HaikuRAG"):
        super().__init__()
        self.client = client
        self._content_widget = Static("Loading...")

    def compose(self) -> ComposeResult:
        with Vertical(id="info-container"):
            yield Static("[bold]Database Info[/bold]", id="info-header")
            with VerticalScroll(id="info-content"):
                yield self._content_widget

    async def on_mount(self) -> None:
        """Load and display database info."""
        lines: list[str] = []

        location = self.client.location
        if location is None:
            # Each database reports independently, failures included. Names
            # only: a location belongs in the configuration.
            blocks = await asyncio.gather(
                *(self._report(name) for name in sorted(self.client.source_names))
            )
            for block in blocks:
                lines.extend(block)
        else:
            lines.append(
                f"[bold $accent]location[/bold $accent]: {escape(str(location))}"
            )
            lines.extend(await database_lines(self.client))

        lines.append("[bold]Versions[/bold]")
        versions = {
            key: escape(str(value)) for key, value in get_package_versions().items()
        }
        lines.append(f"[bold $accent]haiku.rag[/bold $accent]: {versions['haiku_rag']}")
        lines.append(f"[bold $accent]lancedb[/bold $accent]: {versions['lancedb']}")
        lines.append(f"[bold $accent]docling[/bold $accent]: {versions['docling']}")
        lines.append(
            f"[bold $accent]pydantic-ai[/bold $accent]: {versions['pydantic_ai']}"
        )
        lines.append(
            f"[bold $accent]docling-document schema[/bold $accent]: "
            f"{versions['docling_document_schema']}"
        )

        self._content_widget.update("\n".join(lines))

    async def _report(self, name: str) -> list[str]:
        """One database's block, including its own failure to open."""
        lines = [f"[bold]{escape(name)}[/bold]"]
        try:
            (owner,) = await self.client.clients_for([name])
        except Exception as e:
            # The client names a configured database by name and never by
            # location, so its message is safe to show.
            return [*lines, f"[red]{escape(str(e))}[/red]", ""]
        return [*lines, *await database_lines(owner)]

    async def action_dismiss(self, result=None) -> None:
        self.app.pop_screen()

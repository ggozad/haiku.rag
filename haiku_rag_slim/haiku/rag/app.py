import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TransferSpeedColumn,
)

from haiku.rag.client import HaikuRAG, RebuildMode
from haiku.rag.config import AppConfig, Config
from haiku.rag.mcp import create_mcp_server
from haiku.rag.store.models.chunk import SearchType
from haiku.rag.store.models.document import Document

if TYPE_CHECKING:
    from haiku.rag.store.models import SearchResult
from haiku.rag.utils import format_bytes, format_citations_rich

logger = logging.getLogger(__name__)


class HaikuRAGApp:  # pragma: no cover
    def __init__(
        self,
        db_path: Path,
        config: AppConfig = Config,
        read_only: bool = False,
        before: datetime | None = None,
    ):
        self.db_path = db_path
        self.config = config
        self.read_only = read_only
        self.before = before
        self.console = Console()

        from haiku.rag.store.engine import ConnectionMode

        self._is_local = ConnectionMode.from_config(self.config) == ConnectionMode.LOCAL
        self._display_path = self.db_path if self._is_local else self.config.lancedb.uri

    async def init(self):
        """Initialize a new database."""
        if self._is_local and self.db_path.exists():
            self.console.print(
                f"[yellow]Database already exists at {self.db_path}[/yellow]"
            )
            return

        # Create the database
        async with HaikuRAG(db_path=self.db_path, config=self.config, create=True):
            pass
        self.console.print(
            f"[bold green]Database initialized at {self._display_path}[/bold green]"
        )

    async def info(self):
        """Display read-only information about the database without modifying it."""

        from haiku.rag.store.engine import gather_database_info

        if self.before is not None:
            self.console.print(
                "[yellow]Note: --before is not supported by info; showing current state.[/yellow]"
            )

        # Basic: show path/URI
        self.console.print("[bold]haiku.rag database info[/bold]")
        self.console.print(
            f"  [repr.attrib_name]path[/repr.attrib_name]: {self._display_path}"
        )

        if self._is_local and not self.db_path.exists():
            self.console.print("[red]Database path does not exist.[/red]")
            return

        info = await gather_database_info(self.config, self.db_path)

        if not info.exists:
            self.console.print(
                "[red]Database is empty. Use 'haiku-rag init' to initialize.[/red]"
            )
            return

        self.console.print(
            f"  [repr.attrib_name]haiku.rag version (db)[/repr.attrib_name]: {info.stored_version}"
        )
        dim_part = (
            f"{info.embeddings.vector_dim}"
            if info.embeddings.vector_dim is not None
            else "unknown"
        )
        self.console.print(
            "  [repr.attrib_name]embeddings[/repr.attrib_name]: "
            f"{info.embeddings.provider}/{info.embeddings.name} (dim: {dim_part})"
        )

        tables = {t.name: t for t in info.tables}

        # Per-table row counts and sizes. Missing required tables are
        # reported as "absent" rather than raising.
        for name in ("documents", "document_meta", "chunks", "document_items"):
            entry = tables[name]
            if entry.exists:
                self.console.print(
                    f"  [repr.attrib_name]{name}[/repr.attrib_name]: {entry.num_rows} "
                    f"({format_bytes(entry.total_bytes)})"
                )
            else:
                self.console.print(
                    f"  [repr.attrib_name]{name}[/repr.attrib_name]: [yellow]absent[/yellow]"
                )

        # Vector index information
        if tables["chunks"].exists:
            num_chunks = tables["chunks"].num_rows
            if info.vector_index.exists:
                self.console.print(
                    "  [repr.attrib_name]vector index[/repr.attrib_name]: ✓ exists"
                )
                self.console.print(
                    f"  [repr.attrib_name]indexed chunks[/repr.attrib_name]: {info.vector_index.indexed_rows}"
                )
                if info.vector_index.unindexed_rows > 0:
                    self.console.print(
                        f"  [repr.attrib_name]unindexed chunks[/repr.attrib_name]: [yellow]{info.vector_index.unindexed_rows}[/yellow] "
                        "(consider running: haiku-rag create-index)"
                    )
                else:
                    self.console.print(
                        f"  [repr.attrib_name]unindexed chunks[/repr.attrib_name]: {info.vector_index.unindexed_rows}"
                    )
            else:
                if num_chunks >= 256:
                    self.console.print(
                        "  [repr.attrib_name]vector index[/repr.attrib_name]: [yellow]✗ not created[/yellow] "
                        "(run: haiku-rag create-index)"
                    )
                else:
                    self.console.print(
                        f"  [repr.attrib_name]vector index[/repr.attrib_name]: ✗ not created "
                        f"(need {256 - num_chunks} more chunks)"
                    )

        if tables["documents"].exists:
            self.console.print(
                f"  [repr.attrib_name]versions (documents)[/repr.attrib_name]: "
                f"{tables['documents'].num_versions}"
            )
        if tables["document_meta"].exists:
            self.console.print(
                f"  [repr.attrib_name]versions (document_meta)[/repr.attrib_name]: "
                f"{tables['document_meta'].num_versions}"
            )
        if tables["chunks"].exists:
            self.console.print(
                f"  [repr.attrib_name]versions (chunks)[/repr.attrib_name]: "
                f"{tables['chunks'].num_versions}"
            )

        # Migration status
        self.console.rule()
        if info.pending_migrations:
            self.console.print(
                f"[bold yellow]{len(info.pending_migrations)} migration(s) pending.[/bold yellow] "
                "Run [cyan]haiku-rag migrate[/cyan] to upgrade."
            )
            for step in info.pending_migrations:
                self.console.print(
                    f"  [yellow]→[/yellow] {step.version}: {step.description}"
                )
        else:
            self.console.print("[green]Database is up to date.[/green]")

        self.console.rule()
        self.console.print("[bold]Versions[/bold]")
        self.console.print(
            f"  [repr.attrib_name]haiku.rag[/repr.attrib_name]: {info.packages['haiku_rag']}"
        )
        self.console.print(
            f"  [repr.attrib_name]lancedb[/repr.attrib_name]: {info.packages['lancedb']}"
        )
        self.console.print(
            f"  [repr.attrib_name]docling[/repr.attrib_name]: {info.packages['docling']}"
        )
        self.console.print(
            f"  [repr.attrib_name]pydantic-ai[/repr.attrib_name]: {info.packages['pydantic_ai']}"
        )
        self.console.print(
            f"  [repr.attrib_name]docling-document schema[/repr.attrib_name]: {info.packages['docling_document_schema']}"
        )

    async def doctor(self) -> bool:
        """Run health checks and print a report. Returns True if any check failed."""
        import os

        from haiku.rag.doctor import Severity, run_doctor

        self.console.print("[bold]haiku.rag doctor[/bold]")
        self.console.print(
            f"  [repr.attrib_name]path[/repr.attrib_name]: {self._display_path}"
        )

        if self._is_local and not self.db_path.exists():
            self.console.print("[red]Database path does not exist.[/red]")
            return True

        report = await run_doctor(self.config, self.db_path, dict(os.environ))

        glyphs = {
            Severity.OK: "[green]✓[/green]",
            Severity.WARN: "[yellow]![/yellow]",
            Severity.FAIL: "[red]✗[/red]",
        }
        self.console.rule()
        for result in report.results:
            self.console.print(f"{glyphs[result.severity]} {result.message}")
            for detail in result.details:
                self.console.print(f"    [dim]{detail}[/dim]")
            if result.remediation:
                self.console.print(f"    [dim]→ {result.remediation}[/dim]")

        self.console.rule()
        self.console.print(
            f"[green]{report.count(Severity.OK)} ok[/green], "
            f"[yellow]{report.count(Severity.WARN)} warning(s)[/yellow], "
            f"[red]{report.count(Severity.FAIL)} failure(s)[/red]"
        )
        return report.failed

    async def history(self, table: str | None = None, limit: int | None = None):
        """Display version history for database tables.

        Args:
            table: Specific table to show history for (documents, chunks, settings).
                   If None, shows history for all tables.
            limit: Maximum number of versions to show per table.
        """
        from haiku.rag.store.engine import Store

        if self._is_local and not self.db_path.exists():
            self.console.print("[red]Database path does not exist.[/red]")
            return

        async with Store(
            self.db_path,
            config=self.config,
            skip_validation=True,
            read_only=True,
            skip_migration_check=True,
            before=self.before,
        ) as store:
            tables = [
                "documents",
                "document_meta",
                "chunks",
                "document_items",
                "settings",
            ]
            if table:
                if table not in tables:
                    self.console.print(
                        f"[red]Unknown table: {table}. Must be one of: {', '.join(tables)}[/red]"
                    )
                    return
                tables = [table]

            self.console.print("[bold]Version History[/bold]")

            for table_name in tables:
                versions = await store.list_table_versions(table_name)

                # Sort by version descending (newest first)
                versions = sorted(versions, key=lambda v: v["version"], reverse=True)

                if limit:
                    versions = versions[:limit]

                self.console.print(f"\n[bold cyan]{table_name}[/bold cyan]")

                if not versions:
                    self.console.print("  [dim]No versions found[/dim]")
                    continue

                for v in versions:
                    version_num = v["version"]
                    timestamp = v["timestamp"]
                    self.console.print(
                        f"  [repr.attrib_name]v{version_num}[/repr.attrib_name]: {timestamp}"
                    )

    async def list_documents(self, filter: str | None = None):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=True,
            before=self.before,
        ) as self.client:
            documents = await self.client.list_documents(filter=filter)
            for doc in documents:
                self._rich_print_document(doc, truncate=True)

    async def add_document_from_text(
        self, text: str, title: str | None = None, metadata: dict | None = None
    ):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ) as self.client:
            doc = await self.client.create_document(
                text, title=title, metadata=metadata
            )
            self._rich_print_document(doc, truncate=True)
            self.console.print(
                f"[bold green]Document {doc.id} added successfully.[/bold green]"
            )

    async def add_document_from_source(
        self, source: str, title: str | None = None, metadata: dict | None = None
    ):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ) as self.client:
            result = await self.client.create_document_from_source(
                source, title=title, metadata=metadata
            )
            if isinstance(result, list):
                for doc in result:
                    self._rich_print_document(doc, truncate=True)
                self.console.print(
                    f"[bold green]{len(result)} documents added successfully.[/bold green]"
                )
            else:
                self._rich_print_document(result, truncate=True)
                self.console.print(
                    f"[bold green]Document {result.id} added successfully.[/bold green]"
                )

    async def get_document(self, doc_id: str):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=True,
            before=self.before,
        ) as self.client:
            doc = await self.client.get_document_by_id(doc_id)
            if doc is None:
                self.console.print(f"[red]Document with id {doc_id} not found.[/red]")
                return
            self._rich_print_document(doc, truncate=False)

    async def delete_document(self, doc_id: str):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ) as self.client:
            deleted = await self.client.delete_document(doc_id)
            if deleted:
                self.console.print(
                    f"[bold green]Document {doc_id} deleted successfully.[/bold green]"
                )
            else:
                self.console.print(
                    f"[yellow]Document with id {doc_id} not found.[/yellow]"
                )

    async def search(
        self,
        query: str | None = None,
        limit: int | None = None,
        filter: str | None = None,
        search_type: SearchType | None = None,
        image: Path | None = None,
    ):
        if query is None and image is None:
            self.console.print(
                "[red]Provide either a query argument or --image PATH.[/red]"
            )
            return
        if query is not None and image is not None:
            self.console.print("[red]Pass either a query or --image, not both.[/red]")
            return

        if query is None and search_type is not None:
            self.console.print("[red]Pass --search-type only for text queries[/red]")
            return

        search_input: str | bytes
        if image is not None:
            search_input = image.read_bytes()
        else:
            assert query is not None
            search_input = query

        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=True,
            before=self.before,
        ) as self.client:
            results = await self.client.search(
                search_input,
                limit=limit,
                filter=filter,
                search_type=search_type,
            )
            if not results:
                self.console.print("[yellow]No results found.[/yellow]")
                return
            for result in results:
                self._rich_print_search_result(result)

    async def visualize_chunk(self, chunk_id: str):
        """Display visual grounding images for a chunk."""
        from textual_image.renderable import Image as RichImage

        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=True,
            before=self.before,
        ) as self.client:
            chunk = await self.client.get_chunk_by_id(chunk_id)
            if not chunk:
                self.console.print(f"[red]Chunk with id {chunk_id} not found.[/red]")
                return

            images = await self.client.visualize_chunk(chunk)
            if not images:
                self.console.print(
                    "[yellow]No visual grounding available for this chunk.[/yellow]"
                )
                self.console.print(
                    "This may be because the document was converted without page images."
                )
                return

            self.console.print(f"[bold]Visual grounding for chunk {chunk_id}[/bold]")
            if chunk.document_uri:
                self.console.print(
                    f"[repr.attrib_name]document[/repr.attrib_name]: {chunk.document_uri}"
                )

            for i, img in enumerate(images):
                self.console.print(
                    f"\n[bold cyan]Page {i + 1}/{len(images)}[/bold cyan]"
                )
                self.console.print(RichImage(img))

    async def ask(
        self,
        question: str,
        filter: str | None = None,
    ):
        """Ask a question using the RAG system.

        Args:
            question: The question to ask
            filter: SQL WHERE clause to filter documents
        """
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=True,
            before=self.before,
        ) as self.client:
            answer, citations = await self.client.ask(question, filter=filter)

            self.console.print(f"[bold blue]Question:[/bold blue] {question}")
            self.console.print()
            self.console.print("[bold green]Answer:[/bold green]")
            self.console.print(Markdown(answer))
            for renderable in await format_citations_rich(
                citations, client=self.client
            ):
                self.console.print(renderable)

    async def analyze(
        self,
        question: str,
        filter: str | None = None,
    ):
        """Answer a question using the rag-analysis skill.

        Args:
            question: The question to answer
            filter: SQL WHERE clause to filter documents
        """
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=True,
            before=self.before,
        ) as self.client:
            self.console.print(f"[bold blue]Question:[/bold blue] {question}")
            self.console.print()
            self.console.print(
                "[dim]Running analysis skill with code execution...[/dim]"
            )
            self.console.print()

            result = await self.client.analyze(question, filter=filter)

            self.console.print("[bold green]Answer:[/bold green]")
            self.console.print(Markdown(result.answer))
            for renderable in await format_citations_rich(
                result.citations, client=self.client
            ):
                self.console.print(renderable)

    async def rebuild(self, mode: RebuildMode = RebuildMode.FULL):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            skip_validation=True,
            read_only=self.read_only,
            before=self.before,
        ) as client:
            if mode == RebuildMode.SET_EMBEDDER:
                async for _ in client.rebuild_database(mode=mode):
                    pass
                self.console.print(
                    "[bold green]Stored embedder settings updated.[/bold green]"
                )
                return

            documents = await client.list_documents()
            total_docs = len(documents)

            if total_docs == 0:
                self.console.print("[yellow]No documents found in database.[/yellow]")
                return

            mode_desc = {
                RebuildMode.FULL: "full rebuild",
                RebuildMode.RECHUNK: "rechunk",
                RebuildMode.EMBED_ONLY: "embed only",
                RebuildMode.TITLE_ONLY: "title only",
                RebuildMode.DESCRIPTIONS: "picture descriptions",
            }[mode]

            self.console.print(
                f"[bold cyan]Rebuilding database ({mode_desc}) with {total_docs} documents...[/bold cyan]"
            )
            with Progress() as progress:
                task = progress.add_task("Rebuilding...", total=total_docs)
                async for _ in client.rebuild_database(mode=mode):
                    progress.update(task, advance=1)

            self.console.print(
                "[bold green]Database rebuild completed successfully.[/bold green]"
            )

    async def vacuum(self):
        """Run database maintenance: optimize and cleanup table history."""
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            skip_validation=True,
            read_only=self.read_only,
            before=self.before,
        ) as client:
            await client.vacuum()
        self.console.print("[bold green]Vacuum completed successfully.[/bold green]")

    async def migrate(self) -> list[str]:
        """Run pending database migrations.

        Returns:
            List of descriptions of applied migrations.
        """
        from haiku.rag.store.engine import Store

        async with Store(
            self.db_path,
            config=self.config,
            skip_validation=True,
            skip_migration_check=True,
            read_only=self.read_only,
        ) as store:
            return await store.migrate()

    async def create_index(self):
        """Create vector index on the chunks table."""
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            skip_validation=True,
            read_only=self.read_only,
            before=self.before,
        ) as client:
            row_count = await client.store.chunks_table.count_rows()
            self.console.print(f"Chunks in database: {row_count}")

            if row_count < 256:
                self.console.print(
                    f"[yellow]Warning: Need at least 256 chunks to create an index (have {row_count})[/yellow]"
                )
                return

            # Check if index already exists
            indices = await client.store.chunks_table.list_indices()
            has_vector_index = any("vector" in str(idx).lower() for idx in indices)

            if has_vector_index:
                self.console.print(
                    "[yellow]Rebuilding existing vector index...[/yellow]"
                )
            else:
                self.console.print("[bold]Creating vector index...[/bold]")

            await client.store._ensure_vector_index()
            self.console.print(
                "[bold green]Vector index created successfully.[/bold green]"
            )

    async def download_models(self):
        """Download Docling, HuggingFace tokenizer, and Ollama models per config."""
        from haiku.rag.client.downloads import download_models

        progress: Progress | None = None
        task_id: TaskID | None = None
        current_model = ""
        current_digest = ""

        async for event in download_models(self.config):
            if event.status == "start":
                self.console.print(
                    f"[bold blue]Downloading {event.model}...[/bold blue]"
                )
            elif event.status == "done":
                if progress:
                    progress.stop()
                    progress = None
                    task_id = None
                self.console.print(f"[green]✓[/green] {event.model}")
                current_model = ""
                current_digest = ""
            elif event.status == "pulling":
                self.console.print(f"[bold blue]Pulling {event.model}...[/bold blue]")
                current_model = event.model
                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    console=self.console,
                    transient=True,
                    auto_refresh=False,
                )
                progress.start()
                task_id = progress.add_task(event.model, total=None)
            elif event.status == "downloading" and progress and task_id is not None:
                if event.digest != current_digest:
                    current_digest = event.digest
                    short_digest = event.digest[:19] if event.digest else ""
                    progress.update(
                        task_id,
                        description=f"{current_model} ({short_digest})",
                        total=event.total,
                        completed=0,
                    )
                progress.update(task_id, completed=event.completed, refresh=True)
            elif progress and task_id is not None:
                progress.update(
                    task_id,
                    description=f"{current_model}: {event.status}",
                    refresh=True,
                )

    def show_settings(self):
        """Display current configuration settings."""
        self.console.print("[bold]haiku.rag configuration[/bold]")
        self.console.print()

        # Get all config fields dynamically
        for field_name, field_value in self.config.model_dump().items():
            # Format the display value
            if isinstance(field_value, str) and (
                "key" in field_name.lower()
                or "password" in field_name.lower()
                or "token" in field_name.lower()
            ):
                # Hide sensitive values but show if they're set
                display_value = "✓ Set" if field_value else "✗ Not set"
            else:
                display_value = field_value

            self.console.print(
                f"  [repr.attrib_name]{field_name}[/repr.attrib_name]: {display_value}"
            )

    def _rich_print_document(self, doc: Document, truncate: bool = False):
        """Format a document for display."""
        if truncate:
            content = doc.content.splitlines()
            if len(content) > 3:
                content = content[:3] + ["\n…"]
            content = "\n".join(content)
            content = Markdown(content)
        else:
            content = Markdown(doc.content)
        title_part = (
            f" [repr.attrib_name]title[/repr.attrib_name]: {doc.title}"
            if doc.title
            else ""
        )
        self.console.print(
            f"[repr.attrib_name]id[/repr.attrib_name]: {doc.id} "
            f"[repr.attrib_name]uri[/repr.attrib_name]: {doc.uri}"
            + title_part
            + f" [repr.attrib_name]meta[/repr.attrib_name]: {doc.metadata}"
        )
        self.console.print(
            f"[repr.attrib_name]created at[/repr.attrib_name]: {doc.created_at} [repr.attrib_name]updated at[/repr.attrib_name]: {doc.updated_at}"
        )
        self.console.print("[repr.attrib_name]content[/repr.attrib_name]:")
        self.console.print(content)
        self.console.rule()

    def _rich_print_search_result(self, result: "SearchResult"):
        """Format a search result for display."""
        content = Markdown(result.content)
        self.console.print(
            f"[repr.attrib_name]document_id[/repr.attrib_name]: {result.document_id} "
            f"[repr.attrib_name]chunk_id[/repr.attrib_name]: {result.chunk_id} "
            f"[repr.attrib_name]score[/repr.attrib_name]: {result.score:.4f}"
        )
        if result.document_uri:
            self.console.print(
                f"[repr.attrib_name]document uri[/repr.attrib_name]: {result.document_uri}"
            )
        if result.document_title:
            self.console.print("[repr.attrib_name]document title[/repr.attrib_name]:")
            self.console.print(result.document_title)
        if result.page_numbers:
            self.console.print("[repr.attrib_name]pages[/repr.attrib_name]:")
            self.console.print(", ".join(str(p) for p in result.page_numbers))
        if result.headings:
            self.console.print("[repr.attrib_name]headings[/repr.attrib_name]:")
            self.console.print(" > ".join(result.headings))
        self.console.print("[repr.attrib_name]content[/repr.attrib_name]:")
        self.console.print(content)
        self.console.rule()

    async def run_mcp(
        self,
        transport: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8001,
    ):
        """Run the MCP server until interrupted."""
        async with HaikuRAG(
            self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ):
            server = create_mcp_server(
                self.db_path, config=self.config, read_only=self.read_only
            )
            try:
                if transport == "stdio":
                    await server.run_stdio_async()
                else:
                    logger.info(f"Starting MCP server on {host}:{port}")
                    await server.run_http_async(
                        transport="streamable-http", host=host, port=port
                    )
            except KeyboardInterrupt:
                pass

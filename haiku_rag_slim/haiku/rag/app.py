import asyncio
import json
import logging
from datetime import datetime
from importlib.metadata import version as pkg_version
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
from haiku.rag.graph.agui import AGUIConsoleRenderer, stream_graph
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.graph import build_research_graph
from haiku.rag.graph.research.state import ResearchDeps, ResearchState
from haiku.rag.mcp import create_mcp_server
from haiku.rag.monitor import FileWatcher
from haiku.rag.store.models.document import Document

if TYPE_CHECKING:
    from haiku.rag.store.models import SearchResult
from haiku.rag.utils import format_bytes, format_citations_rich

logger = logging.getLogger(__name__)


class HaikuRAGApp:
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

    async def init(self):
        """Initialize a new database."""
        if self.db_path.exists():
            self.console.print(
                f"[yellow]Database already exists at {self.db_path}[/yellow]"
            )
            return

        # Create the database
        client = HaikuRAG(db_path=self.db_path, config=self.config, create=True)
        client.close()
        self.console.print(
            f"[bold green]Database initialized at {self.db_path}[/bold green]"
        )

    async def info(self):
        """Display read-only information about the database without modifying it."""

        import lancedb

        # Basic: show path
        self.console.print("[bold]haiku.rag database info[/bold]")
        self.console.print(
            f"  [repr.attrib_name]path[/repr.attrib_name]: {self.db_path}"
        )

        if not self.db_path.exists():
            self.console.print("[red]Database path does not exist.[/red]")
            return

        # Connect without going through Store to avoid upgrades/validation writes
        try:
            db = lancedb.connect(self.db_path)
            table_names = set(db.table_names())
        except Exception as e:
            self.console.print(f"[red]Failed to open database: {e}[/red]")
            return

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

        # Get comprehensive table statistics (this also runs migrations)
        from haiku.rag.store.engine import Store

        store = Store(self.db_path, config=self.config, skip_validation=True)
        table_stats = store.get_stats()

        # Read settings after Store init (migrations have run)
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

        store.close()

        num_docs = table_stats["documents"].get("num_rows", 0)
        doc_bytes = table_stats["documents"].get("total_bytes", 0)

        num_chunks = table_stats["chunks"].get("num_rows", 0)
        chunk_bytes = table_stats["chunks"].get("total_bytes", 0)

        has_vector_index = table_stats["chunks"].get("has_vector_index", False)
        num_indexed_rows = table_stats["chunks"].get("num_indexed_rows", 0)
        num_unindexed_rows = table_stats["chunks"].get("num_unindexed_rows", 0)

        # Table versions per table (direct API)
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

        self.console.print(
            f"  [repr.attrib_name]haiku.rag version (db)[/repr.attrib_name]: {stored_version}"
        )
        if embed_provider or embed_model or vector_dim:
            provider_part = embed_provider or "unknown"
            model_part = embed_model or "unknown"
            dim_part = f"{vector_dim}" if vector_dim is not None else "unknown"
            self.console.print(
                "  [repr.attrib_name]embeddings[/repr.attrib_name]: "
                f"{provider_part}/{model_part} (dim: {dim_part})"
            )
        else:
            self.console.print(
                "  [repr.attrib_name]embeddings[/repr.attrib_name]: unknown"
            )
        self.console.print(
            f"  [repr.attrib_name]documents[/repr.attrib_name]: {num_docs} "
            f"({format_bytes(doc_bytes)})"
        )
        self.console.print(
            f"  [repr.attrib_name]chunks[/repr.attrib_name]: {num_chunks} "
            f"({format_bytes(chunk_bytes)})"
        )

        # Vector index information
        if has_vector_index:
            self.console.print(
                "  [repr.attrib_name]vector index[/repr.attrib_name]: ✓ exists"
            )
            self.console.print(
                f"  [repr.attrib_name]indexed chunks[/repr.attrib_name]: {num_indexed_rows}"
            )
            if num_unindexed_rows > 0:
                self.console.print(
                    f"  [repr.attrib_name]unindexed chunks[/repr.attrib_name]: [yellow]{num_unindexed_rows}[/yellow] "
                    "(consider running: haiku-rag create-index)"
                )
            else:
                self.console.print(
                    f"  [repr.attrib_name]unindexed chunks[/repr.attrib_name]: {num_unindexed_rows}"
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

        self.console.print(
            f"  [repr.attrib_name]versions (documents)[/repr.attrib_name]: {doc_versions}"
        )
        self.console.print(
            f"  [repr.attrib_name]versions (chunks)[/repr.attrib_name]: {chunk_versions}"
        )
        self.console.rule()
        self.console.print("[bold]Versions[/bold]")
        self.console.print(
            f"  [repr.attrib_name]haiku.rag[/repr.attrib_name]: {hr_version}"
        )
        self.console.print(
            f"  [repr.attrib_name]lancedb[/repr.attrib_name]: {ldb_version}"
        )
        self.console.print(
            f"  [repr.attrib_name]docling[/repr.attrib_name]: {docling_version}"
        )

    async def history(self, table: str | None = None, limit: int | None = None):
        """Display version history for database tables.

        Args:
            table: Specific table to show history for (documents, chunks, settings).
                   If None, shows history for all tables.
            limit: Maximum number of versions to show per table.
        """
        from haiku.rag.store.engine import Store

        if not self.db_path.exists():
            self.console.print("[red]Database path does not exist.[/red]")
            return

        store = Store(self.db_path, config=self.config, skip_validation=True)

        tables = ["documents", "chunks", "settings"]
        if table:
            if table not in tables:
                self.console.print(
                    f"[red]Unknown table: {table}. Must be one of: {', '.join(tables)}[/red]"
                )
                store.close()
                return
            tables = [table]

        self.console.print("[bold]Version History[/bold]")

        for table_name in tables:
            versions = store.list_table_versions(table_name)

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

        store.close()

    async def list_documents(self, filter: str | None = None):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ) as self.client:
            documents = await self.client.list_documents(filter=filter)
            for doc in documents:
                self._rich_print_document(doc, truncate=True)

    async def add_document_from_text(self, text: str, metadata: dict | None = None):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ) as self.client:
            doc = await self.client.create_document(text, metadata=metadata)
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
            read_only=self.read_only,
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
        self, query: str, limit: int | None = None, filter: str | None = None
    ):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ) as self.client:
            results = await self.client.search(query, limit=limit, filter=filter)
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
            read_only=self.read_only,
            before=self.before,
        ) as self.client:
            chunk = await self.client.chunk_repository.get_by_id(chunk_id)
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
        cite: bool = False,
        deep: bool = False,
        verbose: bool = False,
        filter: str | None = None,
    ):
        """Ask a question using the RAG system.

        Args:
            question: The question to ask
            cite: Include citations in the answer
            deep: Use deep QA mode (multi-step reasoning)
            verbose: Show verbose output
            filter: SQL WHERE clause to filter documents
        """
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ) as self.client:
            try:
                citations = []
                if deep:
                    from haiku.rag.graph.research.models import ResearchReport

                    graph = build_research_graph(config=self.config)
                    context = ResearchContext(original_question=question)
                    state = ResearchState.from_config(
                        context=context,
                        config=self.config,
                        max_iterations=2,
                        confidence_threshold=0.0,
                    )
                    state.search_filter = filter
                    deps = ResearchDeps(client=self.client)

                    if verbose:
                        renderer = AGUIConsoleRenderer(self.console)
                        result_dict = await renderer.render(
                            stream_graph(graph, state, deps)
                        )
                        report = (
                            ResearchReport.model_validate(result_dict)
                            if result_dict
                            else None
                        )
                    else:
                        report = await graph.run(state=state, deps=deps)

                    self.console.print(f"[bold blue]Question:[/bold blue] {question}")
                    self.console.print()
                    if report:
                        self.console.print("[bold green]Answer:[/bold green]")
                        self.console.print(Markdown(report.executive_summary))
                        if report.main_findings:
                            self.console.print()
                            self.console.print("[bold cyan]Key Findings:[/bold cyan]")
                            for finding in report.main_findings:
                                self.console.print(f"• {finding}")
                        if report.sources_summary:
                            self.console.print()
                            self.console.print("[bold cyan]Sources:[/bold cyan]")
                            self.console.print(report.sources_summary)
                    else:
                        self.console.print("[yellow]No answer generated.[/yellow]")
                else:
                    answer, citations = await self.client.ask(question, filter=filter)

                    self.console.print(f"[bold blue]Question:[/bold blue] {question}")
                    self.console.print()
                    self.console.print("[bold green]Answer:[/bold green]")
                    self.console.print(Markdown(answer))
                    if cite and citations:
                        for renderable in format_citations_rich(citations):
                            self.console.print(renderable)
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    async def research(
        self, question: str, verbose: bool = False, filter: str | None = None
    ):
        """Run research via the pydantic-graph pipeline.

        Args:
            question: The research question
            verbose: Show AG-UI event stream during execution
            filter: SQL WHERE clause to filter documents
        """
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ) as client:
            try:
                self.console.print("[bold cyan]Starting research[/bold cyan]")
                self.console.print(f"[bold blue]Question:[/bold blue] {question}")
                self.console.print()

                graph = build_research_graph(config=self.config)
                context = ResearchContext(original_question=question)
                state = ResearchState.from_config(context=context, config=self.config)
                state.search_filter = filter
                deps = ResearchDeps(client=client)

                if verbose:
                    # Use AG-UI renderer to process and display events
                    renderer = AGUIConsoleRenderer(self.console)
                    report_dict = await renderer.render(
                        stream_graph(graph, state, deps)
                    )
                else:
                    # Run without rendering events, just get the result
                    report = await graph.run(state=state, deps=deps)
                    report_dict = (
                        report.model_dump() if hasattr(report, "model_dump") else report
                    )

                if report_dict is None:
                    self.console.print("[red]Research did not produce a report.[/red]")
                    return

                # Convert dict to ResearchReport model
                from haiku.rag.graph.research.models import ResearchReport

                report = ResearchReport.model_validate(report_dict)

                # Display the report
                self.console.print("[bold green]Research Report[/bold green]")
                self.console.rule()

                # Title and Executive Summary
                self.console.print(f"[bold]{report.title}[/bold]")
                self.console.print()
                self.console.print("[bold cyan]Executive Summary:[/bold cyan]")
                self.console.print(report.executive_summary)
                self.console.print()

                # Confidence (from last evaluation)
                if state.last_eval:
                    conf = state.last_eval.confidence_score  # type: ignore[attr-defined]
                    self.console.print(f"[bold cyan]Confidence:[/bold cyan] {conf:.1%}")
                    self.console.print()

                # Main Findings
                if report.main_findings:
                    self.console.print("[bold cyan]Main Findings:[/bold cyan]")
                    for finding in report.main_findings:
                        self.console.print(f"• {finding}")
                    self.console.print()

                # (Themes section removed)

                # Conclusions
                if report.conclusions:
                    self.console.print("[bold cyan]Conclusions:[/bold cyan]")
                    for conclusion in report.conclusions:
                        self.console.print(f"• {conclusion}")
                    self.console.print()

                # Recommendations
                if report.recommendations:
                    self.console.print("[bold cyan]Recommendations:[/bold cyan]")
                    for rec in report.recommendations:
                        self.console.print(f"• {rec}")
                    self.console.print()

                # Limitations
                if report.limitations:
                    self.console.print("[bold yellow]Limitations:[/bold yellow]")
                    for limitation in report.limitations:
                        self.console.print(f"• {limitation}")
                    self.console.print()

                # Sources Summary
                if report.sources_summary:
                    self.console.print("[bold cyan]Sources:[/bold cyan]")
                    self.console.print(report.sources_summary)

            except Exception as e:
                self.console.print(f"[red]Error during research: {e}[/red]")

    async def rebuild(self, mode: RebuildMode = RebuildMode.FULL):
        async with HaikuRAG(
            db_path=self.db_path,
            config=self.config,
            skip_validation=True,
            read_only=self.read_only,
            before=self.before,
        ) as client:
            try:
                documents = await client.list_documents()
                total_docs = len(documents)

                if total_docs == 0:
                    self.console.print(
                        "[yellow]No documents found in database.[/yellow]"
                    )
                    return

                mode_desc = {
                    RebuildMode.FULL: "full rebuild",
                    RebuildMode.RECHUNK: "rechunk",
                    RebuildMode.EMBED_ONLY: "embed only",
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
            except Exception as e:
                self.console.print(f"[red]Error rebuilding database: {e}[/red]")

    async def vacuum(self):
        """Run database maintenance: optimize and cleanup table history."""
        try:
            async with HaikuRAG(
                db_path=self.db_path,
                config=self.config,
                skip_validation=True,
                read_only=self.read_only,
                before=self.before,
            ) as client:
                await client.vacuum()
            self.console.print(
                "[bold green]Vacuum completed successfully.[/bold green]"
            )
        except Exception as e:
            self.console.print(f"[red]Error during vacuum: {e}[/red]")

    async def create_index(self):
        """Create vector index on the chunks table."""
        try:
            async with HaikuRAG(
                db_path=self.db_path,
                config=self.config,
                skip_validation=True,
                read_only=self.read_only,
                before=self.before,
            ) as client:
                row_count = client.store.chunks_table.count_rows()
                self.console.print(f"Chunks in database: {row_count}")

                if row_count < 256:
                    self.console.print(
                        f"[yellow]Warning: Need at least 256 chunks to create an index (have {row_count})[/yellow]"
                    )
                    return

                # Check if index already exists
                indices = client.store.chunks_table.list_indices()
                has_vector_index = any("vector" in str(idx).lower() for idx in indices)

                if has_vector_index:
                    self.console.print(
                        "[yellow]Rebuilding existing vector index...[/yellow]"
                    )
                else:
                    self.console.print("[bold]Creating vector index...[/bold]")

                client.store._ensure_vector_index()
                self.console.print(
                    "[bold green]Vector index created successfully.[/bold green]"
                )
        except Exception as e:
            self.console.print(f"[red]Error creating index: {e}[/red]")

    async def download_models(self):
        """Download Docling, HuggingFace tokenizer, and Ollama models per config."""
        from haiku.rag.client import HaikuRAG

        client = HaikuRAG(db_path=None, config=self.config)

        progress: Progress | None = None
        task_id: TaskID | None = None
        current_model = ""
        current_digest = ""

        async for event in client.download_models():
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

    async def serve(
        self,
        enable_monitor: bool = True,
        enable_mcp: bool = True,
        mcp_transport: str | None = None,
        mcp_port: int = 8001,
        enable_agui: bool = False,
    ):
        """Start the server with selected services."""
        async with HaikuRAG(
            self.db_path,
            config=self.config,
            read_only=self.read_only,
            before=self.before,
        ) as client:
            tasks = []

            # Start file monitor if enabled (not available in read-only mode)
            if enable_monitor:
                if self.read_only:
                    logger.warning(
                        "File monitor disabled: cannot monitor files in read-only mode"
                    )
                else:
                    monitor = FileWatcher(client=client, config=self.config)
                    monitor_task = asyncio.create_task(monitor.observe())
                    tasks.append(monitor_task)

            # Start MCP server if enabled
            if enable_mcp:
                server = create_mcp_server(
                    self.db_path, config=self.config, read_only=self.read_only
                )

                async def run_mcp():
                    if mcp_transport == "stdio":
                        await server.run_stdio_async()
                    else:
                        logger.info(f"Starting MCP server on port {mcp_port}")
                        await server.run_http_async(
                            transport="streamable-http", port=mcp_port
                        )

                mcp_task = asyncio.create_task(run_mcp())
                tasks.append(mcp_task)

            # Start AG-UI server if enabled
            if enable_agui:

                async def run_agui():
                    import uvicorn

                    from haiku.rag.graph.agui import create_agui_server

                    logger.info(
                        f"Starting AG-UI server on {self.config.agui.host}:{self.config.agui.port}"
                    )
                    app = create_agui_server(self.config, db_path=self.db_path)
                    config = uvicorn.Config(
                        app=app,
                        host=self.config.agui.host,
                        port=self.config.agui.port,
                        log_level="info",
                    )
                    server = uvicorn.Server(config)
                    await server.serve()

                agui_task = asyncio.create_task(run_agui())
                tasks.append(agui_task)

            if not tasks:
                logger.warning("No services enabled")
                return

            try:
                # Wait for any task to complete (or KeyboardInterrupt)
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                pass
            finally:
                # Cancel all tasks
                for task in tasks:
                    task.cancel()
                # Wait for cancellation
                await asyncio.gather(*tasks, return_exceptions=True)

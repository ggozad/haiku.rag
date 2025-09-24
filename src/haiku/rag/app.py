import asyncio
import json
from importlib.metadata import version as pkg_version
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress

from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.mcp import create_mcp_server
from haiku.rag.monitor import FileWatcher
from haiku.rag.research.dependencies import ResearchContext
from haiku.rag.research.graph import (
    PlanNode,
    ResearchDeps,
    ResearchState,
    build_research_graph,
)
from haiku.rag.research.stream import stream_research_graph
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document


class HaikuRAGApp:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.console = Console()

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
            hr_version = pkg_version("haiku.rag")
        except Exception:
            hr_version = "unknown"
        try:
            docling_version = pkg_version("docling")
        except Exception:
            docling_version = "unknown"

        # Read settings (if present) to find stored haiku.rag version and embedding config
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
                embed_provider = data.get("EMBEDDINGS_PROVIDER")
                embed_model = data.get("EMBEDDINGS_MODEL")
                vector_dim = (
                    int(data.get("EMBEDDINGS_VECTOR_DIM"))  # pyright: ignore[reportArgumentType]
                    if data.get("EMBEDDINGS_VECTOR_DIM") is not None
                    else None
                )

        num_docs = 0
        if "documents" in table_names:
            docs_tbl = db.open_table("documents")
            num_docs = int(docs_tbl.count_rows())  # type: ignore[attr-defined]

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
            f"  [repr.attrib_name]documents[/repr.attrib_name]: {num_docs}"
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

    async def list_documents(self):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            documents = await self.client.list_documents()
            for doc in documents:
                self._rich_print_document(doc, truncate=True)

    async def add_document_from_text(self, text: str, metadata: dict | None = None):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            doc = await self.client.create_document(text, metadata=metadata)
            self._rich_print_document(doc, truncate=True)
            self.console.print(
                f"[bold green]Document {doc.id} added successfully.[/bold green]"
            )

    async def add_document_from_source(
        self, source: str, title: str | None = None, metadata: dict | None = None
    ):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            doc = await self.client.create_document_from_source(
                source, title=title, metadata=metadata
            )
            self._rich_print_document(doc, truncate=True)
            self.console.print(
                f"[bold green]Document {doc.id} added successfully.[/bold green]"
            )

    async def get_document(self, doc_id: str):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            doc = await self.client.get_document_by_id(doc_id)
            if doc is None:
                self.console.print(f"[red]Document with id {doc_id} not found.[/red]")
                return
            self._rich_print_document(doc, truncate=False)

    async def delete_document(self, doc_id: str):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            deleted = await self.client.delete_document(doc_id)
            if deleted:
                self.console.print(
                    f"[bold green]Document {doc_id} deleted successfully.[/bold green]"
                )
            else:
                self.console.print(
                    f"[yellow]Document with id {doc_id} not found.[/yellow]"
                )

    async def search(self, query: str, limit: int = 5):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            results = await self.client.search(query, limit=limit)
            if not results:
                self.console.print("[yellow]No results found.[/yellow]")
                return
            for chunk, score in results:
                self._rich_print_search_result(chunk, score)

    async def ask(self, question: str, cite: bool = False):
        async with HaikuRAG(db_path=self.db_path) as self.client:
            try:
                answer = await self.client.ask(question, cite=cite)
                self.console.print(f"[bold blue]Question:[/bold blue] {question}")
                self.console.print()
                self.console.print("[bold green]Answer:[/bold green]")
                self.console.print(Markdown(answer))
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    async def research(
        self,
        question: str,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
        max_concurrency: int = 1,
        verbose: bool = False,
    ):
        """Run research via the pydantic-graph pipeline (default)."""
        async with HaikuRAG(db_path=self.db_path) as client:
            try:
                if verbose:
                    self.console.print("[bold cyan]Starting research[/bold cyan]")
                    self.console.print(f"[bold blue]Question:[/bold blue] {question}")
                    self.console.print()

                graph = build_research_graph()
                context = ResearchContext(original_question=question)
                state = ResearchState(
                    context=context,
                    max_iterations=max_iterations,
                    confidence_threshold=confidence_threshold,
                    max_concurrency=max_concurrency,
                )
                deps = ResearchDeps(
                    client=client, console=self.console if verbose else None
                )

                start = PlanNode(
                    provider=Config.RESEARCH_PROVIDER or Config.QA_PROVIDER,
                    model=Config.RESEARCH_MODEL or Config.QA_MODEL,
                )
                report = None
                async for event in stream_research_graph(graph, start, state, deps):
                    if event.type == "report":
                        report = event.report
                        break
                    if event.type == "error":
                        self.console.print(
                            f"[red]Error during research: {event.message}[/red]"
                        )
                        return

                if report is None:
                    self.console.print("[red]Research did not produce a report.[/red]")
                    return

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

    async def rebuild(self):
        async with HaikuRAG(db_path=self.db_path, skip_validation=True) as client:
            try:
                documents = await client.list_documents()
                total_docs = len(documents)

                if total_docs == 0:
                    self.console.print(
                        "[yellow]No documents found in database.[/yellow]"
                    )
                    return

                self.console.print(
                    f"[bold cyan]Rebuilding database with {total_docs} documents...[/bold cyan]"
                )
                with Progress() as progress:
                    task = progress.add_task("Rebuilding...", total=total_docs)
                    async for _ in client.rebuild_database():
                        progress.update(task, advance=1)

                self.console.print(
                    "[bold green]Database rebuild completed successfully.[/bold green]"
                )
            except Exception as e:
                self.console.print(f"[red]Error rebuilding database: {e}[/red]")

    async def vacuum(self):
        """Run database maintenance: optimize and cleanup table history."""
        try:
            async with HaikuRAG(db_path=self.db_path, skip_validation=True) as client:
                await client.vacuum()
            self.console.print(
                "[bold green]Vacuum completed successfully.[/bold green]"
            )
        except Exception as e:
            self.console.print(f"[red]Error during vacuum: {e}[/red]")

    def show_settings(self):
        """Display current configuration settings."""
        self.console.print("[bold]haiku.rag configuration[/bold]")
        self.console.print()

        # Get all config fields dynamically
        for field_name, field_value in Config.model_dump().items():
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

    def _rich_print_search_result(self, chunk: Chunk, score: float):
        """Format a search result chunk for display."""
        content = Markdown(chunk.content)
        self.console.print(
            f"[repr.attrib_name]document_id[/repr.attrib_name]: {chunk.document_id} "
            f"[repr.attrib_name]score[/repr.attrib_name]: {score:.4f}"
        )
        if chunk.document_uri:
            self.console.print("[repr.attrib_name]document uri[/repr.attrib_name]:")
            self.console.print(chunk.document_uri)
        if chunk.document_title:
            self.console.print("[repr.attrib_name]document title[/repr.attrib_name]:")
            self.console.print(chunk.document_title)
        if chunk.document_meta:
            self.console.print("[repr.attrib_name]document meta[/repr.attrib_name]:")
            self.console.print(chunk.document_meta)
        self.console.print("[repr.attrib_name]content[/repr.attrib_name]:")
        self.console.print(content)
        self.console.rule()

    async def serve(self, transport: str | None = None):
        """Start the MCP server."""
        async with HaikuRAG(self.db_path) as client:
            monitor = FileWatcher(paths=Config.MONITOR_DIRECTORIES, client=client)
            monitor_task = asyncio.create_task(monitor.observe())
            server = create_mcp_server(self.db_path)

            try:
                if transport == "stdio":
                    await server.run_stdio_async()
                else:
                    await server.run_http_async(transport="streamable-http")
            except KeyboardInterrupt:
                pass
            finally:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

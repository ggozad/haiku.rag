import asyncio
import json
import logging
from importlib.metadata import version as pkg_version
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress

from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.graph.agui import AGUIConsoleRenderer, stream_graph
from haiku.rag.graph.research.dependencies import ResearchContext
from haiku.rag.graph.research.graph import build_research_graph
from haiku.rag.graph.research.state import ResearchDeps, ResearchState
from haiku.rag.mcp import create_mcp_server
from haiku.rag.monitor import FileWatcher
from haiku.rag.store.models.chunk import Chunk
from haiku.rag.store.models.document import Document

logger = logging.getLogger(__name__)


class HaikuRAGApp:
    def __init__(self, db_path: Path, config: AppConfig = Config):
        self.db_path = db_path
        self.config = config
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
            hr_version = pkg_version("haiku.rag-slim")
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
                embeddings = data.get("embeddings", {})
                embed_provider = embeddings.get("provider")
                embed_model = embeddings.get("model")
                vector_dim = embeddings.get("vector_dim")

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

    async def list_documents(self, filter: str | None = None):
        async with HaikuRAG(
            db_path=self.db_path, config=self.config, allow_create=False
        ) as self.client:
            documents = await self.client.list_documents(filter=filter)
            for doc in documents:
                self._rich_print_document(doc, truncate=True)

    async def add_document_from_text(self, text: str, metadata: dict | None = None):
        async with HaikuRAG(db_path=self.db_path, config=self.config) as self.client:
            doc = await self.client.create_document(text, metadata=metadata)
            self._rich_print_document(doc, truncate=True)
            self.console.print(
                f"[bold green]Document {doc.id} added successfully.[/bold green]"
            )

    async def add_document_from_source(
        self, source: str, title: str | None = None, metadata: dict | None = None
    ):
        async with HaikuRAG(db_path=self.db_path, config=self.config) as self.client:
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
            db_path=self.db_path, config=self.config, allow_create=False
        ) as self.client:
            doc = await self.client.get_document_by_id(doc_id)
            if doc is None:
                self.console.print(f"[red]Document with id {doc_id} not found.[/red]")
                return
            self._rich_print_document(doc, truncate=False)

    async def delete_document(self, doc_id: str):
        async with HaikuRAG(db_path=self.db_path, config=self.config) as self.client:
            deleted = await self.client.delete_document(doc_id)
            if deleted:
                self.console.print(
                    f"[bold green]Document {doc_id} deleted successfully.[/bold green]"
                )
            else:
                self.console.print(
                    f"[yellow]Document with id {doc_id} not found.[/yellow]"
                )

    async def search(self, query: str, limit: int = 5, filter: str | None = None):
        async with HaikuRAG(
            db_path=self.db_path, config=self.config, allow_create=False
        ) as self.client:
            results = await self.client.search(query, limit=limit, filter=filter)
            if not results:
                self.console.print("[yellow]No results found.[/yellow]")
                return
            for chunk, score in results:
                self._rich_print_search_result(chunk, score)

    async def ask(
        self,
        question: str,
        cite: bool = False,
        deep: bool = False,
        verbose: bool = False,
    ):
        """Ask a question using the RAG system.

        Args:
            question: The question to ask
            cite: Include citations in the answer
            deep: Use deep QA mode (multi-step reasoning)
            verbose: Show verbose output
        """
        async with HaikuRAG(
            db_path=self.db_path, config=self.config, allow_create=False
        ) as self.client:
            try:
                if deep:
                    from haiku.rag.graph.deep_qa.dependencies import DeepQAContext
                    from haiku.rag.graph.deep_qa.graph import build_deep_qa_graph
                    from haiku.rag.graph.deep_qa.state import DeepQADeps, DeepQAState

                    graph = build_deep_qa_graph(config=self.config)
                    context = DeepQAContext(
                        original_question=question, use_citations=cite
                    )
                    state = DeepQAState.from_config(context=context, config=self.config)
                    deps = DeepQADeps(client=self.client)

                    if verbose:
                        # Use AG-UI renderer to process and display events
                        from haiku.rag.graph.agui import AGUIConsoleRenderer

                        renderer = AGUIConsoleRenderer(self.console)
                        result_dict = await renderer.render(
                            stream_graph(graph, state, deps)
                        )
                        # Result should be a dict with 'answer' key
                        answer = result_dict.get("answer", "") if result_dict else ""
                    else:
                        # Run without rendering events, just get the result
                        result = await graph.run(state=state, deps=deps)
                        answer = result.answer
                else:
                    answer = await self.client.ask(question, cite=cite)

                self.console.print(f"[bold blue]Question:[/bold blue] {question}")
                self.console.print()
                self.console.print("[bold green]Answer:[/bold green]")
                self.console.print(Markdown(answer))
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")

    async def research(self, question: str, verbose: bool = False):
        """Run research via the pydantic-graph pipeline.

        Args:
            question: The research question
            verbose: Show AG-UI event stream during execution
        """
        async with HaikuRAG(
            db_path=self.db_path, config=self.config, allow_create=False
        ) as client:
            try:
                self.console.print("[bold cyan]Starting research[/bold cyan]")
                self.console.print(f"[bold blue]Question:[/bold blue] {question}")
                self.console.print()

                graph = build_research_graph(config=self.config)
                context = ResearchContext(original_question=question)
                state = ResearchState.from_config(context=context, config=self.config)
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

    async def rebuild(self):
        async with HaikuRAG(
            db_path=self.db_path, config=self.config, skip_validation=True
        ) as client:
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
            async with HaikuRAG(
                db_path=self.db_path, config=self.config, skip_validation=True
            ) as client:
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

    async def serve(
        self,
        enable_monitor: bool = True,
        enable_mcp: bool = True,
        mcp_transport: str | None = None,
        mcp_port: int = 8001,
        enable_agui: bool = False,
    ):
        """Start the server with selected services."""
        async with HaikuRAG(self.db_path, config=self.config) as client:
            tasks = []

            # Start file monitor if enabled
            if enable_monitor:
                monitor = FileWatcher(client=client, config=self.config)
                monitor_task = asyncio.create_task(monitor.observe())
                tasks.append(monitor_task)

            # Start MCP server if enabled
            if enable_mcp:
                server = create_mcp_server(self.db_path, config=self.config)

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

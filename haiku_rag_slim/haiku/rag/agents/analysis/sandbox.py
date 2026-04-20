import asyncio
import concurrent.futures
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pydantic_monty
from pydantic_monty import CallbackFile, MemoryFile, MontyRepl, OSAccess

from haiku.rag.agents.analysis.dependencies import AnalysisContext
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult

if TYPE_CHECKING:
    from pathlib import PurePosixPath


@dataclass
class SandboxResult:
    """Result of executing code in the sandbox."""

    stdout: str
    stderr: str
    success: bool


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync context (CallbackFile read)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class Sandbox:
    """Execute code in a sandboxed Python interpreter.

    Uses pydantic-monty, a minimal secure Python interpreter written in Rust.
    External functions (search, llm) are called by Monty code using ``await``
    and resolved asynchronously on the host.
    Documents are exposed via a virtual filesystem at ``/documents/{id}/``.

    The interpreter uses a REPL session — variables persist across
    ``execute()`` calls within the same Sandbox instance.

        sandbox = Sandbox(db_path, config, context)
        result = await sandbox.execute("x = await search('query')")
        result = await sandbox.execute("print(x[0]['content'])")  # x persists
    """

    _db_path: Path
    _config: AppConfig
    _context: AnalysisContext
    _search_results: "list[SearchResult]"
    _repl: MontyRepl | None
    _vfs: OSAccess | None

    def __init__(
        self,
        db_path: Path,
        config: AppConfig,
        context: AnalysisContext,
    ):
        self._db_path = db_path
        self._config = config
        self._context = context
        self._search_results = []
        self._repl = None
        self._vfs = None

    def _build_external_functions(self) -> dict[str, Any]:
        """Build async external functions for the Monty interpreter."""
        db_path = self._db_path
        config = self._config
        context = self._context

        async def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
            from haiku.rag.client import HaikuRAG

            async with HaikuRAG(db_path, config=config, read_only=True) as rag:
                results = await rag.search(query, limit=limit, filter=context.filter)
                expanded = await rag.expand_context(results)
            self._search_results.extend(expanded)
            return [
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content,
                    "document_id": r.document_id,
                    "document_title": r.document_title,
                    "document_uri": r.document_uri,
                    "score": r.score,
                    "page_numbers": r.page_numbers,
                    "headings": r.headings,
                    "doc_item_refs": r.doc_item_refs,
                    "labels": r.labels,
                }
                for r in expanded
            ]

        async def list_documents() -> list[dict[str, Any]]:
            from haiku.rag.client import HaikuRAG

            async with HaikuRAG(db_path, config=config, read_only=True) as rag:
                docs = await rag.list_documents(filter=context.filter)
            return [
                {
                    "id": d.id,
                    "title": d.title,
                    "uri": d.uri,
                    "created_at": str(d.created_at),
                }
                for d in docs
            ]

        async def llm(prompt: str) -> str:
            from pydantic_ai import Agent

            from haiku.rag.utils import get_model

            model = get_model(config.analysis.model, config)
            agent: Agent[None, str] = Agent(model, output_type=str)
            result = await agent.run(prompt)
            return result.output

        return {
            "search": search,
            "list_documents": list_documents,
            "llm": llm,
        }

    async def _build_vfs(self) -> OSAccess:
        """Build the virtual filesystem with document data.

        Mounts per-document directories with:
        - metadata.json: MemoryFile (eager, small)
        - content.txt: CallbackFile (lazy, can be large)
        - items.jsonl: CallbackFile (lazy, can be large)
        """
        from haiku.rag.client import HaikuRAG

        db_path = self._db_path
        config = self._config
        files: list[MemoryFile | CallbackFile] = []

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            docs = await rag.list_documents(filter=self._context.filter)

        for doc in docs:
            if not doc.id:
                continue
            doc_id: str = doc.id
            doc_dir = f"/documents/{doc_id}"

            metadata = json.dumps(
                {
                    "id": doc_id,
                    "title": doc.title,
                    "uri": doc.uri,
                    "created_at": str(doc.created_at),
                },
                ensure_ascii=False,
            )
            files.append(MemoryFile(f"{doc_dir}/metadata.json", metadata))

            def _make_content_reader(
                did: str,
            ) -> Callable[["PurePosixPath"], str]:
                def read_content(_path: "PurePosixPath") -> str:
                    async def _fetch() -> str:
                        from haiku.rag.client import HaikuRAG
                        from haiku.rag.utils import escape_sql_string

                        async with HaikuRAG(
                            db_path, config=config, read_only=True
                        ) as rag:
                            safe_id = escape_sql_string(did)
                            rows = list(
                                rag.store.documents_table.search()
                                .select(["content"])
                                .where(f"id = '{safe_id}'")
                                .limit(1)
                                .to_list()
                            )
                            return rows[0]["content"] if rows else ""

                    return _run_async(_fetch())

                return read_content

            def _make_items_reader(
                did: str,
            ) -> Callable[["PurePosixPath"], str]:
                def read_items(_path: "PurePosixPath") -> str:
                    async def _fetch() -> str:
                        from haiku.rag.client import HaikuRAG

                        async with HaikuRAG(
                            db_path, config=config, read_only=True
                        ) as rag:
                            items = (
                                await rag.document_item_repository.get_items_in_range(
                                    did, 0, 999999
                                )
                            )
                        lines = []
                        for item in items:
                            lines.append(
                                json.dumps(
                                    {
                                        "position": item.position,
                                        "self_ref": item.self_ref,
                                        "label": item.label,
                                        "text": item.text,
                                        "page_numbers": item.page_numbers,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        return "\n".join(lines)

                    return _run_async(_fetch())

                return read_items

            files.append(
                CallbackFile(
                    f"{doc_dir}/content.txt",
                    read=_make_content_reader(doc_id),
                    write=lambda _p, _c: None,
                )
            )
            files.append(
                CallbackFile(
                    f"{doc_dir}/items.jsonl",
                    read=_make_items_reader(doc_id),
                    write=lambda _p, _c: None,
                )
            )

        return OSAccess(files)

    async def _ensure_initialized(self) -> None:
        """Initialize the REPL session and VFS on first use."""
        if self._repl is None:
            self._vfs = await self._build_vfs()
            self._repl = MontyRepl(
                limits={
                    "max_duration_secs": self._config.analysis.code_timeout,
                },
            )
            if self._context.documents:
                await pydantic_monty.run_repl_async(
                    self._repl,
                    "pass",
                    inputs={
                        "documents": [
                            {
                                "id": d.id,
                                "title": d.title,
                                "uri": d.uri,
                                "content": d.content,
                            }
                            for d in self._context.documents
                        ]
                    },
                    external_functions=self._build_external_functions(),
                    os=self._vfs,
                )

    async def execute(self, code: str) -> SandboxResult:
        """Execute Python code in the Monty REPL.

        Variables persist across calls within the same Sandbox instance.
        """
        await self._ensure_initialized()
        assert self._repl is not None
        assert self._vfs is not None

        external_fns = self._build_external_functions()

        stdout_lines: list[str] = []

        def print_callback(_stream: Literal["stdout"], text: str) -> None:
            stdout_lines.append(text)

        max_chars = self._config.analysis.max_output_chars

        try:
            output = await pydantic_monty.run_repl_async(
                self._repl,
                code,
                external_functions=external_fns,
                print_callback=print_callback,
                os=self._vfs,
            )
        except (
            pydantic_monty.MontySyntaxError,
            pydantic_monty.MontyRuntimeError,
        ) as e:
            stdout = "".join(stdout_lines)
            if len(stdout) > max_chars:
                stdout = stdout[:max_chars] + "\n... (output truncated)"
            return SandboxResult(stdout=stdout, stderr=str(e), success=False)

        stdout = "".join(stdout_lines)
        if output is not None:
            stdout_with_output = f"{stdout}{output}" if stdout else str(output)
        else:
            stdout_with_output = stdout

        if len(stdout_with_output) > max_chars:
            stdout_with_output = (
                stdout_with_output[:max_chars] + "\n... (output truncated)"
            )

        return SandboxResult(stdout=stdout_with_output, stderr="", success=True)

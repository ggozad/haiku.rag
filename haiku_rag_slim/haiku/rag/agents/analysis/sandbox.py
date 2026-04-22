import asyncio
import atexit
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


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
atexit.register(_executor.shutdown, wait=False)


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync context (CallbackFile read)."""
    return _executor.submit(asyncio.run, coro).result()


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
    _items_cache: dict[str, str] | None
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
        self._items_cache = None
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

        def _deny_write(_path: "PurePosixPath", _content: str | bytes) -> None:
            raise PermissionError(f"Document files are read-only: {_path}")

        async with HaikuRAG(db_path, config=config, read_only=True) as rag:
            docs = await rag.list_documents(filter=self._context.filter)

        doc_ids = [doc.id for doc in docs if doc.id]

        def _load_items_cache() -> dict[str, str]:
            """Bulk-fetch all document items in one query, serialize to JSONL."""

            async def _fetch() -> dict[str, str]:
                from haiku.rag.client import HaikuRAG

                async with HaikuRAG(db_path, config=config, read_only=True) as rag:
                    grouped = await rag.document_item_repository.get_all_items_grouped(
                        doc_ids
                    )
                result: dict[str, str] = {}
                for did, items in grouped.items():
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
                    result[did] = "\n".join(lines)
                return result

            return _run_async(_fetch())

        sandbox = self

        def _make_items_reader(
            did: str,
        ) -> Callable[["PurePosixPath"], str]:
            def read_items(_path: "PurePosixPath") -> str:
                if sandbox._items_cache is None:
                    sandbox._items_cache = _load_items_cache()
                return sandbox._items_cache.get(did, "")

            return read_items

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

                        async with HaikuRAG(
                            db_path, config=config, read_only=True
                        ) as rag:
                            content = await rag.document_repository.get_content(did)
                            return content or ""

                    return _run_async(_fetch())

                return read_content

            files.append(
                CallbackFile(
                    f"{doc_dir}/content.txt",
                    read=_make_content_reader(doc_id),
                    write=_deny_write,
                )
            )
            files.append(
                CallbackFile(
                    f"{doc_dir}/items.jsonl",
                    read=_make_items_reader(doc_id),
                    write=_deny_write,
                )
            )

        return OSAccess(files)

    async def _ensure_initialized(self) -> tuple[MontyRepl, OSAccess]:
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
        repl = self._repl
        vfs = self._vfs
        if repl is None or vfs is None:
            raise RuntimeError("Sandbox initialization failed")
        return repl, vfs

    async def execute(self, code: str) -> SandboxResult:
        """Execute Python code in the Monty REPL.

        Variables persist across calls within the same Sandbox instance.
        """
        repl, vfs = await self._ensure_initialized()
        external_fns = self._build_external_functions()

        stdout_lines: list[str] = []

        def print_callback(_stream: Literal["stdout"], text: str) -> None:
            stdout_lines.append(text)

        max_chars = self._config.analysis.max_output_chars

        try:
            output = await pydantic_monty.run_repl_async(
                repl,
                code,
                external_functions=external_fns,
                print_callback=print_callback,
                os=vfs,
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

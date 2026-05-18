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
from haiku.rag.store.models.document_item import DocumentItem

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


def _build_toc(items: list["DocumentItem"]) -> list[dict[str, Any]]:
    """Build a nested section tree from items in position order.

    Each ``section_header`` with ``heading_level > 0`` becomes a node. Nesting
    follows the explicit levels: a header pops the stack until the top is at
    a strictly shallower level, then becomes a child of that top (or a root).

    ``item_range = [position, end_exclusive]`` where ``end_exclusive`` is the
    position of the next header whose level is the same or shallower (i.e.
    the next sibling or ancestor that ends this section), or the total item
    count if no such header exists.

    Items without a section_header label (or with ``heading_level == 0``) are
    skipped. PDF-derived corpora where docling collapses every header to level
    1 produce a flat list of sibling nodes.
    """
    headers: list[DocumentItem] = [
        i for i in items if i.label == "section_header" and i.heading_level > 0
    ]
    if not headers:
        return []

    total = max((i.position for i in items), default=-1) + 1

    ends: list[int] = []
    for idx, h in enumerate(headers):
        end = total
        for j in range(idx + 1, len(headers)):
            if headers[j].heading_level <= h.heading_level:
                end = headers[j].position
                break
        ends.append(end)

    roots: list[dict[str, Any]] = []
    stack: list[tuple[int, dict[str, Any]]] = []
    for h, end in zip(headers, ends, strict=True):
        node: dict[str, Any] = {
            "self_ref": h.self_ref,
            "level": h.heading_level,
            "title": h.text,
            "position": h.position,
            "page_numbers": list(h.page_numbers),
            "item_range": [h.position, end],
            "children": [],
        }
        while stack and stack[-1][0] >= h.heading_level:
            stack.pop()
        (stack[-1][1]["children"] if stack else roots).append(node)
        stack.append((h.heading_level, node))
    return roots


class Sandbox:
    """Execute code in a sandboxed Python interpreter.

    Uses pydantic-monty, a minimal secure Python interpreter written in Rust.
    External functions (search, list_documents) are called by Monty code
    using ``await`` and resolved asynchronously on the host. Documents are
    exposed via a virtual filesystem at ``/documents/{id}/``.

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
    _toc_cache: dict[str, str] | None
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
        self._toc_cache = None
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
            out: list[dict[str, Any]] = []
            for r in expanded:
                picture_refs = [
                    ref for ref in r.doc_item_refs if ref.startswith("#/pictures/")
                ]
                out.append(
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
                        "picture_refs": picture_refs,
                    }
                )
            return out

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

        return {
            "search": search,
            "list_documents": list_documents,
        }

    async def _build_vfs(self) -> OSAccess:
        """Build the virtual filesystem with document data.

        Mounts per-document directories with:
        - metadata.json: MemoryFile (eager, small)
        - content.txt: CallbackFile (lazy, can be large)
        - items.jsonl: CallbackFile (lazy, bulk-cached)
        - toc.json: CallbackFile (lazy, bulk-cached)
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
        doc_titles = {doc.id: doc.title for doc in docs if doc.id}

        def _load_caches() -> tuple[dict[str, str], dict[str, str]]:
            """Bulk-fetch document items once and build both items.jsonl and
            toc.json views from the same result. Returns ``(items_cache, toc_cache)``.
            """

            async def _fetch() -> dict[str, list[DocumentItem]]:
                from haiku.rag.client import HaikuRAG

                async with HaikuRAG(db_path, config=config, read_only=True) as rag:
                    return await rag.document_item_repository.get_all_items_grouped(
                        doc_ids
                    )

            grouped = _run_async(_fetch())

            items_cache: dict[str, str] = {}
            toc_cache: dict[str, str] = {}
            for did, items in grouped.items():
                items_cache[did] = "\n".join(
                    json.dumps(
                        {
                            "position": item.position,
                            "self_ref": item.self_ref,
                            "label": item.label,
                            "text": item.text,
                            "page_numbers": item.page_numbers,
                            "heading_level": item.heading_level,
                            "tree_depth": item.tree_depth,
                        },
                        ensure_ascii=False,
                    )
                    for item in items
                )
                toc_cache[did] = json.dumps(
                    {
                        "doc_id": did,
                        "title": doc_titles.get(did),
                        "tree": _build_toc(items),
                    },
                    ensure_ascii=False,
                )
            return items_cache, toc_cache

        sandbox = self

        def _ensure_caches() -> None:
            if sandbox._items_cache is None or sandbox._toc_cache is None:
                items_c, toc_c = _load_caches()
                sandbox._items_cache = items_c
                sandbox._toc_cache = toc_c

        def _make_items_reader(
            did: str,
        ) -> Callable[["PurePosixPath"], str]:
            def read_items(_path: "PurePosixPath") -> str:
                _ensure_caches()
                assert sandbox._items_cache is not None
                return sandbox._items_cache.get(did, "")

            return read_items

        def _make_toc_reader(
            did: str,
        ) -> Callable[["PurePosixPath"], str]:
            def read_toc(_path: "PurePosixPath") -> str:
                _ensure_caches()
                assert sandbox._toc_cache is not None
                return sandbox._toc_cache.get(did, "")

            return read_toc

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
            files.append(
                CallbackFile(
                    f"{doc_dir}/toc.json",
                    read=_make_toc_reader(doc_id),
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
                await self._repl.feed_run_async(
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
        assert self._repl is not None and self._vfs is not None
        return self._repl, self._vfs

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
            output = await repl.feed_run_async(
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

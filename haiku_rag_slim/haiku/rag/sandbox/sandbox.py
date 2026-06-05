import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pydantic_monty
from pydantic_monty import CallbackFile, MemoryFile, MontyRepl, OSAccess

from haiku.rag.config.models import AppConfig
from haiku.rag.sandbox.dependencies import AnalysisContext
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.store.models.document_item import PICTURE_REF_PREFIX, DocumentItem

if TYPE_CHECKING:
    from pathlib import PurePosixPath

    from haiku.rag.client import HaikuRAG


@dataclass
class SandboxResult:
    """Result of executing code in the sandbox."""

    stdout: str
    stderr: str
    success: bool


def _build_toc(
    items: list["DocumentItem"],
    chunk_index: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Build a nested section tree from items in position order.

    Each ``section_header`` with ``heading_level > 0`` becomes a node. Nesting
    follows the explicit levels: a header pops the stack until the top is at
    a strictly shallower level, then becomes a child of that top (or a root).

    ``item_range = [position, end_exclusive]`` where ``end_exclusive`` is the
    position of the next header whose level is the same or shallower (i.e.
    the next sibling or ancestor that ends this section), or the total item
    count if no such header exists.

    ``chunk_ids`` aggregates the chunks covered by all items in the section's
    ``item_range`` (deduped, order preserved). Pass directly to ``cite()`` to
    ground a section-scoped answer without a corpus-wide ``search()`` call.

    Items without a section_header label (or with ``heading_level == 0``) are
    skipped. When all section_headers carry the same level the output is a
    flat sibling list (see docling-project/docling#2121 for an upstream case
    where every PDF section_header is emitted at level=1).
    """
    # Defensive: every consumer is supposed to pass items in position order,
    # but the end_exclusive lookahead below silently miscomputes section
    # boundaries if it's not — better to sort once than trust the caller.
    items = sorted(items, key=lambda i: i.position)
    headers: list[DocumentItem] = [
        i for i in items if i.label == "section_header" and i.heading_level > 0
    ]
    if not headers:
        return []

    total = max((i.position for i in items), default=-1) + 1
    items_by_position: dict[int, DocumentItem] = {i.position: i for i in items}

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
        seen: set[str] = set()
        chunk_ids: list[str] = []
        for pos in range(h.position, end):
            item = items_by_position.get(pos)
            if item is None:
                continue
            for cid in chunk_index.get(item.self_ref, []):
                if cid not in seen:
                    seen.add(cid)
                    chunk_ids.append(cid)
        node: dict[str, Any] = {
            "self_ref": h.self_ref,
            "level": h.heading_level,
            "title": h.text,
            "page_numbers": list(h.page_numbers),
            "item_range": [h.position, end],
            "chunk_ids": chunk_ids,
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

    All database access runs on the event loop that drives ``execute()``. Monty's
    file callbacks are synchronous and run on the interpreter's worker thread, so
    they bridge back to that loop via ``run_coroutine_threadsafe``; the loop is
    free during ``feed_run_async`` (the VM runs on the worker thread), so the
    bridge does not deadlock. When a ``rag`` connection is supplied it is used
    for every read, so an analysis run drives a single connection on a single
    loop; otherwise each read opens an ephemeral read-only connection.
    """

    _db_path: Path
    _config: AppConfig
    _context: AnalysisContext
    _rag: "HaikuRAG | None"
    _search_results: "list[SearchResult]"
    _doc_items: dict[str, list["DocumentItem"]]
    _doc_chunk_index: dict[str, dict[str, list[str]]]
    _items_jsonl_cache: dict[str, str]
    _toc_json_cache: dict[str, str]
    _repl: MontyRepl | None
    _vfs: OSAccess | None
    _loop: asyncio.AbstractEventLoop | None

    def __init__(
        self,
        db_path: Path,
        config: AppConfig,
        context: AnalysisContext,
        rag: "HaikuRAG | None" = None,
    ):
        self._db_path = db_path
        self._config = config
        self._context = context
        self._rag = rag
        self._search_results = []
        self._doc_items = {}
        self._doc_chunk_index = {}
        self._items_jsonl_cache = {}
        self._toc_json_cache = {}
        self._repl = None
        self._vfs = None
        self._loop = None

    @asynccontextmanager
    async def _connection(self) -> "AsyncIterator[HaikuRAG]":
        """Yield the supplied connection, or an ephemeral read-only one."""
        if self._rag is not None:
            yield self._rag
            return
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(self._db_path, config=self._config, read_only=True) as rag:
            yield rag

    def _run_on_loop(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a coroutine on the execute() loop from a synchronous callback.

        Called from Monty's worker thread while ``feed_run_async`` leaves the
        loop free, so scheduling onto it and blocking for the result is safe.
        """
        assert self._loop is not None, (
            "VFS reads happen during execute(); the loop must be captured first."
        )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        """Retained for API compatibility; the sandbox owns no resources."""
        return

    def _build_external_functions(self) -> dict[str, Any]:
        """Build async external functions for the Monty interpreter."""
        context = self._context

        async def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
            # Picture bytes are deliberately not attached to in-code search
            # results: the Monty interpreter has no PIL/base64/hashlib, so the
            # agent's Python can't do anything with them. The driving model
            # gets figures through the top-level `search` tool when the
            # question is visual; in-code search is for structural work.
            async with self._connection() as rag:
                results = await rag.search(query, limit=limit, filter=context.filter)
                expanded = await rag.expand_context(results)
            self._search_results.extend(expanded)
            out: list[dict[str, Any]] = []
            for r in expanded:
                picture_refs = [
                    ref for ref in r.doc_item_refs if ref.startswith(PICTURE_REF_PREFIX)
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
            async with self._connection() as rag:
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
        files: list[MemoryFile | CallbackFile] = []

        def _deny_write(_path: "PurePosixPath", _content: str | bytes) -> None:
            raise PermissionError(f"Document files are read-only: {_path}")

        async with self._connection() as rag:
            docs = await rag.list_documents(filter=self._context.filter)

        doc_titles = {doc.id: doc.title for doc in docs if doc.id}

        sandbox = self

        def _get_items(did: str) -> list[DocumentItem]:
            """Fetch items for one doc, cached on the sandbox."""
            cached = sandbox._doc_items.get(did)
            if cached is not None:
                return cached

            async def _fetch() -> list[DocumentItem]:
                async with sandbox._connection() as rag:
                    return await rag.document_item_repository.get_all_items(did)

            items = sandbox._run_on_loop(_fetch())
            sandbox._doc_items[did] = items
            return items

        def _get_chunk_index(did: str) -> dict[str, list[str]]:
            """Fetch the self_ref → chunk_ids index for one doc, cached."""
            cached = sandbox._doc_chunk_index.get(did)
            if cached is not None:
                return cached

            async def _fetch() -> dict[str, list[str]]:
                async with sandbox._connection() as rag:
                    index = (
                        await rag.chunk_repository.get_chunk_ids_by_self_ref_grouped(
                            [did]
                        )
                    )
                    return index.get(did, {})

            chunk_index = sandbox._run_on_loop(_fetch())
            sandbox._doc_chunk_index[did] = chunk_index
            return chunk_index

        def _make_items_reader(
            did: str,
        ) -> Callable[["PurePosixPath"], str]:
            def read_items(_path: "PurePosixPath") -> str:
                cached = sandbox._items_jsonl_cache.get(did)
                if cached is not None:
                    return cached
                items = _get_items(did)
                chunk_index = _get_chunk_index(did)
                jsonl = "\n".join(
                    json.dumps(
                        {
                            "self_ref": item.self_ref,
                            "label": item.label,
                            "text": item.text,
                            "page_numbers": item.page_numbers,
                            "heading_level": item.heading_level,
                            "chunk_ids": chunk_index.get(item.self_ref, []),
                        },
                        ensure_ascii=False,
                    )
                    for item in items
                )
                sandbox._items_jsonl_cache[did] = jsonl
                return jsonl

            return read_items

        def _make_toc_reader(
            did: str,
        ) -> Callable[["PurePosixPath"], str]:
            def read_toc(_path: "PurePosixPath") -> str:
                cached = sandbox._toc_json_cache.get(did)
                if cached is not None:
                    return cached
                items = _get_items(did)
                chunk_index = _get_chunk_index(did)
                toc = json.dumps(
                    {
                        "doc_id": did,
                        "title": doc_titles.get(did),
                        "tree": _build_toc(items, chunk_index),
                    },
                    ensure_ascii=False,
                )
                sandbox._toc_json_cache[did] = toc
                return toc

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
                        async with sandbox._connection() as rag:
                            content = await rag.document_repository.get_content(did)
                            return content or ""

                    return sandbox._run_on_loop(_fetch())

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
            # HAIKU_RAG_DISABLE_TOC is an evaluation-time toggle for measuring
            # whether toc.json's outline view earns its place in the VFS.
            # Production callers should leave it unset.
            if not os.environ.get("HAIKU_RAG_DISABLE_TOC"):
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
        assert self._repl is not None and self._vfs is not None
        return self._repl, self._vfs

    async def execute(self, code: str) -> SandboxResult:
        """Execute Python code in the Monty REPL.

        Variables persist across calls within the same Sandbox instance.
        """
        # Monty's synchronous file callbacks bridge DB reads back to this loop.
        self._loop = asyncio.get_running_loop()
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

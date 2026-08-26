import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pydantic_monty
from pydantic_monty import (
    AsyncMonty,
    AsyncMontySession,
    CallbackFile,
    OSAccess,
    ResourceLimits,
)

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
    The interpreter runs in a subprocess worker checked out of an ``AsyncMonty``
    pool. External functions (search, list_documents) are called by Monty code
    using ``await`` and resolved asynchronously on the host. Documents are
    exposed via a virtual filesystem at ``/documents/{id}/``.

    The session persists across ``execute()`` calls within the same Sandbox
    instance — variables carry over. Call ``close()`` to return the worker to
    the pool and shut the pool down.

        sandbox = Sandbox(db_path, config, context)
        result = await sandbox.execute("x = await search('query')")
        result = await sandbox.execute("print(x[0]['content'])")  # x persists
        await sandbox.close()

    All database access runs on the event loop that drives ``execute()``. Monty's
    file callbacks are synchronous and run off that loop while ``feed_run`` is
    awaited, so they bridge back to it via ``run_coroutine_threadsafe`` without
    deadlocking. When a ``rag`` connection is supplied it is used for every read,
    so an analysis run drives a single connection on a single loop; otherwise
    each read opens an ephemeral read-only connection.
    """

    _db_path: Path | None
    _config: AppConfig
    _context: AnalysisContext
    _rag: "HaikuRAG | None"
    _owners: dict[str, "HaikuRAG"]
    _lock: "asyncio.Lock | None"
    _search_results: "list[SearchResult]"
    _doc_items: dict[str, list["DocumentItem"]]
    _doc_chunk_index: dict[str, dict[str, list[str]]]
    _items_jsonl_cache: dict[str, str]
    _toc_json_cache: dict[str, str]
    _pool: AsyncMonty | None
    _session: AsyncMontySession | None
    _vfs: OSAccess | None
    _loop: asyncio.AbstractEventLoop | None
    _deadline: float | None

    def __init__(
        self,
        db_path: Path | None,
        config: AppConfig,
        context: AnalysisContext,
        rag: "HaikuRAG | None" = None,
        lock: "asyncio.Lock | None" = None,
    ):
        self._db_path = db_path
        self._config = config
        self._context = context
        self._rag = rag
        self._owners = {}
        self._lock = lock
        self._search_results = []
        self._doc_items = {}
        self._doc_chunk_index = {}
        self._items_jsonl_cache = {}
        self._toc_json_cache = {}
        self._pool = None
        self._session = None
        self._vfs = None
        self._loop = None
        self._deadline = None

    @asynccontextmanager
    async def _connection(
        self, owner: "HaikuRAG | None" = None
    ) -> "AsyncIterator[HaikuRAG]":
        """Yield the shared connection (serialized by the lock), or an ephemeral
        read-only one. The lock guards the whole block so a read's awaits cannot
        interleave with another task's operation on the same connection.

        `owner` is the client holding one document, for the reads addressed to a
        single document. The shared connection covers a set of databases and has
        no repositories of its own, so those reads have to name their owner.
        """
        connection = owner if owner is not None else self._rag
        if connection is not None:
            if self._lock is not None:
                async with self._lock:
                    yield connection
            else:
                yield connection
            return
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(self._db_path, config=self._config, read_only=True) as rag:
            yield rag

    async def _documents(self) -> "tuple[list[Any], dict[str, HaikuRAG]]":
        """Every document in scope, and the client holding each of them.

        The owners are empty where one connection serves every read: a single
        database, or the ephemeral connection opened per read when no client was
        supplied. The selection is resolved the same way a search resolves it, so
        a database the question excluded cannot be mounted.
        """
        async with self._connection() as rag:
            if not rag.covers_multiple:
                if not await rag.clients_covering(self._context.sources):
                    return [], {}
                docs = await rag.list_documents(filter=self._context.filter)
                return docs, {}
            owners = await rag.clients_covering(self._context.sources)
        groups = await asyncio.gather(
            *(owner.list_documents(filter=self._context.filter) for owner in owners)
        )
        # Interleaved, not concatenated: code that prints the listing is read
        # through a truncated output, and concatenating shows one database's
        # documents until the truncation, hiding that there are others.
        docs = [doc for row in zip_longest(*groups) for doc in row if doc is not None]
        return docs, self._holders(owners, groups)

    @staticmethod
    def _holders(
        owners: "list[HaikuRAG]", groups: "list[list[Any]]"
    ) -> "dict[str, HaikuRAG]":
        """Map each document id to the database holding it.

        Document ids are UUID4, so the flat `/documents/{id}/` namespace is
        unambiguous for databases that were filled independently — but not for one
        copied from another, where the same id is in both. Duplicate results are
        merely redundant in a search; here they would be two documents claiming one
        path, and whichever arrived last would answer for both. Rejected rather
        than resolved, since either answer would be wrong half the time.
        """
        holders: dict[str, HaikuRAG] = {}
        held_by: dict[str, str | None] = {}
        for owner, group in zip(owners, groups, strict=True):
            for doc in group:
                if not doc.id:  # pragma: no cover - stored rows always carry an id
                    continue
                if doc.id in holders:
                    raise ValueError(
                        f"document {doc.id} is in databases {held_by[doc.id]!r} and "
                        f"{owner.source!r}; analysis mounts one document per id"
                    )
                holders[doc.id] = owner
                held_by[doc.id] = owner.source
        return holders

    def _run_on_loop(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run a coroutine on the execute() loop from a synchronous callback.

        Called off the loop while ``feed_run`` is awaited, so scheduling onto it
        and blocking for the result is safe.

        Blocking here suspends the worker, and Monty checks its duration budget
        between interpreter steps, so it cannot check while a read is in flight.
        Enforce the budget before starting another read, or code that reads in a
        loop overruns it by however long the outstanding reads take. Raising from
        inside the callback answers the worker's suspension, which keeps the
        session usable — cancelling ``feed_run`` from outside does not, and wedges
        the protocol.
        """
        assert self._loop is not None, (
            "VFS reads happen during execute(); the loop must be captured first."
        )
        if self._deadline is not None and self._loop.time() > self._deadline:
            coro.close()
            raise TimeoutError(
                "time limit exceeded: no further document reads after "
                f"{self._config.analysis.code_timeout}s"
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _discard_session(self) -> None:
        """Drop a session whose worker is gone.

        The session object is unusable once its worker dies: it answers every
        later call with ``RuntimeError: this checkout has already been
        finished``. Clearing it makes ``_ensure_initialized`` check out a
        replacement, at the cost of the variables the dead worker held.
        """
        session, self._session = self._session, None
        if session is not None:
            with suppress(Exception):
                await session.__aexit__(None, None, None)

    async def close(self) -> None:
        """Return the worker to the pool and shut the pool down. Idempotent."""
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._pool is not None:
            await self._pool.__aexit__(None, None, None)
            self._pool = None

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
                results = await rag.search(
                    query,
                    limit=limit,
                    filter=context.filter,
                    sources=context.sources,
                )
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
                        "source": r.source,
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
            docs, _ = await self._documents()
            return [
                {
                    "id": d.id,
                    "title": d.title,
                    "uri": d.uri,
                    "created_at": str(d.created_at),
                    "source": d.source,
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
        - metadata.json: CallbackFile (eager, small)
        - content.txt: CallbackFile (lazy, can be large)
        - items.jsonl: CallbackFile (lazy, bulk-cached)
        - toc.json: CallbackFile (lazy, bulk-cached)
        """
        files: list[CallbackFile] = []

        def _deny_write(_path: "PurePosixPath", _content: str | bytes) -> None:
            raise PermissionError(f"Document files are read-only: {_path}")

        docs, self._owners = await self._documents()

        doc_titles = {doc.id: doc.title for doc in docs if doc.id}

        sandbox = self

        def _get_items(did: str) -> list[DocumentItem]:
            """Fetch items for one doc, cached on the sandbox."""
            cached = sandbox._doc_items.get(did)
            if cached is not None:
                return cached

            async def _fetch() -> list[DocumentItem]:
                async with sandbox._connection(sandbox._owners.get(did)) as rag:
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
                async with sandbox._connection(sandbox._owners.get(did)) as rag:
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
            if not doc.id:  # pragma: no cover - stored rows always carry an id
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

            # MemoryFile has no write hook, so metadata.json goes through the
            # same read and deny pair as the rest. Its content is already built.
            files.append(
                CallbackFile(
                    f"{doc_dir}/metadata.json",
                    read=lambda _path, text=metadata: text,
                    write=_deny_write,
                )
            )

            def _make_content_reader(
                did: str,
            ) -> Callable[["PurePosixPath"], str]:
                def read_content(_path: "PurePosixPath") -> str:
                    async def _fetch() -> str:
                        async with sandbox._connection(sandbox._owners.get(did)) as rag:
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

    def _session_limits(self) -> ResourceLimits:
        """Resource limits for the worker session.

        Monty spends ``max_duration_secs`` across the session's whole life, and
        the session is reused so variables persist between calls. Budget it for
        the run rather than for one call, or the first slow call starves every
        later one. ``code_timeout`` is enforced per call elsewhere: the read
        deadline in ``_run_on_loop`` bounds a call that reads, and the pool's
        ``request_timeout`` bounds one that computes.
        """
        analysis = self._config.analysis
        return {"max_duration_secs": analysis.code_timeout * analysis.max_executions}

    async def _ensure_initialized(self) -> tuple[AsyncMontySession, OSAccess]:
        """Check out a worker session and build the VFS on first use."""
        if self._vfs is None:
            self._vfs = await self._build_vfs()
        if self._pool is None:
            # The watchdog counts only time the worker spends running code, so a
            # read that blocks the worker never trips it. That leaves the two
            # limits disjoint: this one bounds a call that computes, and the read
            # deadline bounds a call that reads.
            pool = AsyncMonty(request_timeout=self._config.analysis.code_timeout)
            await pool.__aenter__()
            self._pool = pool
        if self._session is None:
            session = self._pool.checkout(limits=self._session_limits())
            await session.__aenter__()
            self._session = session
        assert self._session is not None and self._vfs is not None
        return self._session, self._vfs

    async def execute(self, code: str) -> SandboxResult:
        """Execute Python code in the Monty worker session.

        Variables persist across calls within the same Sandbox instance.
        """
        # Monty's synchronous file callbacks bridge DB reads back to this loop.
        self._loop = asyncio.get_running_loop()
        self._deadline = self._loop.time() + self._config.analysis.code_timeout
        session, vfs = await self._ensure_initialized()
        external_fns = self._build_external_functions()

        stdout_lines: list[str] = []

        def print_callback(  # pragma: no cover - runs on Monty's worker thread
            _stream: Literal["stdout", "stderr"], text: str
        ) -> None:
            stdout_lines.append(text)

        max_chars = self._config.analysis.max_output_chars

        try:
            output = await session.feed_run(
                code,
                external_lookup=external_fns,
                print_callback=print_callback,
                os=vfs,
            )
        except (pydantic_monty.MontyError, RuntimeError) as e:
            stdout = "".join(stdout_lines)
            if len(stdout) > max_chars:
                stdout = stdout[:max_chars] + "\n... (output truncated)"
            stderr = str(e)
            # A crash kills the worker, and a protocol error leaves it out of
            # step. Both poison the session. Bad user code does not.
            if isinstance(e, pydantic_monty.MontyCrashedError | RuntimeError):
                await self._discard_session()
                stderr = (
                    f"{stderr}\n\nThe interpreter restarted. Variables from "
                    "earlier calls are gone."
                )
            return SandboxResult(stdout=stdout, stderr=stderr, success=False)

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

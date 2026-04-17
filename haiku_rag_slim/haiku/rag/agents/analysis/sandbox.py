import asyncio
import concurrent.futures
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pydantic_monty
from pydantic_monty import CallbackFile, MemoryFile, OSAccess

from haiku.rag.agents.analysis.dependencies import AnalysisContext
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.chunk import SearchResult

if TYPE_CHECKING:
    from pathlib import PurePosixPath

    from haiku.rag.client import HaikuRAG


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

        sandbox = Sandbox(client, config, context)
        result = await sandbox.execute("print('hello')")
    """

    _client: "HaikuRAG"
    _config: AppConfig
    _context: AnalysisContext
    _search_results: "list[SearchResult]"

    def __init__(
        self,
        client: "HaikuRAG",
        config: AppConfig,
        context: AnalysisContext,
    ):
        self._client = client
        self._config = config
        self._context = context
        self._search_results = []

    def _build_external_functions(self) -> dict[str, Any]:
        """Build async external functions for the Monty interpreter."""
        client = self._client
        config = self._config
        context = self._context

        async def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
            results = await client.search(query, limit=limit, filter=context.filter)
            expanded = await client.expand_context(results)
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
            docs = await client.list_documents(filter=context.filter)
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
        client = self._client
        files: list[MemoryFile | CallbackFile] = []

        docs = await client.list_documents(filter=self._context.filter)

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
                        from haiku.rag.utils import escape_sql_string

                        safe_id = escape_sql_string(did)
                        rows = list(
                            client.store.documents_table.search()
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
                        items = (
                            await client.document_item_repository.get_items_in_range(
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

    async def execute(self, code: str) -> SandboxResult:
        """Execute Python code in the Monty interpreter."""
        external_fns = self._build_external_functions()
        vfs = await self._build_vfs()

        input_names: list[str] = []
        inputs: dict[str, Any] | None = None
        if self._context.documents:
            input_names.append("documents")
            inputs = {
                "documents": [
                    {
                        "id": d.id,
                        "title": d.title,
                        "uri": d.uri,
                        "content": d.content,
                    }
                    for d in self._context.documents
                ]
            }

        try:
            monty = pydantic_monty.Monty(
                code,
                inputs=input_names,
            )
        except (
            pydantic_monty.MontySyntaxError,
            pydantic_monty.MontyRuntimeError,
        ) as e:
            return SandboxResult(stdout="", stderr=str(e), success=False)

        stdout_lines: list[str] = []

        def print_callback(_stream: Literal["stdout"], text: str) -> None:
            stdout_lines.append(text)

        max_chars = self._config.analysis.max_output_chars
        limits: pydantic_monty.ResourceLimits = {
            "max_duration_secs": self._config.analysis.code_timeout,
        }

        try:
            output = await pydantic_monty.run_monty_async(
                monty,
                inputs=inputs,
                external_functions=external_fns,
                limits=limits,
                print_callback=print_callback,
                os=vfs,
            )
        except pydantic_monty.MontyRuntimeError as e:
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

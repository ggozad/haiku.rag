import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pydantic_monty

from haiku.rag.agents.rlm.dependencies import RLMContext
from haiku.rag.config.models import AppConfig
from haiku.rag.store.compression import decompress_json

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG


@dataclass
class SandboxResult:
    """Result of executing code in the sandbox."""

    stdout: str
    stderr: str
    success: bool


class Sandbox:
    """Execute code in a sandboxed Python interpreter.

    Uses pydantic-monty, a minimal secure Python interpreter written in Rust.
    External functions (search, list_documents, etc.) are called by Monty code
    using ``await`` and resolved asynchronously on the host.

        sandbox = Sandbox(client, config, context)
        result = await sandbox.execute("print('hello')")
    """

    _client: "HaikuRAG"
    _config: AppConfig
    _context: RLMContext

    def __init__(
        self,
        client: "HaikuRAG",
        config: AppConfig,
        context: RLMContext,
    ):
        self._client = client
        self._config = config
        self._context = context

    def _build_external_functions(self) -> dict[str, Any]:
        """Build async external functions for the Monty interpreter."""
        client = self._client
        config = self._config
        context = self._context

        async def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
            results = await client.search(query, limit=limit, filter=context.filter)
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
                }
                for r in results
            ]

        async def list_documents(
            limit: int = 10, offset: int = 0
        ) -> list[dict[str, Any]]:
            docs = await client.list_documents(
                limit=limit, offset=offset, filter=context.filter
            )
            return [
                {
                    "id": d.id,
                    "title": d.title,
                    "uri": d.uri,
                    "created_at": str(d.created_at),
                }
                for d in docs
            ]

        async def get_document(id_or_title: str) -> str | None:
            doc = await client.resolve_document(id_or_title)
            return doc.content if doc else None

        async def get_chunk(chunk_id: str) -> dict[str, Any] | None:
            chunk = await client.get_chunk_by_id(chunk_id)
            if not chunk:
                return None
            meta = chunk.get_chunk_metadata()
            doc_title = chunk.document_title
            if not doc_title and chunk.document_id:
                doc = await client.get_document_by_id(chunk.document_id)
                if doc:
                    doc_title = doc.title
            return {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "document_id": chunk.document_id,
                "document_title": doc_title,
                "headings": meta.headings,
                "page_numbers": meta.page_numbers,
                "labels": meta.labels,
            }

        async def get_docling_document(
            document_id: str,
        ) -> dict[str, Any] | None:
            doc = await client.get_document_by_id(document_id)
            if not doc or not doc.docling_document:
                return None
            json_str = decompress_json(doc.docling_document)
            return json.loads(json_str)

        async def llm(prompt: str) -> str:
            from pydantic_ai import Agent

            from haiku.rag.utils import get_model

            model = get_model(config.rlm.model, config)
            agent: Agent[None, str] = Agent(model, output_type=str)
            result = await agent.run(prompt)
            return result.output

        return {
            "search": search,
            "list_documents": list_documents,
            "get_document": get_document,
            "get_chunk": get_chunk,
            "get_docling_document": get_docling_document,
            "llm": llm,
        }

    async def execute(self, code: str) -> SandboxResult:
        """Execute Python code in the Monty interpreter."""
        external_fns = self._build_external_functions()

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
                external_functions=list(external_fns.keys()),
            )
        except (
            pydantic_monty.MontySyntaxError,
            pydantic_monty.MontyRuntimeError,
        ) as e:
            return SandboxResult(stdout="", stderr=str(e), success=False)

        stdout_lines: list[str] = []

        def print_callback(_stream: Literal["stdout"], text: str) -> None:
            stdout_lines.append(text)

        max_chars = self._config.rlm.max_output_chars
        limits: pydantic_monty.ResourceLimits = {
            "max_duration_secs": self._config.rlm.code_timeout,
        }

        try:
            output = await pydantic_monty.run_monty_async(
                monty,
                inputs=inputs,
                external_functions=external_fns,
                limits=limits,
                print_callback=print_callback,
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

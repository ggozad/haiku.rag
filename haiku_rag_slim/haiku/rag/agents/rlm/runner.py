"""Entry point for sandboxed code execution in Docker container."""

import asyncio
import json
import sys
import traceback
from io import StringIO
from typing import Any


def build_namespace(
    client: Any, config: Any, context: Any, loop: asyncio.AbstractEventLoop
) -> dict[str, Any]:
    """Build execution namespace with haiku.rag functions injected."""
    from haiku.rag.store.repositories.document import _escape_sql_string

    def run_async(coro: Any) -> Any:
        """Run async coroutine from sync context using thread-safe scheduling."""
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=config.rlm.code_timeout)

    def search(query: str, limit: int = 10) -> list[dict]:
        async def _search() -> Any:
            return await client.search(query, limit=limit, filter=context.filter)

        results = run_async(_search())
        context.search_results.extend(results)
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

    def list_documents(limit: int = 10, offset: int = 0) -> list[dict]:
        async def _list() -> Any:
            return await client.list_documents(
                limit=limit, offset=offset, filter=context.filter
            )

        docs = run_async(_list())
        return [
            {
                "id": d.id,
                "title": d.title,
                "uri": d.uri,
                "created_at": str(d.created_at),
            }
            for d in docs
        ]

    def get_document(id_or_title: str) -> str | None:
        async def _get() -> str | None:
            doc = await client.get_document_by_id(id_or_title)
            if doc:
                return doc.content
            safe_input = _escape_sql_string(id_or_title)
            docs = await client.list_documents(filter=f"title = '{safe_input}'")
            if docs and docs[0].id:
                full_doc = await client.get_document_by_id(docs[0].id)
                return full_doc.content if full_doc else None
            docs = await client.list_documents(filter=f"uri = '{safe_input}'")
            if docs and docs[0].id:
                full_doc = await client.get_document_by_id(docs[0].id)
                return full_doc.content if full_doc else None
            return None

        return run_async(_get())

    def get_docling_document(id_or_title: str) -> Any:
        async def _get() -> Any:
            doc = await client.get_document_by_id(id_or_title)
            if doc:
                return doc.get_docling_document()
            safe_input = _escape_sql_string(id_or_title)
            docs = await client.list_documents(filter=f"title = '{safe_input}'")
            if docs and docs[0].id:
                full_doc = await client.get_document_by_id(docs[0].id)
                return full_doc.get_docling_document() if full_doc else None
            docs = await client.list_documents(filter=f"uri = '{safe_input}'")
            if docs and docs[0].id:
                full_doc = await client.get_document_by_id(docs[0].id)
                return full_doc.get_docling_document() if full_doc else None
            return None

        return run_async(_get())

    def llm(prompt: str) -> str:
        async def _llm() -> str:
            from pydantic_ai import Agent

            from haiku.rag.utils import get_model

            model = get_model(config.rlm.model, config)
            agent: Agent[None, str] = Agent(model, output_type=str)
            result = await agent.run(prompt)
            return result.output

        return run_async(_llm())

    namespace: dict[str, Any] = {
        "search": search,
        "list_documents": list_documents,
        "get_document": get_document,
        "get_docling_document": get_docling_document,
        "llm": llm,
    }

    if context.documents:
        namespace["documents"] = [
            {"id": d.id, "title": d.title, "uri": d.uri, "content": d.content}
            for d in context.documents
        ]

    return namespace


def execute_code(
    code: str, namespace: dict[str, Any], max_output_chars: int
) -> dict[str, Any]:
    """Execute code and capture output."""
    stdout_capture = StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = stdout_capture
        exec(code, namespace)
        stdout = stdout_capture.getvalue()
        if len(stdout) > max_output_chars:
            stdout = stdout[:max_output_chars] + "\n... (output truncated)"
        return {
            "success": True,
            "stdout": stdout,
            "stderr": "",
        }
    except Exception:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue(),
            "stderr": traceback.format_exc(),
        }
    finally:
        sys.stdout = original_stdout


def send_response(result: dict[str, Any]) -> None:
    """Send length-prefixed JSON response."""
    response = json.dumps(result)
    sys.stdout.write(f"{len(response)}\n")
    sys.stdout.write(response)
    sys.stdout.flush()


async def main() -> None:
    """Main entry point for container execution.

    Runs a loop reading length-prefixed JSON messages and executing code.
    """
    import concurrent.futures
    import os
    from pathlib import Path

    from haiku.rag.agents.rlm.dependencies import RLMContext
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import get_config

    config = get_config()
    db_path = Path(os.environ.get("HAIKU_DB_PATH", "/data/db.lancedb"))
    filter_expr = os.environ.get("HAIKU_FILTER")
    context = RLMContext(filter=filter_expr)
    max_output_chars = config.rlm.max_output_chars

    loop = asyncio.get_running_loop()

    async with HaikuRAG(db_path, config=config, read_only=True) as client:
        namespace = build_namespace(client, config, context, loop)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                # Read length-prefixed message
                length_line = sys.stdin.readline()
                if not length_line:
                    break

                try:
                    length = int(length_line.strip())
                    message = sys.stdin.read(length)
                    request = json.loads(message)
                    code = request.get("code", "")

                    result = await loop.run_in_executor(
                        executor, execute_code, code, namespace, max_output_chars
                    )
                    send_response(result)

                except (ValueError, json.JSONDecodeError) as e:
                    send_response(
                        {
                            "success": False,
                            "stdout": "",
                            "stderr": f"Invalid request: {e}",
                        }
                    )


if __name__ == "__main__":
    asyncio.run(main())

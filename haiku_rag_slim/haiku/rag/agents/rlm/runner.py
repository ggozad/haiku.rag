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

    def run_async(coro: Any) -> Any:
        """Run async coroutine from sync context using thread-safe scheduling."""
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=config.rlm.code_timeout)

    def search(query: str, limit: int = 10) -> list[dict]:
        async def _search() -> Any:
            return await client.search(query, limit=limit, filter=context.filter)

        results = run_async(_search())
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
            doc = await client.resolve_document(id_or_title)
            return doc.content if doc else None

        return run_async(_get())

    def get_docling_document(id_or_title: str) -> Any:
        async def _get() -> Any:
            doc = await client.resolve_document(id_or_title)
            return doc.get_docling_document() if doc else None

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


def send_response(conn: Any, result: dict[str, Any]) -> None:
    """Send length-prefixed JSON response over TCP socket."""
    response = json.dumps(result)
    data = f"{len(response)}\n{response}".encode()
    conn.sendall(data)


def read_message(conn: Any) -> str | None:
    """Read a length-prefixed JSON message from TCP socket."""
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            return None
        buf += chunk

    newline_idx = buf.index(b"\n")
    length = int(buf[:newline_idx].strip())
    buf = buf[newline_idx + 1 :]

    while len(buf) < length:
        chunk = conn.recv(4096)
        if not chunk:
            return None
        buf += chunk

    return buf[:length].decode()


async def main() -> None:
    """Main entry point for container execution.

    Starts a TCP server, prints the port for the host to discover,
    then runs a loop reading length-prefixed JSON messages and executing code.
    """
    import concurrent.futures
    import os
    import socket
    from pathlib import Path

    from haiku.rag.agents.rlm.dependencies import RLMContext
    from haiku.rag.client import HaikuRAG
    from haiku.rag.config import get_config

    config = get_config()
    db_path = Path(os.environ.get("HAIKU_DB_PATH", "/data/db.lancedb"))
    filter_expr = os.environ.get("HAIKU_FILTER")
    context = RLMContext(filter=filter_expr)
    max_output_chars = config.rlm.max_output_chars

    bind_port = int(os.environ.get("HAIKU_SANDBOX_PORT", "0"))

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", bind_port))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]

    sys.stdout.write(f"PORT:{port}\n")
    sys.stdout.flush()

    conn, _ = server_sock.accept()
    server_sock.close()

    loop = asyncio.get_running_loop()

    async with HaikuRAG(db_path, config=config, read_only=True) as client:
        namespace = build_namespace(client, config, context, loop)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                message = read_message(conn)
                if message is None:
                    break

                try:
                    request = json.loads(message)
                    code = request.get("code", "")

                    result = await loop.run_in_executor(
                        executor, execute_code, code, namespace, max_output_chars
                    )
                    send_response(conn, result)

                except (ValueError, json.JSONDecodeError) as e:
                    send_response(
                        conn,
                        {
                            "success": False,
                            "stdout": "",
                            "stderr": f"Invalid request: {e}",
                        },
                    )

    conn.close()


if __name__ == "__main__":
    asyncio.run(main())

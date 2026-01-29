import ast
import asyncio
import concurrent.futures
import sys
import traceback
from io import StringIO
from typing import TYPE_CHECKING, Any

from haiku.rag.agents.rlm.dependencies import RLMContext
from haiku.rag.config.models import RLMConfig

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG


class REPLResult:
    """Result of executing code in the REPL environment."""

    def __init__(
        self, stdout: str, stderr: str, success: bool, locals_: dict | None = None
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.success = success
        self.locals = locals_ or {}

    def __repr__(self) -> str:
        return f"REPLResult(success={self.success}, stdout={self.stdout!r}, stderr={self.stderr!r})"


class REPLEnvironment:
    """Sandboxed Python execution environment with haiku.rag access."""

    SAFE_BUILTINS: dict[str, Any] = {
        "True": True,
        "False": False,
        "None": None,
        "__build_class__": __builtins__["__build_class__"]
        if isinstance(__builtins__, dict)
        else getattr(__builtins__, "__build_class__"),
        "abs": abs,
        "all": all,
        "any": any,
        "ascii": ascii,
        "bin": bin,
        "bool": bool,
        "bytearray": bytearray,
        "bytes": bytes,
        "callable": callable,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hash": hash,
        "hex": hex,
        "id": id,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "object": object,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "RuntimeError": RuntimeError,
        "StopIteration": StopIteration,
        "ZeroDivisionError": ZeroDivisionError,
        "AssertionError": AssertionError,
    }

    ALLOWED_IMPORTS = {
        "json",
        "re",
        "collections",
        "math",
        "statistics",
        "itertools",
        "functools",
        "datetime",
        "typing",
    }

    def __init__(
        self,
        client: "HaikuRAG",
        config: RLMConfig,
        context: RLMContext,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ):
        self.client = client
        self.config = config
        self.context = context
        self._event_loop = event_loop
        self._setup_namespace()

    def _run_async_from_thread(self, coro):
        """Run async coroutine from a worker thread using run_coroutine_threadsafe."""
        if self._event_loop is None:
            raise RuntimeError("Event loop not set. Cannot call async functions.")
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return future.result(timeout=self.config.code_timeout)

    def _setup_namespace(self) -> None:
        """Build execution namespace with haiku.rag functions."""
        self.globals: dict[str, Any] = {
            "__builtins__": dict(self.SAFE_BUILTINS),
            "__name__": "__sandbox__",
            "search": self._make_search(),
            "list_documents": self._make_list_documents(),
            "get_document": self._make_get_document(),
            "get_docling_document": self._make_get_docling_document(),
            "ask": self._make_ask(),
        }
        self.locals: dict[str, Any] = {}

        if self.context.documents:
            self.globals["documents"] = [
                {"id": d.id, "title": d.title, "uri": d.uri, "content": d.content}
                for d in self.context.documents
            ]

    def _make_search(self):
        """Create sync search function that bridges to async client."""

        def search(query: str, limit: int = 10) -> list[dict]:
            async def _search():
                return await self.client.search(
                    query, limit=limit, filter=self.context.filter
                )

            results = self._run_async_from_thread(_search())
            self.context.search_results.extend(results)
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

        return search

    def _make_list_documents(self):
        """Create sync list_documents function."""

        def list_documents(limit: int = 10, offset: int = 0) -> list[dict]:
            async def _list():
                return await self.client.list_documents(
                    limit=limit, offset=offset, filter=self.context.filter
                )

            docs = self._run_async_from_thread(_list())
            return [
                {
                    "id": d.id,
                    "title": d.title,
                    "uri": d.uri,
                    "created_at": str(d.created_at),
                }
                for d in docs
            ]

        return list_documents

    def _make_get_document(self):
        """Create sync get_document function that returns text content."""

        def get_document(id_or_title: str) -> str | None:
            async def _get():
                doc = await self.client.get_document_by_id(id_or_title)
                if doc:
                    return doc.content
                docs = await self.client.list_documents(
                    filter=f"title = '{id_or_title}'"
                )
                if docs and docs[0].id:
                    full_doc = await self.client.get_document_by_id(docs[0].id)
                    return full_doc.content if full_doc else None
                docs = await self.client.list_documents(filter=f"uri = '{id_or_title}'")
                if docs and docs[0].id:
                    full_doc = await self.client.get_document_by_id(docs[0].id)
                    return full_doc.content if full_doc else None
                return None

            return self._run_async_from_thread(_get())

        return get_document

    def _make_get_docling_document(self):
        """Create sync get_docling_document function that returns DoclingDocument."""

        def get_docling_document(id_or_title: str):
            async def _get():
                doc = await self.client.get_document_by_id(id_or_title)
                if doc:
                    return doc.get_docling_document()
                docs = await self.client.list_documents(
                    filter=f"title = '{id_or_title}'"
                )
                if docs and docs[0].id:
                    full_doc = await self.client.get_document_by_id(docs[0].id)
                    return full_doc.get_docling_document() if full_doc else None
                docs = await self.client.list_documents(filter=f"uri = '{id_or_title}'")
                if docs and docs[0].id:
                    full_doc = await self.client.get_document_by_id(docs[0].id)
                    return full_doc.get_docling_document() if full_doc else None
                return None

            return self._run_async_from_thread(_get())

        return get_docling_document

    def _make_ask(self):
        """Create sync ask function that uses QA agent."""

        def ask(question: str) -> str:
            async def _ask():
                answer, citations = await self.client.ask(
                    question, filter=self.context.filter
                )
                for c in citations:
                    for sr in self.context.search_results:
                        if sr.chunk_id == c.chunk_id:
                            break
                    else:
                        from haiku.rag.store.models import SearchResult

                        self.context.search_results.append(
                            SearchResult(
                                chunk_id=c.chunk_id,
                                document_id=c.document_id,
                                document_title=c.document_title or "",
                                document_uri=c.document_uri,
                                content=c.content,
                                score=1.0,
                                page_numbers=c.page_numbers,
                                headings=c.headings or [],
                            )
                        )
                return answer

            return self._run_async_from_thread(_ask())

        return ask

    def _safe_import(
        self,
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple = (),
        level: int = 0,
    ):
        """Import hook that only allows safe modules."""
        base_module = name.split(".")[0]
        if base_module not in self.ALLOWED_IMPORTS:
            raise ImportError(f"Import of '{name}' is not allowed in sandbox")

        import importlib

        module = importlib.import_module(name)
        if fromlist:
            for attr in fromlist:
                if not hasattr(module, attr):
                    raise ImportError(f"cannot import name '{attr}' from '{name}'")
            return module
        return module

    def _validate_code(self, code: str) -> None:
        """Validate code AST for security issues."""
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_") and node.attr not in (
                    "__init__",
                    "__str__",
                    "__repr__",
                    "__class__",
                    "__name__",
                    "__doc__",
                    "__dict__",
                ):
                    raise SecurityError(
                        f"Access to private/dunder attribute '{node.attr}' is not allowed"
                    )

    def _execute_sync(self, code: str) -> REPLResult:
        """Internal synchronous execution - must be called from executor thread."""
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            self._validate_code(code)
        except SyntaxError as e:
            return REPLResult(
                stdout="",
                stderr=f"SyntaxError: {e}",
                success=False,
            )
        except SecurityError as e:
            return REPLResult(
                stdout="",
                stderr=str(e),
                success=False,
            )

        exec_globals = dict(self.globals)
        exec_globals["__builtins__"] = dict(self.SAFE_BUILTINS)
        exec_globals["__builtins__"]["__import__"] = self._safe_import

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            exec(code, exec_globals, self.locals)

            for key, value in self.locals.items():
                if not key.startswith("_"):
                    self.globals[key] = value

            stdout = stdout_capture.getvalue()
            if len(stdout) > self.config.max_output_chars:
                stdout = (
                    stdout[: self.config.max_output_chars] + "\n... (output truncated)"
                )

            return REPLResult(
                stdout=stdout,
                stderr=stderr_capture.getvalue(),
                success=True,
                locals_=dict(self.locals),
            )

        except Exception:
            tb = traceback.format_exc()
            return REPLResult(
                stdout=stdout_capture.getvalue(),
                stderr=tb,
                success=False,
            )

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    def execute(self, code: str) -> REPLResult:
        """Execute code in sandbox synchronously.

        This method runs code directly in the current thread.
        For async contexts, use execute_async() instead.
        """
        return self._execute_sync(code)

    async def execute_async(self, code: str) -> REPLResult:
        """Execute code in sandbox from async context.

        Runs the synchronous code in a thread executor, allowing
        sandbox functions to call back to async client methods.
        """
        loop = asyncio.get_running_loop()
        self._event_loop = loop

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, self._execute_sync, code),
                timeout=self.config.code_timeout,
            )
        return result


class SecurityError(Exception):
    """Raised when sandbox security is violated."""

    pass

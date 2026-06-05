import asyncio
import threading
from pathlib import Path

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.sandbox import AnalysisContext, Sandbox, SandboxResult


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent / "cassettes" / "test_sandbox")


class TestSandboxBasics:
    """Test basic sandbox functionality."""

    @pytest.mark.asyncio
    async def test_execute_simple_code(self, sandbox):
        """Test executing simple code in the sandbox."""
        result = await sandbox.execute("print('hello world')")
        assert isinstance(result, SandboxResult)
        assert result.success
        assert "hello world" in result.stdout
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_execute_expression_output(self, sandbox):
        """Test that expression values are captured."""
        result = await sandbox.execute("1 + 2")
        assert result.success
        assert "3" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_print_and_expression(self, sandbox):
        """Test print output combined with expression value."""
        result = await sandbox.execute("print('hello')\n42")
        assert result.success
        assert "hello" in result.stdout
        assert "42" in result.stdout


class TestSandboxErrors:
    """Test error handling in sandbox."""

    @pytest.mark.asyncio
    async def test_syntax_error(self, sandbox):
        """Test that syntax errors are reported."""
        result = await sandbox.execute("def foo(")
        assert not result.success
        assert result.stderr != ""

    @pytest.mark.asyncio
    async def test_runtime_error(self, sandbox):
        """Test that runtime errors are reported."""
        result = await sandbox.execute("x = 1/0")
        assert not result.success
        assert "ZeroDivisionError" in result.stderr

    @pytest.mark.asyncio
    async def test_name_error(self, sandbox):
        """Test that name errors are reported."""
        result = await sandbox.execute("print(undefined_variable)")
        assert not result.success
        assert "NameError" in result.stderr

    @pytest.mark.asyncio
    async def test_multi_module_import(self, sandbox):
        """Test that unsupported multi-module imports are caught gracefully."""
        result = await sandbox.execute("import json, string")
        assert not result.success
        assert result.stderr != ""


class TestSandboxListDocuments:
    """Test list_documents function in sandbox."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, sandbox):
        """list_documents returns empty list for empty database."""
        result = await sandbox.execute(
            "docs = await list_documents()\nprint(type(docs).__name__, len(docs))"
        )
        assert result.success
        assert "list 0" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_list_documents_with_data(self, temp_db_path):
        """list_documents returns documents when populated."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="Test content",
                uri="test://doc1",
                title="Test Document",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "docs = await list_documents()\n"
                "print(len(docs))\n"
                "print(docs[0]['title'])"
            )
            assert result.success
            assert "1" in result.stdout
            assert "Test Document" in result.stdout


class TestSandboxSearch:
    """Test search function in sandbox."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_search_with_data(self, temp_db_path):
        """Test search function works."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="The quick brown fox jumps over the lazy dog.",
                uri="test://animals",
                title="Animals",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "results = await search('fox', limit=5)\n"
                "print(len(results))\n"
                "if results:\n"
                "    print('fox' in results[0]['content'].lower())"
            )
            assert result.success
            assert "True" in result.stdout or "1" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_search_returns_doc_item_refs_and_labels(self, temp_db_path):
        """Search results include doc_item_refs and labels."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="The quick brown fox jumps over the lazy dog.",
                uri="test://animals",
                title="Animals",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "results = await search('fox', limit=1)\n"
                "r = results[0]\n"
                "print('doc_item_refs' in r)\n"
                "print('labels' in r)\n"
                "print(type(r['doc_item_refs']).__name__)\n"
                "print(type(r['labels']).__name__)"
            )
            assert result.success
            assert "True\nTrue" in result.stdout
            assert "list\nlist" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_search_returns_expanded_content(self, temp_db_path):
        """search() returns context-expanded results."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="The quick brown fox jumps over the lazy dog.",
                uri="test://animals",
                title="Animals",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "results = await search('fox', limit=1)\n"
                "print(type(results[0]['content']).__name__)\n"
                "print('fox' in results[0]['content'].lower())"
            )
            assert result.success
            assert "str" in result.stdout
            assert "True" in result.stdout


class TestSandboxExternalFunctionEdgeCases:
    """Test edge cases in external function dispatch."""

    @pytest.mark.asyncio
    async def test_unknown_external_function(self, sandbox):
        """Test that calling an unregistered external function resumes with KeyError."""
        original_build = sandbox._build_external_functions

        def patched_build():
            fns = original_build()
            fns["search"] = None
            return fns

        sandbox._build_external_functions = patched_build

        result = await sandbox.execute(
            "try:\n"
            "    await search('hello')\n"
            "except:\n"
            "    print('caught')\n"
            "print('done')"
        )
        assert result.success
        assert "caught" in result.stdout
        assert "done" in result.stdout

    @pytest.mark.asyncio
    async def test_external_function_raises_exception(self, sandbox):
        """Test that exceptions from async external functions surface as errors.

        With run_monty_async, exceptions from async external functions
        propagate as MontyRuntimeError rather than being catchable inside
        Monty's try/except.
        """
        original_build = sandbox._build_external_functions

        def patched_build():
            fns = original_build()

            async def failing_search(*args, **kwargs):
                raise ValueError("external error")

            fns["search"] = failing_search
            return fns

        sandbox._build_external_functions = patched_build

        result = await sandbox.execute("await search('hello')")
        assert not result.success
        assert "external error" in result.stderr


class TestSandboxOutputTruncation:
    """Test output truncation behavior."""

    @pytest.mark.asyncio
    async def test_truncate_stdout_on_runtime_error(self, temp_db_path):
        """Test stdout is truncated when a runtime error occurs after large output."""
        async with HaikuRAG(temp_db_path, create=True):
            config = AppConfig()
            config.analysis.max_output_chars = 20
            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute("print('a' * 100)\nx = 1/0")
            assert not result.success
            assert "ZeroDivisionError" in result.stderr
            assert result.stdout.endswith("... (output truncated)")
            assert len(result.stdout) < 100

    @pytest.mark.asyncio
    async def test_truncate_successful_output(self, temp_db_path):
        """Test output is truncated on successful execution with large output."""
        async with HaikuRAG(temp_db_path, create=True):
            config = AppConfig()
            config.analysis.max_output_chars = 20
            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute("print('b' * 100)")
            assert result.success
            assert result.stdout.endswith("... (output truncated)")
            assert len(result.stdout) < 100


class TestSandboxVFS:
    """Test virtual filesystem for document access."""

    @pytest.mark.asyncio
    async def test_empty_database_has_no_documents(self, sandbox):
        """Empty database has no document directories."""
        result = await sandbox.execute(
            "from pathlib import Path\nprint(Path('/documents').exists())"
        )
        assert result.success
        # /documents dir may or may not exist when empty, both are valid
        # The key is it doesn't error

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_iterdir_discovers_documents(self, temp_db_path):
        """Path('/documents').iterdir() lists document directories."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="Test content",
                uri="test://doc1",
                title="Test Document",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "from pathlib import Path\n"
                "dirs = list(Path('/documents').iterdir())\n"
                "print(len(dirs))\n"
                "print(dirs[0].is_dir())"
            )
            assert result.success
            assert "1" in result.stdout
            assert "True" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_metadata_json(self, temp_db_path):
        """metadata.json contains document title and uri."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Test content",
                uri="test://doc1",
                title="Test Document",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "from pathlib import Path\n"
                "import json\n"
                f"meta = json.loads(Path('/documents/{doc.id}/metadata.json').read_text())\n"
                "print(meta['title'])\n"
                "print(meta['uri'])"
            )
            assert result.success
            assert "Test Document" in result.stdout
            assert "test://doc1" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_content_txt(self, temp_db_path):
        """content.txt returns full document text (lazy loaded)."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Content about foxes and dogs.",
                uri="test://doc",
                title="Fox Document",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "from pathlib import Path\n"
                f"content = Path('/documents/{doc.id}/content.txt').read_text()\n"
                "print('foxes' in content.lower())"
            )
            assert result.success
            assert "True" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_items_jsonl(self, temp_db_path):
        """items.jsonl returns document items as JSONL (lazy loaded)."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="The quick brown fox jumps over the lazy dog.",
                uri="test://animals",
                title="Animals",
            )

            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "from pathlib import Path\n"
                "import json\n"
                f"text = Path('/documents/{doc.id}/items.jsonl').read_text()\n"
                "lines = text.strip().split('\\n')\n"
                "print(len(lines) > 0)\n"
                "item = json.loads(lines[0])\n"
                "print('self_ref' in item)\n"
                "print('label' in item)\n"
                "print('text' in item)\n"
                "print('page_numbers' in item)\n"
                "print('chunk_ids' in item)"
            )
            assert result.success
            assert result.stdout.count("True") == 6

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_context_filter_limits_vfs(self, temp_db_path):
        """Context filter restricts which documents appear in VFS."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="Public content",
                uri="public://doc1",
                title="Public Doc",
            )
            await client.create_document(
                content="Private content",
                uri="private://doc2",
                title="Private Doc",
            )

            context = AnalysisContext(filter="uri LIKE 'public://%'")
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "from pathlib import Path\n"
                "import json\n"
                "dirs = list(Path('/documents').iterdir())\n"
                "print(len(dirs))\n"
                "meta = json.loads((dirs[0] / 'metadata.json').read_text())\n"
                "print(meta['title'])"
            )
            assert result.success
            assert "1" in result.stdout
            assert "Public Doc" in result.stdout
            assert "Private Doc" not in result.stdout


class TestSandboxHeldConnection:
    """VFS reads must work while another connection to the same DB stays open.

    Mirrors the analysis-skill lifespan, which keeps a read-only connection open
    for the whole turn while sandboxed code reads the document VFS.
    """

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_vfs_reads_while_connection_held(self, temp_db_path):
        """All three VFS readers work while another connection to the DB is open."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs roam the quiet hills.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        async with HaikuRAG(temp_db_path, config=config, read_only=True):
            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            try:
                result = await sb.execute(
                    "from pathlib import Path\n"
                    f"d = Path('/documents/{doc.id}')\n"
                    "print('foxes' in (d / 'content.txt').read_text().lower())\n"
                    "print(len((d / 'items.jsonl').read_text().strip().split('\\n')))\n"
                    "print('tree' in (d / 'toc.json').read_text())"
                )
                assert result.success, result.stderr
                lines = result.stdout.strip().split("\n")
                assert lines[0] == "True"
                assert int(lines[1]) > 0
                assert lines[2] == "True"
            finally:
                sb.close()

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_vfs_reads_use_injected_connection(self, temp_db_path):
        """An injected connection services VFS reads on the calling loop.

        No dedicated background loop/thread is spawned: all DB access for the
        sandbox runs on the loop driving execute(), through the one connection.
        """
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs roam the quiet hills.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        async with HaikuRAG(temp_db_path, config=config, read_only=True) as rag:
            sb = Sandbox(
                db_path=temp_db_path,
                config=config,
                context=AnalysisContext(),
                rag=rag,
            )
            result = await sb.execute(
                "from pathlib import Path\n"
                f"print(Path('/documents/{doc.id}/content.txt').read_text())"
            )
            assert result.success, result.stderr
            assert "Foxes" in result.stdout
            assert not any(t.name == "sandbox-vfs" for t in threading.enumerate())

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_vfs_read_concurrent_with_connection_read(self, temp_db_path):
        """A VFS read and a direct read on the shared connection run together.

        Both serialize through the shared lock, so concurrent tasks never have
        two operations in flight on the one connection at once.
        """
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs roam the quiet hills.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        async with HaikuRAG(temp_db_path, config=config, read_only=True) as rag:
            lock = asyncio.Lock()
            sb = Sandbox(
                db_path=temp_db_path,
                config=config,
                context=AnalysisContext(),
                rag=rag,
                lock=lock,
            )
            read_code = (
                "from pathlib import Path\n"
                f"print(Path('/documents/{doc.id}/content.txt').read_text())"
            )

            async def direct_read() -> str | None:
                async with lock:
                    return await rag.document_repository.get_content(doc.id)

            exec_result, content = await asyncio.gather(
                sb.execute(read_code),
                direct_read(),
            )
            assert exec_result.success, exec_result.stderr
            assert "Foxes" in exec_result.stdout
            assert content and "Foxes" in content

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_vfs_reads_repeatable_across_executes(self, temp_db_path):
        """Repeated VFS reads across execute() calls return consistent content."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        context = AnalysisContext()
        sb = Sandbox(db_path=temp_db_path, config=config, context=context)
        try:
            read = (
                "from pathlib import Path\n"
                f"print(Path('/documents/{doc.id}/content.txt').read_text())"
            )
            first = await sb.execute(read)
            second = await sb.execute(read)
            assert first.success and second.success
            assert "Foxes" in first.stdout
            assert first.stdout == second.stdout
            assert not any(t.name == "sandbox-vfs" for t in threading.enumerate())
        finally:
            sb.close()

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_variables_persist_across_executes_after_vfs_read(self, temp_db_path):
        """REPL state persists across execute() calls, including after a VFS read."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        context = AnalysisContext()
        sb = Sandbox(db_path=temp_db_path, config=config, context=context)
        try:
            first = await sb.execute(
                "from pathlib import Path\n"
                f"x = len(Path('/documents/{doc.id}/content.txt').read_text())"
            )
            assert first.success, first.stderr
            second = await sb.execute("print(x)")
            assert second.success, second.stderr
            assert int(second.stdout.strip()) > 0
        finally:
            sb.close()

    @pytest.mark.asyncio
    async def test_close_is_safe_without_vfs_read(self, temp_db_path):
        """close() is a no-op (and safe to call twice) when no VFS read happened."""
        async with HaikuRAG(temp_db_path, create=True):
            config = AppConfig()
            sb = Sandbox(db_path=temp_db_path, config=config, context=AnalysisContext())
            sb.close()
            sb.close()

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_close_is_idempotent_after_vfs_read(self, temp_db_path):
        """close() is a safe no-op after a VFS read; no background thread lingers."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Foxes and dogs.",
                uri="test://doc",
                title="Doc",
            )
        assert doc.id

        context = AnalysisContext()
        sb = Sandbox(db_path=temp_db_path, config=config, context=context)
        result = await sb.execute(
            "from pathlib import Path\n"
            f"print(Path('/documents/{doc.id}/content.txt').read_text())"
        )
        assert result.success
        assert not any(t.name == "sandbox-vfs" for t in threading.enumerate())
        sb.close()
        sb.close()

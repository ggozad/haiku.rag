from pathlib import Path

import pytest

from haiku.rag.agents.analysis.dependencies import AnalysisContext
from haiku.rag.agents.analysis.sandbox import Sandbox, SandboxResult
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models import Document


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_sandbox")


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
                "print('position' in item)\n"
                "print('self_ref' in item)\n"
                "print('label' in item)\n"
                "print('text' in item)\n"
                "print('page_numbers' in item)"
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


class TestSandboxPreloadedDocuments:
    """Test pre-loaded documents context variable."""

    @pytest.mark.asyncio
    async def test_documents_variable_not_available_without_preload(self, sandbox):
        """documents variable is not available when context.documents is None."""
        result = await sandbox.execute("print(documents)")
        assert not result.success
        assert "NameError" in result.stderr

    @pytest.mark.asyncio
    async def test_documents_variable_available_with_preload(self, temp_db_path):
        """documents variable is available when context.documents is set."""
        async with HaikuRAG(temp_db_path, create=True):
            config = AppConfig()
            docs = [
                Document(id="1", content="Content A", title="Doc A", uri="a://1"),
                Document(id="2", content="Content B", title="Doc B", uri="b://2"),
            ]
            context = AnalysisContext(documents=docs)
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "print(len(documents))\n"
                "print(documents[0]['title'])\n"
                "print(documents[1]['title'])"
            )
            assert result.success
            assert "2" in result.stdout
            assert "Doc A" in result.stdout
            assert "Doc B" in result.stdout


class TestSandboxLLM:
    """Test llm() external function."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_llm_function(self, allow_model_requests, temp_db_path):
        """Test llm() calls the model and returns a string."""
        async with HaikuRAG(temp_db_path, create=True):
            config = AppConfig()
            context = AnalysisContext()
            sb = Sandbox(db_path=temp_db_path, config=config, context=context)
            result = await sb.execute(
                "answer = await llm('What is 2 + 2? Reply with just the number.')\n"
                "print(answer)"
            )
            assert result.success
            assert "4" in result.stdout

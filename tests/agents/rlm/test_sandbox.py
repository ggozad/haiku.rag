from pathlib import Path

import pytest

from haiku.rag.agents.rlm.dependencies import RLMContext
from haiku.rag.agents.rlm.sandbox import Sandbox, SandboxResult
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


class TestSandboxHaikuRAG:
    """Test haiku.rag functions in sandbox."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, sandbox):
        """Test list_documents returns empty list for empty database."""
        result = await sandbox.execute(
            "docs = await list_documents()\nprint(type(docs).__name__, len(docs))"
        )
        assert result.success
        assert "list 0" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_list_documents_with_data(self, temp_db_path):
        """Test list_documents returns documents when populated."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="Test content",
                uri="test://doc1",
                title="Test Document",
            )

            context = RLMContext()
            sb = Sandbox(client=client, config=config, context=context)
            result = await sb.execute(
                "docs = await list_documents()\n"
                "print(len(docs))\n"
                "print(docs[0]['title'])"
            )
            assert result.success
            assert "1" in result.stdout
            assert "Test Document" in result.stdout

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

            context = RLMContext()
            sb = Sandbox(client=client, config=config, context=context)
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
    async def test_get_document(self, temp_db_path):
        """Test get_document function."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Content about foxes and dogs.",
                uri="test://doc",
                title="Fox Document",
            )

            context = RLMContext()
            sb = Sandbox(client=client, config=config, context=context)
            result = await sb.execute(
                f"content = await get_document('{doc.id}')\n"
                "print('foxes' in content.lower() if content else 'None')"
            )
            assert result.success
            assert "True" in result.stdout

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, sandbox):
        """Test get_document returns None for missing document."""
        result = await sandbox.execute(
            "content = await get_document('nonexistent-id')\nprint(content is None)"
        )
        assert result.success
        assert "True" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_get_chunk(self, temp_db_path):
        """Test get_chunk function returns chunk with metadata."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="Content about foxes and dogs.",
                uri="test://doc",
                title="Fox Document",
            )

            context = RLMContext()
            sb = Sandbox(client=client, config=config, context=context)
            # First search to get a chunk_id
            result = await sb.execute(
                "results = await search('foxes', limit=1)\n"
                "chunk_id = results[0]['chunk_id']\n"
                "chunk = await get_chunk(chunk_id)\n"
                "print(chunk['document_title'])\n"
                "print('content' in chunk)"
            )
            assert result.success
            assert "Fox Document" in result.stdout
            assert "True" in result.stdout

    @pytest.mark.asyncio
    async def test_get_chunk_not_found(self, sandbox):
        """Test get_chunk returns None for missing chunk."""
        result = await sandbox.execute(
            "chunk = await get_chunk('nonexistent-id')\nprint(chunk is None)"
        )
        assert result.success
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
    async def test_truncate_stdout_on_runtime_error(self, empty_client):
        """Test stdout is truncated when a runtime error occurs after large output."""
        config = AppConfig()
        config.rlm.max_output_chars = 20
        context = RLMContext()
        sb = Sandbox(client=empty_client, config=config, context=context)
        result = await sb.execute("print('a' * 100)\nx = 1/0")
        assert not result.success
        assert "ZeroDivisionError" in result.stderr
        assert result.stdout.endswith("... (output truncated)")
        assert len(result.stdout) < 100

    @pytest.mark.asyncio
    async def test_truncate_successful_output(self, empty_client):
        """Test output is truncated on successful execution with large output."""
        config = AppConfig()
        config.rlm.max_output_chars = 20
        context = RLMContext()
        sb = Sandbox(client=empty_client, config=config, context=context)
        result = await sb.execute("print('b' * 100)")
        assert result.success
        assert result.stdout.endswith("... (output truncated)")
        assert len(result.stdout) < 100


class TestSandboxContextFilter:
    """Test context filter is applied."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_filter_applied_to_list_documents(self, temp_db_path):
        """Test that context filter is passed to list_documents."""
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

            context = RLMContext(filter="uri LIKE 'public://%'")
            sb = Sandbox(client=client, config=config, context=context)
            result = await sb.execute(
                "docs = await list_documents()\n"
                "print(len(docs))\n"
                "if docs:\n"
                "    print(docs[0]['title'])"
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
    async def test_documents_variable_available_with_preload(self, empty_client):
        """documents variable is available when context.documents is set."""
        config = AppConfig()
        docs = [
            Document(id="1", content="Content A", title="Doc A", uri="a://1"),
            Document(id="2", content="Content B", title="Doc B", uri="b://2"),
        ]
        context = RLMContext(documents=docs)
        sb = Sandbox(client=empty_client, config=config, context=context)
        result = await sb.execute(
            "print(len(documents))\n"
            "print(documents[0]['title'])\n"
            "print(documents[1]['title'])"
        )
        assert result.success
        assert "2" in result.stdout
        assert "Doc A" in result.stdout
        assert "Doc B" in result.stdout


class TestSandboxDoclingDocument:
    """Test get_docling_document() external function."""

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_document(self, sandbox):
        """get_docling_document returns None for a non-existent document."""
        result = await sandbox.execute(
            "doc = await get_docling_document('nonexistent-id')\nprint(doc is None)"
        )
        assert result.success
        assert "True" in result.stdout

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_returns_dict_for_document_with_docling_data(self, temp_db_path):
        """get_docling_document returns a dict for a document with docling data."""
        config = AppConfig()
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Docling processed content",
                uri="test://docling",
                title="Docling Doc",
            )

            context = RLMContext()
            sb = Sandbox(client=client, config=config, context=context)
            result = await sb.execute(
                f"doc = await get_docling_document('{doc.id}')\n"
                "print(type(doc).__name__)\n"
                "print(doc['name'])\n"
                "print('texts' in doc)"
            )
            assert result.success
            assert "dict" in result.stdout
            assert "True" in result.stdout


class TestSandboxRegex:
    """Test regex external functions."""

    @pytest.mark.asyncio
    async def test_regex_findall(self, sandbox):
        """regex_findall extracts all matches."""
        result = await sandbox.execute(
            r"matches = await regex_findall(r'\d+', 'abc 123 def 456')"
            "\nprint(matches)"
        )
        assert result.success
        assert "['123', '456']" in result.stdout

    @pytest.mark.asyncio
    async def test_regex_sub(self, sandbox):
        """regex_sub replaces matches."""
        result = await sandbox.execute(
            r"out = await regex_sub(r'\d+', 'X', 'abc 123 def 456')"
            "\nprint(out)"
        )
        assert result.success
        assert "abc X def X" in result.stdout

    @pytest.mark.asyncio
    async def test_regex_search_match(self, sandbox):
        """regex_search returns match dict when pattern matches."""
        result = await sandbox.execute(
            r"m = await regex_search(r'(\d+)', 'abc 123')"
            "\nprint(m['group'])"
            "\nprint(m['start'])"
            "\nprint(m['end'])"
        )
        assert result.success
        assert "123" in result.stdout
        assert "4" in result.stdout
        assert "7" in result.stdout

    @pytest.mark.asyncio
    async def test_regex_search_no_match(self, sandbox):
        """regex_search returns None when pattern doesn't match."""
        result = await sandbox.execute(
            r"m = await regex_search(r'\d+', 'abc')"
            "\nprint(m is None)"
        )
        assert result.success
        assert "True" in result.stdout

    @pytest.mark.asyncio
    async def test_regex_split(self, sandbox):
        """regex_split splits on pattern."""
        result = await sandbox.execute(
            "out = await regex_split(',', 'a,b,,c')\nprint(out)"
        )
        assert result.success
        assert "['a', 'b', '', 'c']" in result.stdout

    @pytest.mark.asyncio
    async def test_regex_invalid_pattern(self, sandbox):
        """Invalid regex pattern surfaces as an error."""
        result = await sandbox.execute("await regex_findall('[invalid', 'text')")
        assert not result.success
        assert result.stderr != ""


class TestSandboxLLM:
    """Test llm() external function."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_llm_function(self, allow_model_requests, empty_client):
        """Test llm() calls the model and returns a string."""
        config = AppConfig()
        context = RLMContext()
        sb = Sandbox(client=empty_client, config=config, context=context)
        result = await sb.execute(
            "answer = await llm('What is 2 + 2? Reply with just the number.')\n"
            "print(answer)"
        )
        assert result.success
        assert "4" in result.stdout

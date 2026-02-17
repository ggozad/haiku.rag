from pathlib import Path

import pytest

from haiku.rag.agents.rlm.dependencies import RLMContext
from haiku.rag.agents.rlm.sandbox import Sandbox, SandboxResult
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig


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


class TestSandboxHaikuRAG:
    """Test haiku.rag functions in sandbox."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, sandbox):
        """Test list_documents returns empty list for empty database."""
        result = await sandbox.execute(
            "docs = list_documents()\nprint(type(docs).__name__, len(docs))"
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
            async with Sandbox(client=client, config=config, context=context) as sb:
                result = await sb.execute(
                    "docs = list_documents()\nprint(len(docs))\nprint(docs[0]['title'])"
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
            async with Sandbox(client=client, config=config, context=context) as sb:
                result = await sb.execute(
                    "results = search('fox', limit=5)\n"
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
            async with Sandbox(client=client, config=config, context=context) as sb:
                result = await sb.execute(
                    f"content = get_document('{doc.id}')\n"
                    "print('foxes' in content.lower() if content else 'None')"
                )
                assert result.success
                assert "True" in result.stdout

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, sandbox):
        """Test get_document returns None for missing document."""
        result = await sandbox.execute(
            "content = get_document('nonexistent-id')\nprint(content is None)"
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
            async with Sandbox(client=client, config=config, context=context) as sb:
                # First search to get a chunk_id
                result = await sb.execute(
                    "results = search('foxes', limit=1)\n"
                    "chunk_id = results[0]['chunk_id']\n"
                    "chunk = get_chunk(chunk_id)\n"
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
            "chunk = get_chunk('nonexistent-id')\nprint(chunk is None)"
        )
        assert result.success
        assert "True" in result.stdout


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
            async with Sandbox(client=client, config=config, context=context) as sb:
                result = await sb.execute(
                    "docs = list_documents()\n"
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

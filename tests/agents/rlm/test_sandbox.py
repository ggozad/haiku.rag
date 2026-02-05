from pathlib import Path

import pytest

from haiku.rag.agents.rlm.dependencies import RLMContext
from haiku.rag.agents.rlm.docker_sandbox import DockerSandbox, SandboxResult
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import RLMConfig


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_sandbox")


def is_docker_available() -> bool:
    """Check if Docker daemon is available."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


docker_required = pytest.mark.skipif(
    not is_docker_available(),
    reason="Docker daemon not available",
)


@pytest.mark.integration
class TestDockerSandboxBasics:
    """Test basic Docker sandbox functionality."""

    @docker_required
    @pytest.mark.asyncio
    async def test_execute_simple_code(self, docker_sandbox):
        """Test executing simple code in the sandbox."""
        result = await docker_sandbox.execute("print('hello world')")
        assert isinstance(result, SandboxResult)
        assert result.success
        assert "hello world" in result.stdout
        assert result.stderr == ""


@pytest.mark.integration
class TestDockerSandboxErrors:
    """Test error handling in Docker sandbox."""

    @docker_required
    @pytest.mark.asyncio
    async def test_syntax_error(self, docker_sandbox):
        """Test that syntax errors are reported."""
        result = await docker_sandbox.execute("def foo(")
        assert not result.success
        assert "SyntaxError" in result.stderr

    @docker_required
    @pytest.mark.asyncio
    async def test_runtime_error(self, docker_sandbox):
        """Test that runtime errors are reported."""
        result = await docker_sandbox.execute("x = 1/0")
        assert not result.success
        assert "ZeroDivisionError" in result.stderr

    @docker_required
    @pytest.mark.asyncio
    async def test_name_error(self, docker_sandbox):
        """Test that name errors are reported."""
        result = await docker_sandbox.execute("print(undefined_variable)")
        assert not result.success
        assert "NameError" in result.stderr

    @docker_required
    @pytest.mark.asyncio
    async def test_missing_image(self, temp_db_path):
        """Test error when Docker image is not found."""
        async with HaikuRAG(temp_db_path, create=True) as client:
            config = RLMConfig(docker_image="nonexistent-image:v999.999.999")
            context = RLMContext()
            async with DockerSandbox(
                client=client, config=config, context=context, image=config.docker_image
            ) as sandbox:
                result = await sandbox.execute("print('hello')")
                assert not result.success
                assert (
                    "not found" in result.stderr.lower()
                    or "error" in result.stderr.lower()
                )


@pytest.mark.integration
class TestDockerSandboxHaikuRAG:
    """Test haiku.rag functions in Docker sandbox."""

    @docker_required
    @pytest.mark.asyncio
    async def test_list_documents_empty(self, docker_sandbox):
        """Test list_documents returns empty list for empty database."""
        result = await docker_sandbox.execute(
            "docs = list_documents()\nprint(type(docs).__name__, len(docs))"
        )
        assert result.success
        assert "list 0" in result.stdout

    @docker_required
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_list_documents_with_data(self, temp_db_path, test_docker_image):
        """Test list_documents returns documents when populated."""
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="Test content",
                uri="test://doc1",
                title="Test Document",
            )

            config = RLMConfig(docker_image=test_docker_image)
            context = RLMContext()
            async with DockerSandbox(
                client=client, config=config, context=context, image=test_docker_image
            ) as sandbox:
                result = await sandbox.execute(
                    "docs = list_documents()\nprint(len(docs))\nprint(docs[0]['title'])"
                )
                assert result.success
                assert "1" in result.stdout
                assert "Test Document" in result.stdout

    @docker_required
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_search_with_data(self, temp_db_path, test_docker_image):
        """Test search function works."""
        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                content="The quick brown fox jumps over the lazy dog.",
                uri="test://animals",
                title="Animals",
            )

            config = RLMConfig(docker_image=test_docker_image)
            context = RLMContext()
            async with DockerSandbox(
                client=client, config=config, context=context, image=test_docker_image
            ) as sandbox:
                result = await sandbox.execute(
                    "results = search('fox', limit=5)\n"
                    "print(len(results))\n"
                    "if results:\n"
                    "    print('fox' in results[0]['content'].lower())"
                )
                assert result.success
                # Search should return at least one result
                assert "True" in result.stdout or "1" in result.stdout

    @docker_required
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_get_document(self, temp_db_path, test_docker_image):
        """Test get_document function."""
        async with HaikuRAG(temp_db_path, create=True) as client:
            doc = await client.create_document(
                content="Content about foxes and dogs.",
                uri="test://doc",
                title="Fox Document",
            )

            config = RLMConfig(docker_image=test_docker_image)
            context = RLMContext()
            async with DockerSandbox(
                client=client, config=config, context=context, image=test_docker_image
            ) as sandbox:
                result = await sandbox.execute(
                    f"content = get_document('{doc.id}')\n"
                    "print('foxes' in content.lower() if content else 'None')"
                )
                assert result.success
                assert "True" in result.stdout

    @docker_required
    @pytest.mark.asyncio
    async def test_get_document_not_found(self, docker_sandbox):
        """Test get_document returns None for missing document."""
        result = await docker_sandbox.execute(
            "content = get_document('nonexistent-id')\nprint(content is None)"
        )
        assert result.success
        assert "True" in result.stdout


@pytest.mark.integration
class TestDockerSandboxContextFilter:
    """Test context filter is applied."""

    @docker_required
    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_filter_applied_to_list_documents(
        self, temp_db_path, test_docker_image
    ):
        """Test that context filter is passed to list_documents."""
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

            config = RLMConfig(docker_image=test_docker_image)
            context = RLMContext(filter="uri LIKE 'public://%'")
            async with DockerSandbox(
                client=client, config=config, context=context, image=test_docker_image
            ) as sandbox:
                result = await sandbox.execute(
                    "docs = list_documents()\n"
                    "print(len(docs))\n"
                    "if docs:\n"
                    "    print(docs[0]['title'])"
                )
                assert result.success
                assert "1" in result.stdout
                assert "Public Doc" in result.stdout
                assert "Private Doc" not in result.stdout


@pytest.mark.integration
class TestDockerSandboxPreloadedDocuments:
    """Test pre-loaded documents context variable."""

    @docker_required
    @pytest.mark.asyncio
    async def test_documents_variable_not_available_without_preload(
        self, docker_sandbox
    ):
        """documents variable is not available when context.documents is None."""
        result = await docker_sandbox.execute("print(documents)")
        assert not result.success
        assert "NameError" in result.stderr

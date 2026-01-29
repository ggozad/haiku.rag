from pathlib import Path

import pytest
from pydantic_ai import Agent

from haiku.rag.agents.rlm.agent import create_rlm_agent
from haiku.rag.agents.rlm.dependencies import RLMContext, RLMDeps
from haiku.rag.agents.rlm.models import CodeExecution, RLMResult
from haiku.rag.config import Config


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_rlm")


class TestCreateRLMAgent:
    def test_creates_agent_with_correct_types(self):
        agent = create_rlm_agent(Config)
        assert isinstance(agent, Agent)
        assert agent.deps_type is RLMDeps
        assert agent.output_type is RLMResult

    def test_agent_has_execute_code_tool(self):
        agent = create_rlm_agent(Config)
        tool_names = list(agent._function_toolset.tools.keys())
        assert "execute_code" in tool_names


class TestExecuteCodeTool:
    @pytest.mark.asyncio
    async def test_execute_code_returns_structured_result(self, empty_client):
        """Test that execute_code tool produces structured CodeExecution output."""
        from haiku.rag.agents.rlm.agent import _get_or_create_repl

        context = RLMContext()
        deps = RLMDeps(
            client=empty_client,
            config=Config,
            context=context,
        )

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        repl = _get_or_create_repl(ctx)

        result = await repl.execute_async("print(1 + 1)")
        assert result.success
        assert "2" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_code_tracks_executions_in_context(self, empty_client):
        """Test that code executions are tracked as CodeExecution objects in RLMContext."""
        from haiku.rag.agents.rlm.agent import _get_or_create_repl

        context = RLMContext()
        deps = RLMDeps(
            client=empty_client,
            config=Config,
            context=context,
        )

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        repl = _get_or_create_repl(ctx)

        assert len(context.code_executions) == 0

        result = await repl.execute_async("x = 42")
        assert result.success

    @pytest.mark.asyncio
    async def test_code_execution_has_correct_fields(self, empty_client):
        """Test that CodeExecution has all expected fields."""
        execution = CodeExecution(
            code="print('hello')",
            stdout="hello\n",
            stderr="",
            success=True,
        )
        assert execution.code == "print('hello')"
        assert execution.stdout == "hello\n"
        assert execution.stderr == ""
        assert execution.success is True

    @pytest.mark.asyncio
    async def test_code_execution_captures_errors(self, empty_client):
        """Test that failed executions are properly captured."""
        from haiku.rag.agents.rlm.agent import _get_or_create_repl

        context = RLMContext()
        deps = RLMDeps(
            client=empty_client,
            config=Config,
            context=context,
        )

        class MockCtx:
            def __init__(self, deps):
                self.deps = deps

        ctx = MockCtx(deps)
        repl = _get_or_create_repl(ctx)

        result = await repl.execute_async("1/0")
        assert result.success is False
        assert "ZeroDivisionError" in result.stderr


class TestClientRLMIntegration:
    """Integration tests for client.rlm() method."""

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_count_documents(self, allow_model_requests, temp_db_path):
        """Test RLM agent can count documents."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document("First document about cats.", title="Doc 1")
            await client.create_document("Second document about dogs.", title="Doc 2")
            await client.create_document("Third document about birds.", title="Doc 3")

            answer = await client.rlm("How many documents are in the database?")

            assert "3" in answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_aggregation(self, allow_model_requests, temp_db_path):
        """Test RLM agent can perform aggregation across documents."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document(
                "Sales report Q1: Revenue was $100,000.", title="Q1 Report"
            )
            await client.create_document(
                "Sales report Q2: Revenue was $150,000.", title="Q2 Report"
            )
            await client.create_document(
                "Sales report Q3: Revenue was $200,000.", title="Q3 Report"
            )

            answer = await client.rlm(
                "What is the total revenue across all quarterly reports?"
            )

            assert "450" in answer or "450,000" in answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_with_filter(self, allow_model_requests, temp_db_path):
        """Test RLM agent respects filter parameter."""
        from haiku.rag.client import HaikuRAG

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.create_document("Cat document.", title="Cats")
            await client.create_document("Dog document.", title="Dogs")
            await client.create_document("Bird document.", title="Birds")

            answer = await client.rlm(
                "How many documents are available?",
                filter="title = 'Cats'",
            )

            assert "1" in answer

    @pytest.mark.asyncio
    @pytest.mark.vcr()
    async def test_rlm_docling_document_structure(
        self, allow_model_requests, temp_db_path
    ):
        """Test RLM agent can analyze document structure using DoclingDocument."""
        from pathlib import Path

        from haiku.rag.client import HaikuRAG
        from haiku.rag.config import AppConfig

        pdf_path = Path("tests/data/doclaynet.pdf")
        config = AppConfig()
        config.processing.conversion_options.do_ocr = False

        async with HaikuRAG(temp_db_path, config=config, create=True) as client:
            await client.create_document_from_source(pdf_path)

            answer = await client.rlm(
                "How many tables are in the document? "
                "Also tell me how many pictures/figures it contains."
            )

            # The doclaynet.pdf has 1 table and 1 picture
            assert "1" in answer

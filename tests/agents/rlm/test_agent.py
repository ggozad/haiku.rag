import pytest
from pydantic_ai import Agent

from haiku.rag.agents.rlm.agent import create_rlm_agent
from haiku.rag.agents.rlm.dependencies import RLMContext, RLMDeps
from haiku.rag.agents.rlm.models import CodeExecution, RLMResult
from haiku.rag.config import Config


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

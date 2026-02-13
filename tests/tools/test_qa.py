from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import ToolReturn

from haiku.rag.tools import ToolContext, prepare_context
from haiku.rag.tools.models import QAResult
from haiku.rag.tools.qa import (
    MAX_QA_HISTORY,
    QA_SESSION_NAMESPACE,
    QASessionState,
    create_qa_toolset,
    run_qa_core,
)
from haiku.rag.tools.session import SESSION_NAMESPACE, SessionState


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent / "cassettes" / "test_qa_tools")


def make_ctx(client, context=None):
    """Create a lightweight RunContext-like object for direct tool function calls."""
    return SimpleNamespace(deps=SimpleNamespace(client=client, tool_context=context))


class TestQAToolset:
    """Tests for create_qa_toolset."""

    def test_create_qa_toolset_returns_function_toolset(self, qa_config):
        """create_qa_toolset returns a FunctionToolset."""
        from pydantic_ai import FunctionToolset

        toolset = create_qa_toolset(qa_config)
        assert isinstance(toolset, FunctionToolset)

    def test_qa_toolset_has_ask_tool(self, qa_config):
        """The toolset includes an 'ask' tool."""
        toolset = create_qa_toolset(qa_config)
        assert "ask" in toolset.tools

    def test_qa_toolset_custom_tool_name(self, qa_config):
        """Toolset supports custom tool name."""
        toolset = create_qa_toolset(qa_config, tool_name="answer_question")
        assert "answer_question" in toolset.tools
        assert "ask" not in toolset.tools


@pytest.mark.vcr()
class TestRunQACore:
    """Tests for run_qa_core."""

    @pytest.mark.asyncio
    async def test_run_qa_core_with_session_state(
        self, allow_model_requests, qa_client, qa_config
    ):
        """run_qa_core with SessionState assigns citation indices via registry."""
        context = ToolContext()
        prepare_context(context, features=["qa"])

        result = await run_qa_core(
            client=qa_client,
            config=qa_config,
            question="What is Python?",
            context=context,
        )

        assert isinstance(result, QAResult)
        assert result.answer

        session_state = context.get(SESSION_NAMESPACE, SessionState)
        assert session_state is not None
        # If citations were returned, they should use registry indices
        if result.citations:
            assert len(session_state.citation_registry) > 0

    @pytest.mark.asyncio
    async def test_run_qa_core_without_context(
        self, allow_model_requests, qa_client, qa_config
    ):
        """run_qa_core without context uses sequential fallback indices."""
        result = await run_qa_core(
            client=qa_client,
            config=qa_config,
            question="What is Python?",
            context=None,
        )

        assert isinstance(result, QAResult)
        assert result.answer
        # Without context, citation indices are i+1
        for i, c in enumerate(result.citations):
            assert c.index == i + 1

    @pytest.mark.asyncio
    async def test_run_qa_core_on_qa_complete_callback(
        self, allow_model_requests, qa_client, qa_config
    ):
        """run_qa_core invokes on_qa_complete callback when context is provided."""
        context = ToolContext()
        prepare_context(context, features=["qa"])

        callback_calls: list[tuple] = []

        def on_complete(qa_session_state, config):
            callback_calls.append((qa_session_state, config))

        await run_qa_core(
            client=qa_client,
            config=qa_config,
            question="What is Python?",
            context=context,
            on_qa_complete=on_complete,
        )

        assert len(callback_calls) == 1
        assert isinstance(callback_calls[0][0], QASessionState)

    @pytest.mark.asyncio
    async def test_run_qa_core_fifo_limit(
        self, allow_model_requests, qa_client, qa_config
    ):
        """run_qa_core trims qa_history beyond MAX_QA_HISTORY."""
        context = ToolContext()
        prepare_context(context, features=["qa"])

        qa_session_state = context.get(QA_SESSION_NAMESPACE, QASessionState)
        assert qa_session_state is not None
        # Pre-fill with MAX_QA_HISTORY entries
        from haiku.rag.tools.qa import QAHistoryEntry

        qa_session_state.qa_history = [
            QAHistoryEntry(question=f"Q{i}", answer=f"A{i}", confidence=0.9)
            for i in range(MAX_QA_HISTORY)
        ]

        await run_qa_core(
            client=qa_client,
            config=qa_config,
            question="One more question?",
            context=context,
        )

        # After adding one more, FIFO should trim to MAX_QA_HISTORY
        assert len(qa_session_state.qa_history) == MAX_QA_HISTORY
        # The oldest entry (Q0) should have been trimmed
        assert qa_session_state.qa_history[0].question != "Q0"


@pytest.mark.vcr()
class TestRunQACoreWithPriorAnswers:
    """Tests for run_qa_core prior answer matching."""

    @pytest.mark.asyncio
    async def test_run_qa_core_matches_prior_answers(
        self, allow_model_requests, qa_client, qa_config
    ):
        """run_qa_core matches prior answers when embedding similarity is high."""
        from unittest.mock import AsyncMock, patch

        from haiku.rag.tools.qa import QAHistoryEntry

        context = ToolContext()
        prepare_context(context, features=["qa"])

        qa_session_state = context.get(QA_SESSION_NAMESPACE, QASessionState)
        assert qa_session_state is not None

        # Pre-populate with a prior answer that has a known embedding
        prior_embedding = [0.5] * 2560
        qa_session_state.qa_history = [
            QAHistoryEntry(
                question="What is Python?",
                answer="A programming language.",
                confidence=0.9,
                question_embedding=prior_embedding,
            )
        ]

        # Mock the embedder to return a near-identical embedding for the new question
        mock_embedder = AsyncMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.5] * 2560)

        with patch("haiku.rag.tools.qa.get_embedder", return_value=mock_embedder):
            result = await run_qa_core(
                client=qa_client,
                config=qa_config,
                question="Tell me about Python",
                context=context,
            )

        assert isinstance(result, QAResult)
        assert result.answer


@pytest.mark.vcr()
class TestAskTool:
    """Tests for the ask tool in create_qa_toolset."""

    @pytest.mark.asyncio
    async def test_ask_without_tool_context(
        self, allow_model_requests, qa_client, qa_config
    ):
        """ask tool without tool context returns raw QAResult."""
        toolset = create_qa_toolset(qa_config)
        ask_tool = toolset.tools["ask"]

        ctx = make_ctx(qa_client, None)
        result = await ask_tool.function(ctx, "What is Python?")

        assert isinstance(result, QAResult)
        assert result.answer

    @pytest.mark.asyncio
    async def test_ask_with_tool_context_returns_tool_return(
        self, allow_model_requests, qa_client, qa_config
    ):
        """ask tool with tool context returns ToolReturn with state snapshot."""
        context = ToolContext()
        prepare_context(context, features=["qa"])

        toolset = create_qa_toolset(qa_config)
        ask_tool = toolset.tools["ask"]

        ctx = make_ctx(qa_client, context)
        result = await ask_tool.function(ctx, "What is Python?")

        assert isinstance(result, ToolReturn)
        assert result.metadata is not None
        assert len(result.metadata) > 0


@pytest.fixture
async def qa_client(temp_db_path):
    """Create a HaikuRAG client with test documents for QA tests."""
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(temp_db_path, create=True) as rag:
        await rag.create_document(
            "Python is a programming language. It is widely used for web development.",
            uri="test://python",
            title="Python Guide",
        )
        yield rag


@pytest.fixture
def qa_config():
    """Default AppConfig for QA tests."""
    from haiku.rag.config import Config

    return Config

import pytest

from haiku.rag.tools import QAResult, ToolContext
from haiku.rag.tools.qa import QA_NAMESPACE, QAState, create_qa_toolset


class TestQAState:
    """Tests for QAState model."""

    def test_qa_state_defaults(self):
        """QAState initializes with empty history."""
        state = QAState()
        assert state.history == []

    def test_qa_state_add_result(self):
        """Can add QAResult to history."""
        state = QAState()
        result = QAResult(question="What is Python?", answer="A programming language.")
        state.history.append(result)
        assert len(state.history) == 1
        assert state.history[0].question == "What is Python?"

    def test_qa_state_serialization(self):
        """QAState serializes and deserializes correctly."""
        state = QAState()
        state.history.append(
            QAResult(
                question="Test?",
                answer="Answer.",
                confidence=0.95,
            )
        )

        data = state.model_dump()
        restored = QAState.model_validate(data)
        assert len(restored.history) == 1
        assert restored.history[0].confidence == 0.95


class TestQAToolset:
    """Tests for create_qa_toolset."""

    def test_create_qa_toolset_returns_function_toolset(
        self, qa_client_simple, qa_config
    ):
        """create_qa_toolset returns a FunctionToolset."""
        from pydantic_ai import FunctionToolset

        toolset = create_qa_toolset(qa_client_simple, qa_config)
        assert isinstance(toolset, FunctionToolset)

    def test_qa_toolset_has_ask_tool(self, qa_client_simple, qa_config):
        """The toolset includes an 'ask' tool."""
        toolset = create_qa_toolset(qa_client_simple, qa_config)
        assert "ask" in toolset.tools

    def test_qa_toolset_registers_state(self, qa_client_simple, qa_config):
        """Toolset registers QAState under QA_NAMESPACE."""
        context = ToolContext()
        create_qa_toolset(qa_client_simple, qa_config, context=context)

        state = context.get(QA_NAMESPACE)
        assert state is not None
        assert isinstance(state, QAState)

    def test_qa_toolset_custom_tool_name(self, qa_client_simple, qa_config):
        """Toolset supports custom tool name."""
        toolset = create_qa_toolset(
            qa_client_simple, qa_config, tool_name="answer_question"
        )
        assert "answer_question" in toolset.tools
        assert "ask" not in toolset.tools


@pytest.fixture
def qa_client_simple(temp_db_path):
    """Create a HaikuRAG client without documents for basic tests."""
    import asyncio

    from haiku.rag.client import HaikuRAG

    async def setup():
        rag = HaikuRAG(temp_db_path, create=True)
        await rag.__aenter__()
        return rag

    return asyncio.get_event_loop().run_until_complete(setup())


@pytest.fixture
def qa_config():
    """Default AppConfig for QA tests."""
    from haiku.rag.config import Config

    return Config

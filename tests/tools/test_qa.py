import pytest

from haiku.rag.tools.qa import create_qa_toolset


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

    def test_qa_toolset_custom_tool_name(self, qa_client_simple, qa_config):
        """Toolset supports custom tool name."""
        toolset = create_qa_toolset(
            qa_client_simple, qa_config, tool_name="answer_question"
        )
        assert "answer_question" in toolset.tools
        assert "ask" not in toolset.tools


@pytest.fixture
async def qa_client_simple(temp_db_path):
    """Create a HaikuRAG client without documents for basic tests."""
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(temp_db_path, create=True) as rag:
        yield rag


@pytest.fixture
def qa_config():
    """Default AppConfig for QA tests."""
    from haiku.rag.config import Config

    return Config

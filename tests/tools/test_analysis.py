import pytest

from haiku.rag.tools import ToolContext
from haiku.rag.tools.analysis import (
    ANALYSIS_NAMESPACE,
    AnalysisState,
    create_analysis_toolset,
)


class TestAnalysisState:
    """Tests for AnalysisState model."""

    def test_analysis_state_defaults(self):
        """AnalysisState initializes with empty programs."""
        state = AnalysisState()
        assert state.programs == []

    def test_analysis_state_serialization(self):
        """AnalysisState serializes and deserializes correctly."""
        state = AnalysisState()
        state.programs.append("print('hello')")

        data = state.model_dump()
        restored = AnalysisState.model_validate(data)
        assert len(restored.programs) == 1
        assert restored.programs[0] == "print('hello')"


class TestAnalysisToolset:
    """Tests for create_analysis_toolset."""

    def test_create_analysis_toolset_returns_function_toolset(
        self, analysis_client, analysis_config
    ):
        """create_analysis_toolset returns a FunctionToolset."""
        from pydantic_ai import FunctionToolset

        toolset = create_analysis_toolset(analysis_client, analysis_config)
        assert isinstance(toolset, FunctionToolset)

    def test_analysis_toolset_has_analyze_tool(self, analysis_client, analysis_config):
        """The toolset includes an 'analyze' tool."""
        toolset = create_analysis_toolset(analysis_client, analysis_config)
        assert "analyze" in toolset.tools

    def test_analysis_toolset_registers_state(self, analysis_client, analysis_config):
        """Toolset registers AnalysisState under ANALYSIS_NAMESPACE."""
        context = ToolContext()
        create_analysis_toolset(analysis_client, analysis_config, context=context)

        state = context.get(ANALYSIS_NAMESPACE)
        assert state is not None
        assert isinstance(state, AnalysisState)

    def test_analysis_toolset_custom_tool_name(self, analysis_client, analysis_config):
        """Toolset supports custom tool name."""
        toolset = create_analysis_toolset(
            analysis_client, analysis_config, tool_name="run_code"
        )
        assert "run_code" in toolset.tools
        assert "analyze" not in toolset.tools


@pytest.fixture
async def analysis_client(temp_db_path):
    """Create a HaikuRAG client for analysis tests."""
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(temp_db_path, create=True) as rag:
        yield rag


@pytest.fixture
def analysis_config():
    """Default AppConfig for analysis tests."""
    from haiku.rag.config import Config

    return Config

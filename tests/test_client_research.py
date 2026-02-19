from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from haiku.rag.agents.research.models import ConversationalAnswer, ResearchReport
from haiku.rag.client import HaikuRAG


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent / "cassettes" / "test_client_research")


async def test_client_research_report(temp_db_path):
    """Test client.research() delegates to research graph in report mode."""
    mock_report = ResearchReport(
        title="Test Report",
        executive_summary="Summary",
        main_findings=["Finding 1"],
        conclusions=["Conclusion 1"],
        sources_summary="Sources",
    )

    with patch("haiku.rag.agents.research.graph.build_research_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.run = AsyncMock(return_value=mock_report)
        mock_build.return_value = mock_graph

        async with HaikuRAG(temp_db_path, create=True) as client:
            result = await client.research(question="What is X?")

        assert result is mock_report
        mock_build.assert_called_once()
        # Verify output_mode passed correctly
        _, kwargs = mock_build.call_args
        assert kwargs["output_mode"] == "report"

        # Verify graph.run was called with correct state/deps
        mock_graph.run.assert_called_once()
        call_kwargs = mock_graph.run.call_args[1]
        assert call_kwargs["state"].context.original_question == "What is X?"
        assert isinstance(call_kwargs["deps"].client, HaikuRAG)


async def test_client_research_conversational(temp_db_path):
    """Test client.research() with conversational output mode."""
    mock_answer = ConversationalAnswer(
        answer="The answer is 42.",
        confidence=0.95,
    )

    with patch("haiku.rag.agents.research.graph.build_research_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.run = AsyncMock(return_value=mock_answer)
        mock_build.return_value = mock_graph

        async with HaikuRAG(temp_db_path, create=True) as client:
            result = await client.research(
                question="What is X?",
                output_mode="conversational",
            )

        assert result is mock_answer
        _, kwargs = mock_build.call_args
        assert kwargs["output_mode"] == "conversational"


async def test_client_research_passes_filter(temp_db_path):
    """Test client.research() passes filter to state."""
    mock_report = ResearchReport(
        title="Test",
        executive_summary="Summary",
        main_findings=[],
        conclusions=[],
        sources_summary="",
    )

    with patch("haiku.rag.agents.research.graph.build_research_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.run = AsyncMock(return_value=mock_report)
        mock_build.return_value = mock_graph

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.research(
                question="What is X?",
                filter="uri LIKE '%test%'",
            )

        call_kwargs = mock_graph.run.call_args[1]
        assert call_kwargs["state"].search_filter == "uri LIKE '%test%'"


async def test_client_research_uses_config(temp_db_path):
    """Test client.research() passes config to graph builder and state."""
    mock_report = ResearchReport(
        title="Test",
        executive_summary="Summary",
        main_findings=[],
        conclusions=[],
        sources_summary="",
    )

    with patch("haiku.rag.agents.research.graph.build_research_graph") as mock_build:
        mock_graph = AsyncMock()
        mock_graph.run = AsyncMock(return_value=mock_report)
        mock_build.return_value = mock_graph

        async with HaikuRAG(temp_db_path, create=True) as client:
            await client.research(question="What is X?")

        _, kwargs = mock_build.call_args
        assert kwargs["config"] is client._config

        call_kwargs = mock_graph.run.call_args[1]
        state = call_kwargs["state"]
        assert state.max_iterations == client._config.research.max_iterations
        assert state.max_concurrency == client._config.research.max_concurrency

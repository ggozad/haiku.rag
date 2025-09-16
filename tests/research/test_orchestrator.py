"""Tests for the research orchestrator."""

from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.research.dependencies import ResearchContext, ResearchDependencies
from haiku.rag.research.evaluation_agent import EvaluationResult
from haiku.rag.research.orchestrator import ResearchOrchestrator, ResearchPlan
from haiku.rag.store.models.chunk import Chunk


@pytest.fixture
def mock_client():
    """Create a mock HaikuRAG client."""
    client = create_autospec(HaikuRAG, instance=True)
    client.search = AsyncMock()
    client.expand_context = AsyncMock()
    return client


@pytest.fixture
def research_context():
    """Create a research context."""
    return ResearchContext(original_question="What is climate change?")


@pytest.fixture
def research_deps(mock_client, research_context):
    """Create research dependencies."""
    return ResearchDependencies(client=mock_client, context=research_context)


def create_mock_chunk(chunk_id: str, content: str, score: float = 0.8):
    """Helper to create mock chunk objects."""
    return Chunk(
        id=chunk_id,
        document_id=f"doc_{chunk_id}",
        content=content,
        document_uri=f"doc_{chunk_id}.md",
        metadata={},
    ), score


class TestResearchOrchestrator:
    """Test suite for ResearchOrchestrator."""

    def test_orchestrator_uses_config_defaults(self):
        """Test that orchestrator uses config defaults when no args provided."""
        orchestrator = ResearchOrchestrator()

        # Should use RESEARCH_PROVIDER/MODEL if set, else QA_PROVIDER/MODEL
        assert orchestrator.provider is not None
        assert orchestrator.model is not None

        # All agents should use the same provider/model
        assert orchestrator.search_agent.provider == orchestrator.provider
        assert orchestrator.search_agent.model == orchestrator.model
        assert orchestrator.evaluation_agent.provider == orchestrator.provider
        assert orchestrator.evaluation_agent.model == orchestrator.model
        assert orchestrator.synthesis_agent.provider == orchestrator.provider
        assert orchestrator.synthesis_agent.model == orchestrator.model

    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes all agents correctly."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Check all agents are initialized
        assert orchestrator.search_agent is not None
        assert orchestrator.evaluation_agent is not None
        assert orchestrator.synthesis_agent is not None

        # Check they all use the same provider and model
        assert orchestrator.search_agent.provider == "openai"
        assert orchestrator.search_agent.model == "gpt-4"
        assert orchestrator.evaluation_agent.provider == "openai"
        assert orchestrator.evaluation_agent.model == "gpt-4"
        assert orchestrator.synthesis_agent.provider == "openai"
        assert orchestrator.synthesis_agent.model == "gpt-4"

    def test_orchestrator_has_correct_output_type(self):
        """Test that orchestrator's output type is ResearchPlan."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")
        assert orchestrator.output_type == ResearchPlan

    def test_orchestrator_has_no_tools(self):
        """Test that orchestrator no longer registers tools (direct agent calls now)."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Get the tools from the agent
        tools = orchestrator.agent._function_toolset.tools
        tool_names = list(tools.keys())

        # Should have no tools since we call agents directly now
        assert len(tool_names) == 0

    def test_should_stop_research_logic(self):
        """Test the stopping logic based on EvaluationResult."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Create mock evaluation results
        sufficient_result = MagicMock()
        sufficient_result.output = EvaluationResult(
            key_insights=["Climate is changing", "Human activity is the cause"],
            new_questions=[],
            confidence_score=0.9,
            is_sufficient=True,
            reasoning="All aspects covered comprehensively",
        )

        insufficient_result = MagicMock()
        insufficient_result.output = EvaluationResult(
            key_insights=["Some data found"],
            new_questions=[
                "What about economic impacts?",
                "Regional variations?",
            ],
            confidence_score=0.4,
            is_sufficient=False,
            reasoning="Major gaps remain in understanding",
        )

        # Test with sufficient research (threshold 0.8)
        assert orchestrator._should_stop_research(sufficient_result, 0.8)

        # Test with insufficient research
        assert not orchestrator._should_stop_research(insufficient_result, 0.8)

        # Test with high confidence but below threshold
        sufficient_result.output.confidence_score = 0.75
        assert not orchestrator._should_stop_research(sufficient_result, 0.8)

        # Test with is_sufficient=False even with high confidence
        insufficient_result.output.confidence_score = 0.95
        assert not orchestrator._should_stop_research(insufficient_result, 0.8)

    @pytest.mark.asyncio
    async def test_conduct_research_workflow(self, mock_client):
        """Test the basic research workflow."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Mock the agent runs
        # Mock initial plan
        plan_mock = MagicMock()
        plan_mock.output = ResearchPlan(
            main_question="What is climate change?",
            sub_questions=[
                "What causes climate change?",
                "What are the effects?",
                "What can be done?",
            ],
        )
        orchestrator.run = AsyncMock(return_value=plan_mock)

        # Mock search agent
        search_mock = MagicMock()
        search_mock.output = "Climate change is caused by greenhouse gases."
        orchestrator.search_agent.run = AsyncMock(return_value=search_mock)

        # Mock evaluation agent - make it stop after first iteration
        eval_mock = MagicMock()
        eval_mock.output = EvaluationResult(
            key_insights=["Climate change is real"],
            new_questions=[],
            confidence_score=0.9,
            is_sufficient=True,
            reasoning="Sufficient information gathered",
        )
        orchestrator.evaluation_agent.run = AsyncMock(return_value=eval_mock)

        # Mock synthesis agent
        from haiku.rag.research.synthesis_agent import ResearchReport

        synthesis_mock = MagicMock()
        synthesis_mock.output = ResearchReport(
            title="Climate Change Report",
            executive_summary="Summary",
            main_findings=["Finding 1"],
            themes={},
            conclusions=[],
            limitations=[],
            recommendations=[],
            sources_summary="Sources",
        )
        orchestrator.synthesis_agent.run = AsyncMock(return_value=synthesis_mock)

        # Mock client search and expand
        mock_client.search.return_value = []
        mock_client.expand_context.return_value = []

        # Run the research
        report = await orchestrator.conduct_research(
            "What is climate change?", mock_client, max_iterations=3
        )

        # Verify we got a report
        assert report.title == "Climate Change Report"

        # Verify search was called for all 3 sub-questions
        assert orchestrator.search_agent.run.call_count == 3

from unittest.mock import AsyncMock, create_autospec

import pytest
from pydantic_ai.models.test import TestModel

from haiku.rag.client import HaikuRAG
from haiku.rag.research.dependencies import ResearchContext, ResearchDependencies
from haiku.rag.research.evaluation_agent import EvaluationResult
from haiku.rag.research.orchestrator import ResearchOrchestrator, ResearchPlan
from haiku.rag.research.synthesis_agent import ResearchReport
from haiku.rag.store.models.chunk import Chunk


@pytest.fixture
def test_model():
    """Create a test model for orchestrator testing."""
    return TestModel()


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
        from unittest.mock import MagicMock

        # Sufficient research result
        sufficient_result = MagicMock()
        sufficient_result.output = EvaluationResult(
            key_insights=["Climate is changing", "Human activity is the cause"],
            new_questions=[],
            confidence_score=0.9,
            is_sufficient=True,
            reasoning="All aspects covered comprehensively",
        )

        # Insufficient research result
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
    async def test_conduct_research_workflow(self, test_model, mock_client):
        """Test the basic research workflow using TestModel."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Setup mock client returns
        mock_chunks = [
            create_mock_chunk("1", "Climate change information"),
        ]
        mock_client.search.return_value = mock_chunks
        mock_client.expand_context.return_value = mock_chunks

        # Use TestModel for all agents
        with orchestrator.agent.override(model=test_model):
            with orchestrator.search_agent.agent.override(model=test_model):
                with orchestrator.evaluation_agent.agent.override(model=test_model):
                    with orchestrator.synthesis_agent.agent.override(model=test_model):
                        # Run the research
                        report = await orchestrator.conduct_research(
                            "What is climate change?", mock_client, max_iterations=1
                        )

                        # Verify we got a valid report structure
                        assert isinstance(report, ResearchReport)
                        assert report.title
                        assert report.executive_summary
                        assert isinstance(report.main_findings, list)
                        assert isinstance(report.themes, dict)
                        assert isinstance(report.conclusions, list)
                        assert isinstance(report.limitations, list)
                        assert isinstance(report.recommendations, list)
                        assert report.sources_summary

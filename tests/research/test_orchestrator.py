"""Tests for the research orchestrator."""

from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from haiku.rag.client import HaikuRAG
from haiku.rag.research.analysis_agent import AnalysisResult
from haiku.rag.research.base import SearchResult
from haiku.rag.research.clarification_agent import ClarificationResult
from haiku.rag.research.dependencies import ResearchContext, ResearchDependencies
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
        assert orchestrator.analysis_agent.provider == orchestrator.provider
        assert orchestrator.analysis_agent.model == orchestrator.model

    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes all agents correctly."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Check all agents are initialized
        assert orchestrator.search_agent is not None
        assert orchestrator.analysis_agent is not None
        assert orchestrator.clarification_agent is not None
        assert orchestrator.synthesis_agent is not None

        # Check they all use the same provider and model
        assert orchestrator.search_agent.provider == "openai"
        assert orchestrator.search_agent.model == "gpt-4"
        assert orchestrator.analysis_agent.provider == "openai"
        assert orchestrator.clarification_agent.provider == "openai"
        assert orchestrator.synthesis_agent.provider == "openai"

    def test_orchestrator_has_correct_output_type(self):
        """Test that orchestrator's output type is ResearchPlan."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")
        assert orchestrator.output_type == ResearchPlan

    def test_orchestrator_registers_delegation_tools(self):
        """Test that orchestrator registers all delegation tools."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Get the tools from the agent
        tools = orchestrator.agent._function_toolset.tools
        tool_names = list(tools.keys())

        # Check all delegation tools are registered
        assert "delegate_search" in tool_names
        assert "delegate_analysis" in tool_names
        assert "delegate_clarification" in tool_names
        assert "generate_report" in tool_names

    def test_should_stop_research_logic(self):
        """Test the stopping logic based on ClarificationResult."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Create mock clarification results
        sufficient_result = MagicMock()
        sufficient_result.output = ClarificationResult(
            information_gaps=[],
            follow_up_questions=[],
            suggested_searches=[],
            completeness_assessment="Research is comprehensive",
            priority_areas=[],
            is_sufficient=True,
            confidence_score=0.9,
            reasoning="All aspects covered",
        )

        insufficient_result = MagicMock()
        insufficient_result.output = ClarificationResult(
            information_gaps=["Missing data on impacts"],
            follow_up_questions=["What about economic impacts?"],
            suggested_searches=["economic impact climate change"],
            completeness_assessment="More research needed",
            priority_areas=["Economic analysis"],
            is_sufficient=False,
            confidence_score=0.4,
            reasoning="Major gaps remain",
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
    async def test_delegate_search_tool(self, research_deps):
        """Test the delegate_search tool function."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Get the delegate_search tool
        tools = orchestrator.agent._function_toolset.tools
        delegate_search = tools["delegate_search"].function

        # Mock the search agent's run method
        orchestrator.search_agent.run = AsyncMock(
            return_value=MagicMock(output=["results"])
        )

        # Create context and call the tool
        ctx = RunContext(deps=research_deps, model=TestModel(), usage=RunUsage())
        await delegate_search(ctx, queries=["climate change"])

        # Verify the search agent was called
        orchestrator.search_agent.run.assert_called_once()
        assert "climate change" in orchestrator.search_agent.run.call_args[0][0]

    @pytest.mark.asyncio
    async def test_delegate_analysis_tool(self, research_deps):
        """Test the delegate_analysis tool function."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Add some search results to context
        research_deps.context.search_results = [
            {
                "query": "test",
                "results": [
                    SearchResult(
                        content="Climate data",
                        score=0.9,
                        document_uri="doc1.md",
                        metadata={},
                    )
                ],
            }
        ]

        # Get the delegate_analysis tool
        tools = orchestrator.agent._function_toolset.tools
        delegate_analysis = tools["delegate_analysis"].function

        # Mock the analysis agent's run method
        mock_result = MagicMock()
        mock_result.output = AnalysisResult(
            key_insights=["Climate is changing"],
            themes={"warming": ["temperature rise"]},
            summary="Analysis complete",
            evidence_quality="strong",
            recommendations=["More research needed"],
        )
        orchestrator.analysis_agent.run = AsyncMock(return_value=mock_result)

        # Create context and call the tool
        ctx = RunContext(deps=research_deps, model=TestModel(), usage=RunUsage())
        await delegate_analysis(ctx)

        # Verify the analysis agent was called
        orchestrator.analysis_agent.run.assert_called_once()

        # Verify insights were added to context
        assert "Climate is changing" in research_deps.context.insights

    @pytest.mark.asyncio
    async def test_delegate_clarification_tool(self, research_deps):
        """Test the delegate_clarification tool function."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Get the delegate_clarification tool
        tools = orchestrator.agent._function_toolset.tools
        delegate_clarification = tools["delegate_clarification"].function

        # Mock the clarification agent's run method
        mock_result = MagicMock()
        mock_result.output = ClarificationResult(
            information_gaps=["Missing economic data"],
            follow_up_questions=["What about costs?"],
            suggested_searches=["climate change costs"],
            completeness_assessment="Needs more data",
            priority_areas=["Economics"],
            is_sufficient=False,
            confidence_score=0.6,
            reasoning="Missing key information",
        )
        orchestrator.clarification_agent.run = AsyncMock(return_value=mock_result)

        # Create context and call the tool
        ctx = RunContext(deps=research_deps, model=TestModel(), usage=RunUsage())
        await delegate_clarification(ctx)

        # Verify the clarification agent was called
        orchestrator.clarification_agent.run.assert_called_once()

        # Verify gaps and questions were added to context
        assert "Missing economic data" in research_deps.context.gaps
        assert "What about costs?" in research_deps.context.follow_up_questions

    @pytest.mark.asyncio
    async def test_generate_report_tool(self, research_deps):
        """Test the generate_report tool function."""
        orchestrator = ResearchOrchestrator(provider="openai", model="gpt-4")

        # Get the generate_report tool
        tools = orchestrator.agent._function_toolset.tools
        generate_report = tools["generate_report"].function

        # Mock the synthesis agent's run method
        from haiku.rag.research.synthesis_agent import ResearchReport

        mock_result = MagicMock()
        mock_result.output = ResearchReport(
            title="Climate Change Research",
            executive_summary="Summary of findings",
            main_findings=["Finding 1", "Finding 2"],
            themes={"warming": "Global temperature rise"},
            conclusions=["Conclusion 1"],
            limitations=["Limited data"],
            recommendations=["More research"],
            sources_summary="Various sources",
        )
        orchestrator.synthesis_agent.run = AsyncMock(return_value=mock_result)

        # Create context and call the tool
        ctx = RunContext(deps=research_deps, model=TestModel(), usage=RunUsage())
        result = await generate_report(ctx)

        # Verify the synthesis agent was called
        orchestrator.synthesis_agent.run.assert_called_once()

        # Verify we got a ResearchReport
        assert isinstance(result, ResearchReport)
        assert result.title == "Climate Change Research"

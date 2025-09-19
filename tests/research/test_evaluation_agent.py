from haiku.rag.config import Config
from haiku.rag.research.evaluation_agent import (
    AnalysisEvaluationAgent,
    EvaluationResult,
)


class TestAnalysisEvaluationAgent:
    """Lean tests for AnalysisEvaluationAgent without LLM mocking."""

    def test_agent_initialization(self):
        agent = AnalysisEvaluationAgent(
            provider=Config.RESEARCH_PROVIDER, model=Config.RESEARCH_MODEL
        )
        assert agent.provider == Config.RESEARCH_PROVIDER
        assert agent.model == Config.RESEARCH_MODEL
        assert agent.output_type == EvaluationResult

from haiku.rag.research.evaluation_agent import (
    AnalysisEvaluationAgent,
    EvaluationResult,
)


class TestAnalysisEvaluationAgent:
    """Lean tests for AnalysisEvaluationAgent without LLM mocking."""

    def test_agent_initialization(self):
        agent = AnalysisEvaluationAgent(provider="openai", model="gpt-4")
        assert agent.provider == "openai"
        assert agent.model == "gpt-4"
        assert agent.output_type == EvaluationResult

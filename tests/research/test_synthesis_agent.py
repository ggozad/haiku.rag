from haiku.rag.config import Config
from haiku.rag.research.synthesis_agent import ResearchReport, SynthesisAgent


class TestSynthesisAgent:
    """Lean tests for SynthesisAgent without LLM mocking."""

    def test_agent_initialization(self):
        agent = SynthesisAgent(
            provider=Config.RESEARCH_PROVIDER, model=Config.RESEARCH_MODEL
        )
        assert agent.provider == Config.RESEARCH_PROVIDER
        assert agent.model == Config.RESEARCH_MODEL
        assert agent.output_type == ResearchReport

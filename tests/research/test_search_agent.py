from haiku.rag.config import Config
from haiku.rag.research import SearchAnswer, SearchSpecialistAgent


class TestSearchSpecialistAgent:
    """Lean tests for SearchSpecialistAgent without LLM mocking."""

    def test_agent_initialization(self):
        agent = SearchSpecialistAgent(
            provider=Config.RESEARCH_PROVIDER, model=Config.RESEARCH_MODEL
        )
        assert agent.provider == Config.RESEARCH_PROVIDER
        assert agent.model == Config.RESEARCH_MODEL
        assert agent.output_type is SearchAnswer

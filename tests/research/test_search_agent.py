from haiku.rag.research import SearchAnswer, SearchSpecialistAgent


class TestSearchSpecialistAgent:
    """Lean tests for SearchSpecialistAgent without LLM mocking."""

    def test_agent_initialization(self):
        agent = SearchSpecialistAgent(provider="openai", model="gpt-4")
        assert agent.provider == "openai"
        assert agent.model == "gpt-4"
        assert agent.output_type is SearchAnswer

from haiku.rag.research.synthesis_agent import ResearchReport, SynthesisAgent


class TestSynthesisAgent:
    """Lean tests for SynthesisAgent without LLM mocking."""

    def test_agent_initialization(self):
        agent = SynthesisAgent(provider="openai", model="gpt-4")
        assert agent.provider == "openai"
        assert agent.model == "gpt-4"
        assert agent.output_type == ResearchReport

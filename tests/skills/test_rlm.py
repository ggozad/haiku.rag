from unittest.mock import AsyncMock

from haiku.rag.agents.rlm.models import RLMResult
from haiku.rag.client import HaikuRAG

from .conftest import _get_tool, _make_ctx


class TestRLMSkillCreation:
    def test_create_skill_returns_valid_skill(self, temp_db_path):
        from haiku.rag.skills.rlm import create_skill

        skill = create_skill(db_path=temp_db_path)
        assert skill.metadata.name == "rag-rlm"
        assert skill.metadata.description
        assert skill.instructions

    def test_create_skill_has_expected_tools(self, temp_db_path):
        from haiku.rag.skills.rlm import create_skill

        skill = create_skill(db_path=temp_db_path)
        tool_names = {getattr(t, "__name__") for t in skill.tools if callable(t)}
        assert tool_names == {"analyze"}

    def test_create_skill_has_state(self, temp_db_path):
        from haiku.rag.skills.rlm import RLMState, create_skill

        skill = create_skill(db_path=temp_db_path)
        assert skill._state_type is RLMState
        assert skill._state_namespace == "rlm"

    def test_create_skill_from_env(self, monkeypatch, temp_db_path):
        monkeypatch.setenv("HAIKU_RAG_DB", str(temp_db_path))
        from haiku.rag.skills.rlm import create_skill

        skill = create_skill()
        assert skill.metadata.name == "rag-rlm"


class TestAnalyzeTool:
    async def test_analyze_returns_result(self, rag_db, monkeypatch):
        from haiku.rag.skills.rlm import create_skill

        monkeypatch.setattr(
            HaikuRAG,
            "rlm",
            AsyncMock(return_value=RLMResult(answer="42", program="print(42)")),
        )

        skill = create_skill(db_path=rag_db)
        analyze = _get_tool(skill, "analyze")
        ctx = _make_ctx()
        result = await analyze(ctx, question="How many documents?")
        assert isinstance(result, str)
        assert "42" in result
        assert "print(42)" in result

    async def test_analyze_updates_state(self, rag_db, monkeypatch):
        from haiku.rag.skills.rlm import RLMState, create_skill

        monkeypatch.setattr(
            HaikuRAG,
            "rlm",
            AsyncMock(return_value=RLMResult(answer="42", program="print(42)")),
        )

        skill = create_skill(db_path=rag_db)
        analyze = _get_tool(skill, "analyze")
        state = RLMState()
        ctx = _make_ctx(state)
        await analyze(ctx, question="How many documents?")
        assert len(state.analyses) == 1
        assert state.analyses[0].question == "How many documents?"
        assert state.analyses[0].answer == "42"
        assert state.analyses[0].program == "print(42)"

    async def test_analyze_with_document_and_filter(self, rag_db, monkeypatch):
        from haiku.rag.skills.rlm import create_skill

        captured_kwargs = {}

        async def mock_rlm(self, question, **kwargs):
            captured_kwargs.update(kwargs)
            return RLMResult(answer="Result", program="code()")

        monkeypatch.setattr(HaikuRAG, "rlm", mock_rlm)

        skill = create_skill(db_path=rag_db)
        analyze = _get_tool(skill, "analyze")
        ctx = _make_ctx()
        await analyze(
            ctx,
            question="Count pages",
            document="AI Overview",
            filter="title = 'AI Overview'",
        )
        assert captured_kwargs.get("documents") == ["AI Overview"]
        assert captured_kwargs.get("filter") == "title = 'AI Overview'"

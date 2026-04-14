from unittest.mock import AsyncMock

from haiku.rag.agents.analysis.models import AnalysisResult
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig
from haiku.rag.skills.analysis import (
    STATE_NAMESPACE,
    STATE_TYPE,
    AnalysisState,
    instructions,
    skill_metadata,
    state_metadata,
)
from haiku.skills.models import SkillMetadata, StateMetadata

from .conftest import _get_tool, _make_ctx


class TestAnalysisModuleAPI:
    def test_state_type_is_analysis_state(self):
        assert STATE_TYPE is AnalysisState

    def test_state_namespace(self):
        assert STATE_NAMESPACE == "analysis"

    def test_state_metadata_returns_state_metadata(self):
        result = state_metadata()
        assert isinstance(result, StateMetadata)
        assert result.namespace == "analysis"
        assert result.type is AnalysisState
        assert result.schema == AnalysisState.model_json_schema()

    def test_skill_metadata_returns_skill_metadata(self):
        result = skill_metadata()
        assert isinstance(result, SkillMetadata)
        assert result.name == "rag-analysis"

    def test_instructions_returns_string(self):
        result = instructions()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_constants_match_create_skill(self, test_app_config, temp_db_path):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.state_type is STATE_TYPE
        assert skill.state_namespace == STATE_NAMESPACE
        assert skill.metadata == skill_metadata()
        assert skill.instructions == instructions()


class TestAnalysisSkillCreation:
    def test_create_skill_returns_valid_skill(self, test_app_config, temp_db_path):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.metadata.name == "rag-analysis"
        assert skill.metadata.description
        assert skill.instructions

    def test_create_skill_has_expected_tools(self, test_app_config, temp_db_path):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        tool_names = {getattr(t, "__name__") for t in skill.tools if callable(t)}
        assert tool_names == {"analyze"}

    def test_create_skill_has_state(self, test_app_config, temp_db_path):
        from haiku.rag.skills.analysis import AnalysisState, create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill._state_type is AnalysisState
        assert skill._state_namespace == "analysis"

    def test_create_skill_has_extras(self, test_app_config, temp_db_path):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.extras["config"] is test_app_config
        assert skill.extras["db_path"] is temp_db_path
        assert "visualize_chunk" in skill.extras
        assert "list_documents" in skill.extras
        assert callable(skill.extras["visualize_chunk"])
        assert callable(skill.extras["list_documents"])

    def test_create_skill_from_env(self, monkeypatch, temp_db_path):
        monkeypatch.setenv("HAIKU_RAG_DB", str(temp_db_path))
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill()
        assert skill.metadata.name == "rag-analysis"


class TestDomainPreambleInAnalysisSkillInstructions:
    def test_create_skill_without_domain_preamble(self, test_app_config, temp_db_path):
        from haiku.rag.skills.analysis import create_skill, instructions

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.instructions == instructions()

    def test_create_skill_with_domain_preamble(self, temp_db_path):
        from haiku.rag.config.models import PromptsConfig
        from haiku.rag.skills.analysis import create_skill, instructions

        config = AppConfig(
            prompts=PromptsConfig(
                domain_preamble="This knowledge base contains Helios solar panel documentation."
            )
        )
        skill = create_skill(config=config, db_path=temp_db_path)
        assert skill.instructions is not None
        assert skill.instructions.startswith(
            "This knowledge base contains Helios solar panel documentation."
        )
        base_instructions = instructions()
        assert base_instructions is not None
        assert base_instructions in skill.instructions


class TestAnalyzeTool:
    async def test_analyze_returns_result(self, rag_db, monkeypatch):
        from haiku.rag.skills.analysis import create_skill

        monkeypatch.setattr(
            HaikuRAG,
            "analyze",
            AsyncMock(return_value=AnalysisResult(answer="42", program="print(42)")),
        )

        skill = create_skill(db_path=rag_db)
        analyze = _get_tool(skill, "analyze")
        ctx = _make_ctx()
        result = await analyze(ctx, question="How many documents?")
        assert isinstance(result, str)
        assert "42" in result
        assert "print(42)" in result

    async def test_analyze_updates_state(self, rag_db, monkeypatch):
        from haiku.rag.skills.analysis import AnalysisState, create_skill

        monkeypatch.setattr(
            HaikuRAG,
            "analyze",
            AsyncMock(return_value=AnalysisResult(answer="42", program="print(42)")),
        )

        skill = create_skill(db_path=rag_db)
        analyze = _get_tool(skill, "analyze")
        state = AnalysisState()
        ctx = _make_ctx(state)
        await analyze(ctx, question="How many documents?")
        assert len(state.analyses) == 1
        assert state.analyses[0].question == "How many documents?"
        assert state.analyses[0].answer == "42"
        assert state.analyses[0].program == "print(42)"

    async def test_analyze_applies_document_filter_from_state(
        self, rag_db, monkeypatch
    ):
        from haiku.rag.skills.analysis import AnalysisState, create_skill

        captured_kwargs = {}

        async def mock_analyze(self, question, **kwargs):
            captured_kwargs.update(kwargs)
            return AnalysisResult(answer="42", program="print(42)")

        monkeypatch.setattr(HaikuRAG, "analyze", mock_analyze)

        skill = create_skill(db_path=rag_db)
        analyze = _get_tool(skill, "analyze")
        state = AnalysisState(document_filter="title = 'AI Overview'")
        ctx = _make_ctx(state)
        await analyze(ctx, question="How many documents?")
        assert captured_kwargs.get("filter") == "title = 'AI Overview'"

    async def test_analyze_combines_state_filter_with_explicit_filter(
        self, rag_db, monkeypatch
    ):
        from haiku.rag.skills.analysis import AnalysisState, create_skill

        captured_kwargs = {}

        async def mock_analyze(self, question, **kwargs):
            captured_kwargs.update(kwargs)
            return AnalysisResult(answer="Result", program="code()")

        monkeypatch.setattr(HaikuRAG, "analyze", mock_analyze)

        skill = create_skill(db_path=rag_db)
        analyze = _get_tool(skill, "analyze")
        state = AnalysisState(document_filter="title = 'AI Overview'")
        ctx = _make_ctx(state)
        await analyze(
            ctx,
            question="Count pages",
            filter="uri LIKE '%test%'",
        )
        result_filter = captured_kwargs["filter"]
        assert isinstance(result_filter, str)
        assert "title = 'AI Overview'" in result_filter
        assert "uri LIKE '%test%'" in result_filter

    async def test_analyze_with_document_and_filter(self, rag_db, monkeypatch):
        from haiku.rag.skills.analysis import create_skill

        captured_kwargs = {}

        async def mock_analyze(self, question, **kwargs):
            captured_kwargs.update(kwargs)
            return AnalysisResult(answer="Result", program="code()")

        monkeypatch.setattr(HaikuRAG, "analyze", mock_analyze)

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

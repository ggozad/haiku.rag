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
        assert tool_names == {"search", "list_documents", "execute_code", "cite"}

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


class TestExecuteCodeTool:
    async def test_execute_code_returns_output(self, rag_db):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState()
        ctx = _make_ctx(state)
        result = await execute_code(ctx, code="print('hello')")
        assert "hello" in result

    async def test_execute_code_updates_state(self, rag_db):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState()
        ctx = _make_ctx(state)
        await execute_code(ctx, code="print('hello')")
        assert len(state.executions) == 1
        assert state.executions[0].code == "print('hello')"
        assert state.executions[0].success is True
        assert "hello" in state.executions[0].stdout

    async def test_execute_code_reports_errors(self, rag_db):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState()
        ctx = _make_ctx(state)
        result = await execute_code(ctx, code="x = 1/0")
        assert "Error" in result
        assert "ZeroDivisionError" in result
        assert state.executions[0].success is False

    async def test_execute_code_applies_document_filter(self, rag_db):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState(document_filter="title = 'AI Overview'")
        ctx = _make_ctx(state)
        result = await execute_code(
            ctx, code="docs = await list_documents()\nprint(len(docs))"
        )
        assert "1" in result

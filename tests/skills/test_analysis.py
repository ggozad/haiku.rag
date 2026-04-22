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
    async def test_execute_code_returns_output(self, rag_db, sandbox_factory):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState()
        ctx = _make_ctx(state, sandbox=sandbox_factory())
        result = await execute_code(ctx, code="print('hello')")
        assert "hello" in result

    async def test_execute_code_updates_state(self, rag_db, sandbox_factory):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState()
        ctx = _make_ctx(state, sandbox=sandbox_factory())
        await execute_code(ctx, code="print('hello')")
        assert len(state.executions) == 1
        assert state.executions[0].code == "print('hello')"
        assert state.executions[0].success is True
        assert "hello" in state.executions[0].stdout

    async def test_execute_code_reports_errors(self, rag_db, sandbox_factory):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState()
        ctx = _make_ctx(state, sandbox=sandbox_factory())
        result = await execute_code(ctx, code="x = 1/0")
        assert "Error" in result
        assert "ZeroDivisionError" in result
        assert state.executions[0].success is False

    async def test_execute_code_applies_document_filter(self, rag_db, sandbox_factory):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState(document_filter="title = 'AI Overview'")
        ctx = _make_ctx(state, sandbox=sandbox_factory(filter=state.document_filter))
        result = await execute_code(
            ctx, code="docs = await list_documents()\nprint(len(docs))"
        )
        assert "1" in result

    async def test_execute_code_accumulates_search_results(
        self, rag_db, sandbox_factory
    ):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState()
        ctx = _make_ctx(state, sandbox=sandbox_factory())
        await execute_code(
            ctx, code="results = await search('intelligence')\nprint(len(results))"
        )
        assert "_sandbox" in state.searches
        assert len(state.searches["_sandbox"]) > 0

    async def test_execute_code_vfs_write_denied(self, rag_db, sandbox_factory):
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState()
        ctx = _make_ctx(state, sandbox=sandbox_factory())
        result = await execute_code(
            ctx,
            code=(
                "from pathlib import Path\n"
                "import json\n"
                "dirs = list(Path('/documents').iterdir())\n"
                "p = dirs[0] / 'content.txt'\n"
                "p.write_text('hacked')"
            ),
        )
        assert "Error" in result
        assert "read-only" in result

    async def test_execute_code_variables_persist_within_invocation(
        self, rag_db, sandbox_factory
    ):
        """Same sandbox across two calls → vars persist (one skill invocation)."""
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")
        state = AnalysisState()
        ctx = _make_ctx(state, sandbox=sandbox_factory())

        await execute_code(ctx, code="x = 42")
        result = await execute_code(ctx, code="print(x * 2)")
        assert "84" in result

    async def test_execute_code_isolated_across_invocations(
        self, rag_db, sandbox_factory
    ):
        """Different Sandbox instances → no cross-invocation leak."""
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(db_path=rag_db)
        execute_code = _get_tool(skill, "execute_code")

        ctx1 = _make_ctx(AnalysisState(), sandbox=sandbox_factory())
        await execute_code(ctx1, code="secret = 'do not leak'")

        ctx2 = _make_ctx(AnalysisState(), sandbox=sandbox_factory())
        result = await execute_code(ctx2, code="print(secret)")
        assert not result.startswith("do not leak")
        assert "Error" in result or "NameError" in result


class TestAnalysisLifespan:
    async def test_opens_client_and_sandbox_per_invocation(self, rag_db):
        from haiku.rag.agents.analysis.sandbox import Sandbox
        from haiku.rag.skills._deps import AnalysisRunDeps, make_analysis_lifespan

        config = AppConfig()
        lifespan = make_analysis_lifespan(rag_db, config)
        deps = AnalysisRunDeps()
        async with lifespan(deps):
            assert deps.rag is not None
            assert deps.rag.is_read_only
            assert deps.search_count == 0
            assert isinstance(deps.sandbox, Sandbox)
            docs = await deps.rag.list_documents()
            assert len(docs) == 2

    async def test_lifespan_reads_document_filter_from_state(self, rag_db):
        from haiku.rag.skills._deps import AnalysisRunDeps, make_analysis_lifespan
        from haiku.rag.skills.analysis import AnalysisState

        config = AppConfig()
        lifespan = make_analysis_lifespan(rag_db, config)
        state = AnalysisState(document_filter="title = 'AI Overview'")
        deps = AnalysisRunDeps(state=state)
        async with lifespan(deps):
            assert deps.sandbox is not None
            assert deps.sandbox._context.filter == "title = 'AI Overview'"

    async def test_skill_has_lifespan_and_deps_type(
        self, test_app_config, temp_db_path
    ):
        from haiku.rag.skills._deps import AnalysisRunDeps
        from haiku.rag.skills.analysis import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.deps_type is AnalysisRunDeps
        assert skill.lifespan is not None

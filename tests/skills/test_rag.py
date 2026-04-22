from haiku.rag.config.models import AppConfig
from haiku.rag.skills.rag import (
    STATE_NAMESPACE,
    STATE_TYPE,
    RAGState,
    instructions,
    skill_metadata,
    state_metadata,
)
from haiku.rag.store.models.chunk import SearchResult
from haiku.skills.models import SkillMetadata, StateMetadata

from .conftest import _get_tool, _make_ctx


class TestRAGModuleAPI:
    def test_state_type_is_rag_state(self):
        assert STATE_TYPE is RAGState

    def test_state_namespace(self):
        assert STATE_NAMESPACE == "rag"

    def test_state_metadata_returns_state_metadata(self):
        result = state_metadata()
        assert isinstance(result, StateMetadata)
        assert result.namespace == "rag"
        assert result.type is RAGState
        assert result.schema == RAGState.model_json_schema()

    def test_skill_metadata_returns_skill_metadata(self):
        result = skill_metadata()
        assert isinstance(result, SkillMetadata)
        assert result.name == "rag"

    def test_instructions_returns_string(self):
        result = instructions()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_constants_match_create_skill(self, test_app_config, temp_db_path):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.state_type is STATE_TYPE
        assert skill.state_namespace == STATE_NAMESPACE
        assert skill.metadata == skill_metadata()
        assert skill.instructions == instructions()


class TestGetAgentPreamble:
    def test_without_domain_preamble(self):
        from haiku.rag.skills.rag import AGENT_PREAMBLE, get_agent_preamble

        config = AppConfig()
        assert get_agent_preamble(config) == AGENT_PREAMBLE

    def test_with_domain_preamble(self):
        from haiku.rag.config.models import PromptsConfig
        from haiku.rag.skills.rag import AGENT_PREAMBLE, get_agent_preamble

        config = AppConfig(
            prompts=PromptsConfig(
                domain_preamble="This knowledge base contains Helios solar panel documentation."
            )
        )
        result = get_agent_preamble(config)
        assert result.startswith(
            "This knowledge base contains Helios solar panel documentation."
        )
        assert AGENT_PREAMBLE in result


class TestDomainPreambleInSkillInstructions:
    def test_create_skill_without_domain_preamble(self, test_app_config, temp_db_path):
        from haiku.rag.skills.rag import create_skill, instructions

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.instructions == instructions()

    def test_create_skill_with_domain_preamble(self, temp_db_path):
        from haiku.rag.config.models import PromptsConfig
        from haiku.rag.skills.rag import create_skill, instructions

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


class TestRAGSkillCreation:
    def test_create_skill_returns_valid_skill(self, test_app_config, temp_db_path):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.metadata.name == "rag"
        assert skill.metadata.description
        assert skill.instructions

    def test_create_skill_has_expected_tools(self, test_app_config, temp_db_path):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        tool_names = {getattr(t, "__name__") for t in skill.tools if callable(t)}
        assert tool_names == {"search", "list_documents", "get_document", "cite"}

    def test_create_skill_has_state(self, test_app_config, temp_db_path):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill._state_type is RAGState
        assert skill._state_namespace == "rag"

    def test_create_skill_has_extras(self, test_app_config, temp_db_path):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.extras["config"] is test_app_config
        assert skill.extras["db_path"] is temp_db_path
        assert "visualize_chunk" in skill.extras
        assert "list_documents" in skill.extras

    def test_create_skill_from_env(self, monkeypatch, temp_db_path):
        monkeypatch.setenv("HAIKU_RAG_DB", str(temp_db_path))
        from haiku.rag.skills.rag import create_skill

        skill = create_skill()
        assert skill.metadata.name == "rag"


class TestSkillExtras:
    async def test_list_documents_returns_all(self, test_app_config, rag_db):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(config=test_app_config, db_path=rag_db)
        list_docs = skill.extras["list_documents"]
        results = await list_docs()
        assert len(results) == 2
        assert all(k in results[0] for k in ("id", "title", "uri", "metadata"))

    async def test_list_documents_with_filter(self, test_app_config, rag_db):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(config=test_app_config, db_path=rag_db)
        list_docs = skill.extras["list_documents"]
        results = await list_docs(filter="title = 'AI Overview'")
        assert len(results) == 1
        assert results[0]["title"] == "AI Overview"


class TestSearchTool:
    async def test_search_returns_formatted_string(self, rag_db, rag_client):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        ctx = _make_ctx(rag=rag_client)
        result = await search(ctx, query="artificial intelligence")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_search_updates_state(self, rag_db, rag_client):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        state = RAGState()
        ctx = _make_ctx(state, rag=rag_client)
        await search(ctx, query="artificial intelligence")
        assert "artificial intelligence" in state.searches
        results = state.searches["artificial intelligence"]
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

    async def test_search_applies_document_filter_from_state(self, rag_db, rag_client):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        state = RAGState(document_filter="title = 'AI Overview'")
        ctx = _make_ctx(state, rag=rag_client)
        result = await search(ctx, query="artificial intelligence")
        assert "AI Overview" in result
        assert "ML Basics" not in result

    async def test_search_without_state(self, rag_db, rag_client):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        ctx = _make_ctx(state=None, rag=rag_client)
        result = await search(ctx, query="artificial intelligence")
        assert isinstance(result, str)

    async def test_search_rate_limited(self, rag_db, rag_client):
        from haiku.rag.skills.rag import RAGState, create_skill

        config = AppConfig()
        config.qa.max_searches = 2
        skill = create_skill(db_path=rag_db, config=config)
        search = _get_tool(skill, "search")
        state = RAGState()
        ctx = _make_ctx(state, rag=rag_client)

        await search(ctx, query="first")
        await search(ctx, query="second")
        result = await search(ctx, query="third")
        assert "Search limit reached" in result
        assert ctx.deps.search_count == 3
        assert len(state.searches) == 2


class TestListDocumentsTool:
    async def test_list_documents_returns_results(self, rag_db, rag_client):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        list_docs = _get_tool(skill, "list_documents")
        ctx = _make_ctx(rag=rag_client)
        results = await list_docs(ctx)
        assert isinstance(results, list)
        assert len(results) == 2

    async def test_list_documents_applies_document_filter_from_state(
        self, rag_db, rag_client
    ):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        list_docs = _get_tool(skill, "list_documents")
        state = RAGState(document_filter="title = 'AI Overview'")
        ctx = _make_ctx(state, rag=rag_client)
        results = await list_docs(ctx)
        assert len(results) == 1
        assert results[0]["title"] == "AI Overview"


class TestGetDocumentTool:
    async def test_get_document_by_title(self, rag_db, rag_client):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        get_doc = _get_tool(skill, "get_document")
        ctx = _make_ctx(rag=rag_client)
        result = await get_doc(ctx, query="AI Overview")
        assert result is not None
        assert result["title"] == "AI Overview"

    async def test_get_document_not_found(self, rag_db, rag_client):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        get_doc = _get_tool(skill, "get_document")
        ctx = _make_ctx(rag=rag_client)
        result = await get_doc(ctx, query="nonexistent document xyz")
        assert result is None


class TestCiteTool:
    async def test_cite_registers_citations(self, rag_db, rag_client):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        cite = _get_tool(skill, "cite")
        state = RAGState()
        ctx = _make_ctx(state, rag=rag_client)

        await search(ctx, query="artificial intelligence")
        chunk_ids = [
            sr.chunk_id
            for results in state.searches.values()
            for sr in results
            if sr.chunk_id
        ][:2]

        result = await cite(ctx, chunk_ids=chunk_ids)
        assert "Registered" in result
        assert len(state.citations) == 2
        assert all(cid in state.citations for cid in chunk_ids)
        assert all(cid in state.citation_index for cid in chunk_ids)

    async def test_cite_deduplicates_in_index(self, rag_db, rag_client):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        cite = _get_tool(skill, "cite")
        state = RAGState()
        ctx = _make_ctx(state, rag=rag_client)

        await search(ctx, query="artificial intelligence")
        chunk_ids = [
            sr.chunk_id
            for results in state.searches.values()
            for sr in results
            if sr.chunk_id
        ][:1]

        await cite(ctx, chunk_ids=chunk_ids)
        await cite(ctx, chunk_ids=chunk_ids)
        assert len(state.citation_index) == 1
        assert len(state.citations) == 1

    async def test_cite_without_state(self, rag_db):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        cite = _get_tool(skill, "cite")
        ctx = _make_ctx(state=None)
        result = await cite(ctx, chunk_ids=["nonexistent"])
        assert "No state" in result


class TestLifespan:
    async def test_opens_one_client_per_invocation(self, rag_db):
        """Lifespan opens one HaikuRAG client, available on ctx.deps.rag throughout."""
        from haiku.rag.skills._deps import RAGRunDeps, make_rag_lifespan

        config = AppConfig()
        lifespan = make_rag_lifespan(rag_db, config)
        deps = RAGRunDeps()
        async with lifespan(deps):
            assert deps.rag is not None
            assert deps.rag.is_read_only
            assert deps.search_count == 0
            docs = await deps.rag.list_documents()
            assert len(docs) == 2
        # after exit the client has been closed; field still references it
        assert deps.rag is not None

    async def test_search_count_resets_per_invocation(self, rag_db):
        from haiku.rag.skills._deps import RAGRunDeps, make_rag_lifespan

        config = AppConfig()
        lifespan = make_rag_lifespan(rag_db, config)

        deps = RAGRunDeps(search_count=42)
        async with lifespan(deps):
            assert deps.search_count == 0

        deps2 = RAGRunDeps(search_count=5)
        async with lifespan(deps2):
            assert deps2.search_count == 0

    def test_skill_has_lifespan_and_deps_type(self, test_app_config, temp_db_path):
        from haiku.rag.skills._deps import RAGRunDeps
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(config=test_app_config, db_path=temp_db_path)
        assert skill.deps_type is RAGRunDeps
        assert skill.lifespan is not None

    async def test_run_skill_end_to_end_opens_and_closes_client(
        self, allow_model_requests, rag_db
    ):
        """Full sub-agent path: lifespan opens the client, tools see it, lifespan closes it."""
        from pydantic_ai.models.test import TestModel

        from haiku.rag.skills.rag import create_skill
        from haiku.skills.agent import _run_skill

        skill = create_skill(db_path=rag_db)
        result, *_ = await _run_skill(TestModel(), skill, "List the documents.")
        assert result

    async def test_lifespan_clears_citations_and_searches_but_keeps_index(self, rag_db):
        from haiku.rag.agents.research.models import Citation
        from haiku.rag.skills._deps import RAGRunDeps, make_rag_lifespan
        from haiku.rag.skills.rag import RAGState

        config = AppConfig()
        lifespan = make_rag_lifespan(rag_db, config)

        state = RAGState(
            document_filter="title = 'AI Overview'",
            citation_index={
                "c1": Citation(
                    index=1,
                    chunk_id="c1",
                    document_id="d1",
                    document_title="t",
                    document_uri="u",
                    content="x",
                    page_numbers=[],
                    headings=[],
                )
            },
            citations=["c1"],
            searches={"prior": []},
        )
        deps = RAGRunDeps(state=state)
        async with lifespan(deps):
            assert state.citations == []
            assert state.searches == {}
            assert "c1" in state.citation_index  # preserved for cross-turn lookup
            assert state.document_filter == "title = 'AI Overview'"

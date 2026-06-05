import asyncio

import pytest

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
        assert tool_names == {"search", "cite"}

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

    async def test_concurrent_searches_serialize_on_lock(
        self, rag_db, rag_client, monkeypatch
    ):
        """Tool calls in a turn run concurrently; the shared lock keeps only one
        connection operation in flight (pydantic-ai borrow guard)."""
        from haiku.rag.skills import _tools
        from haiku.rag.skills.rag import create_skill

        original = _tools.skill_search
        inflight = {"now": 0, "max": 0}

        async def tracking_search(*args, **kwargs):
            inflight["now"] += 1
            inflight["max"] = max(inflight["max"], inflight["now"])
            try:
                await asyncio.sleep(0.05)  # widen the window an overlap would use
                return await original(*args, **kwargs)
            finally:
                inflight["now"] -= 1

        monkeypatch.setattr(_tools, "skill_search", tracking_search)

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        ctx = _make_ctx(rag=rag_client)  # one ctx -> one shared lock, as in a turn
        results = await asyncio.gather(
            search(ctx, query="artificial intelligence"),
            search(ctx, query="machine learning"),
        )
        assert all(isinstance(r, str) and r for r in results)
        assert inflight["max"] == 1

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

    async def test_cite_raises_modelretry_when_chunk_ids_unresolved(
        self, rag_db, rag_client
    ):
        """When supplied chunk_ids don't match any search result, cite raises
        ModelRetry so pydantic-ai prompts the model to retry with valid ids
        instead of silently registering zero citations."""
        from pydantic_ai import ModelRetry

        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        cite = _get_tool(skill, "cite")
        state = RAGState()
        ctx = _make_ctx(state, rag=rag_client)

        await search(ctx, query="artificial intelligence")
        assert state.searches, "fixture should have produced some search results"

        with pytest.raises(ModelRetry) as exc_info:
            await cite(ctx, chunk_ids=["372c9ddf-not-a-real-id"])
        message = str(exc_info.value)
        assert "verbatim" in message
        assert "372c9ddf-not-a-real-id" in message

    async def test_cite_raises_modelretry_when_chunk_id_does_not_exist(
        self, rag_db, rag_client
    ):
        """With no prior search() and a chunk_id that doesn't exist in the DB,
        cite raises ModelRetry — the DB-fallback path tried and found nothing."""
        from pydantic_ai import ModelRetry

        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        cite = _get_tool(skill, "cite")
        state = RAGState()
        ctx = _make_ctx(state, rag=rag_client)
        assert not state.searches

        with pytest.raises(ModelRetry) as exc_info:
            await cite(ctx, chunk_ids=["nonexistent-chunk-id"])
        message = str(exc_info.value)
        assert "verbatim" in message
        assert "nonexistent-chunk-id" in message

    async def test_cite_returns_message_when_chunk_ids_empty(self, rag_db):
        """An empty chunk_ids list is a no-op, not a retry trigger."""
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        cite = _get_tool(skill, "cite")
        state = RAGState()
        ctx = _make_ctx(state)
        result = await cite(ctx, chunk_ids=[])
        assert "0" in result

    async def test_cite_accepts_chunk_id_from_db_without_prior_search(
        self, rag_db, rag_client
    ):
        """chunk_ids sourced from items.jsonl / toc.json are valid citations.

        The skill calls cite directly with chunk_ids it read from the VFS;
        no search() has been recorded in state.searches. cite must look the
        chunk up in the DB and build a Citation with full document context.
        """
        from haiku.rag.skills.rag import RAGState, create_skill

        docs = await rag_client.list_documents(limit=1)
        assert docs, "fixture should have at least one document"
        doc_id = docs[0].id
        chunks = await rag_client.chunk_repository.get_by_document_id(doc_id)
        assert chunks, "fixture document should have chunks"
        chunk_id = chunks[0].id

        skill = create_skill(db_path=rag_db)
        cite = _get_tool(skill, "cite")
        state = RAGState()
        ctx = _make_ctx(state, rag=rag_client)
        assert not state.searches, "this test exercises the no-prior-search path"

        result = await cite(ctx, chunk_ids=[chunk_id])
        assert "Registered 1 citation(s)." == result
        assert chunk_id in state.citations
        registered = state.citation_index[chunk_id]
        assert registered.document_id == doc_id
        assert registered.document_uri  # uri must be populated from doc lookup


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
        from haiku.skills.agent import run_skill

        skill = create_skill(db_path=rag_db)
        result, *_ = await run_skill(TestModel(), skill, "List the documents.")
        assert result

    async def test_lifespan_clears_citations_and_searches_but_keeps_index(self, rag_db):
        from haiku.rag.skills._deps import RAGRunDeps, make_rag_lifespan
        from haiku.rag.skills.rag import RAGState
        from haiku.rag.store.models.citation import Citation

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

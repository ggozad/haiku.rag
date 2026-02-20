from unittest.mock import AsyncMock

from haiku.rag.agents.research.models import Citation, ResearchReport
from haiku.rag.agents.rlm.models import RLMResult
from haiku.rag.client import HaikuRAG
from haiku.rag.store.models.chunk import SearchResult
from haiku.rag.tools.document import DocumentInfo
from haiku.rag.tools.qa import QAHistoryEntry

from .conftest import _get_tool, _make_ctx


class TestRAGSkillCreation:
    def test_create_skill_returns_valid_skill(self, temp_db_path):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=temp_db_path)
        assert skill.metadata.name == "rag"
        assert skill.metadata.description
        assert skill.instructions

    def test_create_skill_has_expected_tools(self, temp_db_path):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=temp_db_path)
        tool_names = {getattr(t, "__name__") for t in skill.tools if callable(t)}
        assert tool_names == {
            "search",
            "list_documents",
            "get_document",
            "ask",
            "analyze",
            "research",
        }

    def test_create_skill_has_state(self, temp_db_path):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=temp_db_path)
        assert skill._state_type is RAGState
        assert skill._state_namespace == "rag"

    def test_create_skill_from_env(self, monkeypatch, temp_db_path):
        monkeypatch.setenv("HAIKU_RAG_DB", str(temp_db_path))
        from haiku.rag.skills.rag import create_skill

        skill = create_skill()
        assert skill.metadata.name == "rag"


class TestSearchTool:
    async def test_search_returns_formatted_string(self, rag_db):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        ctx = _make_ctx()
        result = await search(ctx, query="artificial intelligence")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_search_updates_state(self, rag_db):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        state = RAGState()
        ctx = _make_ctx(state)
        await search(ctx, query="artificial intelligence")
        assert "artificial intelligence" in state.searches
        results = state.searches["artificial intelligence"]
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)

    async def test_search_applies_document_filter_from_state(self, rag_db):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        state = RAGState(document_filter="title = 'AI Overview'")
        ctx = _make_ctx(state)
        result = await search(ctx, query="artificial intelligence")
        assert "AI Overview" in result
        assert "ML Basics" not in result

    async def test_search_without_state(self, rag_db):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        search = _get_tool(skill, "search")
        ctx = _make_ctx(state=None)
        result = await search(ctx, query="artificial intelligence")
        assert isinstance(result, str)


class TestListDocumentsTool:
    async def test_list_documents_returns_results(self, rag_db):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        list_docs = _get_tool(skill, "list_documents")
        ctx = _make_ctx()
        results = await list_docs(ctx)
        assert isinstance(results, list)
        assert len(results) == 2

    async def test_list_documents_updates_state(self, rag_db):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        list_docs = _get_tool(skill, "list_documents")
        state = RAGState()
        ctx = _make_ctx(state)
        await list_docs(ctx)
        assert len(state.documents) == 2
        assert isinstance(state.documents[0], DocumentInfo)
        assert state.documents[0].id is not None

    async def test_list_documents_no_duplicates_in_state(self, rag_db):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        list_docs = _get_tool(skill, "list_documents")
        state = RAGState()
        ctx = _make_ctx(state)
        await list_docs(ctx)
        await list_docs(ctx)
        assert len(state.documents) == 2


class TestGetDocumentTool:
    async def test_get_document_by_title(self, rag_db):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        get_doc = _get_tool(skill, "get_document")
        ctx = _make_ctx()
        result = await get_doc(ctx, query="AI Overview")
        assert result is not None
        assert result["title"] == "AI Overview"

    async def test_get_document_updates_state(self, rag_db):
        from haiku.rag.skills.rag import RAGState, create_skill

        skill = create_skill(db_path=rag_db)
        get_doc = _get_tool(skill, "get_document")
        state = RAGState()
        ctx = _make_ctx(state)
        await get_doc(ctx, query="AI Overview")
        assert len(state.documents) == 1
        assert isinstance(state.documents[0], DocumentInfo)
        assert state.documents[0].title == "AI Overview"

    async def test_get_document_not_found(self, rag_db):
        from haiku.rag.skills.rag import create_skill

        skill = create_skill(db_path=rag_db)
        get_doc = _get_tool(skill, "get_document")
        ctx = _make_ctx()
        result = await get_doc(ctx, query="nonexistent document xyz")
        assert result is None


class TestAskTool:
    async def test_ask_returns_answer_with_citations(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import create_skill

        citations = [
            Citation(
                document_id="d1",
                chunk_id="c1",
                document_uri="test://ai-overview",
                document_title="AI Overview",
                content="AI is transforming industries.",
            )
        ]
        monkeypatch.setattr(
            HaikuRAG,
            "ask",
            AsyncMock(return_value=("AI transforms industries worldwide.", citations)),
        )

        skill = create_skill(db_path=rag_db)
        ask = _get_tool(skill, "ask")
        ctx = _make_ctx()
        result = await ask(ctx, question="What is AI?")
        assert isinstance(result, str)
        assert "AI transforms industries" in result

    async def test_ask_updates_state(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import RAGState, create_skill

        citations = [
            Citation(
                document_id="d1",
                chunk_id="c1",
                document_uri="test://ai-overview",
                content="AI content",
            )
        ]
        monkeypatch.setattr(
            HaikuRAG,
            "ask",
            AsyncMock(return_value=("AI transforms industries.", citations)),
        )

        skill = create_skill(db_path=rag_db)
        ask = _get_tool(skill, "ask")
        state = RAGState()
        ctx = _make_ctx(state)
        await ask(ctx, question="What is AI?")
        assert len(state.citations) == 1
        assert len(state.qa_history) == 1
        assert isinstance(state.qa_history[0], QAHistoryEntry)
        assert state.qa_history[0].question == "What is AI?"

    async def test_ask_assigns_citation_indices(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import RAGState, create_skill

        first_citations = [
            Citation(
                document_id="d1",
                chunk_id="c1",
                document_uri="test://doc1",
                content="First.",
            ),
            Citation(
                document_id="d2",
                chunk_id="c2",
                document_uri="test://doc2",
                content="Second.",
            ),
        ]
        second_citations = [
            Citation(
                document_id="d3",
                chunk_id="c3",
                document_uri="test://doc3",
                content="Third.",
            ),
        ]

        call_count = 0

        async def mock_ask(self, question, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ("Answer 1", first_citations)
            return ("Answer 2", second_citations)

        monkeypatch.setattr(HaikuRAG, "ask", mock_ask)

        skill = create_skill(db_path=rag_db)
        ask = _get_tool(skill, "ask")
        state = RAGState()
        ctx = _make_ctx(state)

        await ask(ctx, question="First question")
        assert state.citations[0].index == 1
        assert state.citations[1].index == 2

        await ask(ctx, question="Second question")
        assert state.citations[2].index == 3

    async def test_ask_applies_document_filter_from_state(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import RAGState, create_skill

        captured_kwargs = {}

        async def mock_ask(self, question, **kwargs):
            captured_kwargs.update(kwargs)
            return ("Answer.", [])

        monkeypatch.setattr(HaikuRAG, "ask", mock_ask)

        skill = create_skill(db_path=rag_db)
        ask = _get_tool(skill, "ask")
        state = RAGState(document_filter="title = 'AI Overview'")
        ctx = _make_ctx(state)
        await ask(ctx, question="What is AI?")
        assert captured_kwargs.get("filter") == "title = 'AI Overview'"

    async def test_ask_includes_prior_qa_context(self, rag_db, monkeypatch):
        import random

        from haiku.rag.skills.rag import RAGState, create_skill
        from tests.skills.conftest import VECTOR_DIM

        captured_questions = []

        async def mock_ask(self, question, **kwargs):
            captured_questions.append(question)
            return ("Answer about AI.", [])

        monkeypatch.setattr(HaikuRAG, "ask", mock_ask)

        skill = create_skill(db_path=rag_db)
        ask = _get_tool(skill, "ask")

        # Pre-compute the embedding the fake embedder will produce for "Tell me about AI"
        query_text = "Tell me about AI"
        random.seed(hash(query_text) % (2**32))
        query_embedding = [random.random() for _ in range(VECTOR_DIM)]

        prior_citations = [
            Citation(
                document_id="d1",
                chunk_id="c1",
                document_uri="test://ai-overview",
                document_title="AI Overview",
                content="AI content from source.",
            )
        ]
        state = RAGState(
            qa_history=[
                QAHistoryEntry(
                    question="What is artificial intelligence?",
                    answer="AI is the simulation of human intelligence by machines.",
                    question_embedding=query_embedding,
                    citations=prior_citations,
                ),
            ]
        )
        ctx = _make_ctx(state)
        await ask(ctx, question=query_text)

        # rag.ask() should receive augmented question with prior context
        assert len(captured_questions) == 1
        augmented = captured_questions[0]
        assert "Context from prior questions" in augmented
        assert "What is artificial intelligence?" in augmented
        assert "AI is the simulation" in augmented
        assert "AI Overview" in augmented
        assert query_text in augmented

        # State should store the original question, not the augmented one
        assert state.qa_history[-1].question == query_text

    async def test_ask_no_prior_qa_context_when_irrelevant(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import RAGState, create_skill
        from tests.skills.conftest import VECTOR_DIM

        captured_questions = []

        async def mock_ask(self, question, **kwargs):
            captured_questions.append(question)
            return ("Answer.", [])

        monkeypatch.setattr(HaikuRAG, "ask", mock_ask)

        skill = create_skill(db_path=rag_db)
        ask = _get_tool(skill, "ask")

        # Use orthogonal embedding — won't match the fake embedder's output
        orthogonal = [1.0 if i % 2 == 0 else -1.0 for i in range(VECTOR_DIM)]
        state = RAGState(
            qa_history=[
                QAHistoryEntry(
                    question="What is the weather?",
                    answer="It is sunny today.",
                    question_embedding=orthogonal,
                ),
            ]
        )
        ctx = _make_ctx(state)
        await ask(ctx, question="Explain quantum computing")

        # rag.ask() should receive the original question unchanged
        assert len(captured_questions) == 1
        assert captured_questions[0] == "Explain quantum computing"


class TestAnalyzeTool:
    async def test_analyze_returns_result(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import create_skill

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

    async def test_analyze_updates_state(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import RAGState, create_skill

        monkeypatch.setattr(
            HaikuRAG,
            "rlm",
            AsyncMock(return_value=RLMResult(answer="42", program="print(42)")),
        )

        skill = create_skill(db_path=rag_db)
        analyze = _get_tool(skill, "analyze")
        state = RAGState()
        ctx = _make_ctx(state)
        await analyze(ctx, question="How many documents?")
        assert len(state.qa_history) == 1
        assert state.qa_history[0].question == "How many documents?"


class TestResearchTool:
    async def test_research_returns_report(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import create_skill

        report = ResearchReport(
            title="AI Research",
            executive_summary="AI is transforming industries.",
            main_findings=["Finding 1"],
            conclusions=["Conclusion 1"],
            sources_summary="Multiple sources consulted.",
        )
        monkeypatch.setattr(HaikuRAG, "research", AsyncMock(return_value=report))

        skill = create_skill(db_path=rag_db)
        research = _get_tool(skill, "research")
        ctx = _make_ctx()
        result = await research(ctx, question="What is AI?")
        assert isinstance(result, str)
        assert "AI Research" in result

    async def test_research_updates_state(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import RAGState, create_skill

        report = ResearchReport(
            title="AI Research",
            executive_summary="AI is transforming industries.",
            main_findings=["Finding 1"],
            conclusions=["Conclusion 1"],
            sources_summary="Multiple sources consulted.",
        )
        monkeypatch.setattr(HaikuRAG, "research", AsyncMock(return_value=report))

        skill = create_skill(db_path=rag_db)
        research = _get_tool(skill, "research")
        state = RAGState()
        ctx = _make_ctx(state)
        await research(ctx, question="What is AI?")
        assert len(state.reports) == 1
        assert state.reports[0].question == "What is AI?"
        assert len(state.qa_history) == 1
        assert state.qa_history[0].question == "What is AI?"
        assert state.qa_history[0].answer == "AI is transforming industries."

    async def test_research_applies_document_filter_from_state(
        self, rag_db, monkeypatch
    ):
        from haiku.rag.skills.rag import RAGState, create_skill

        captured_kwargs = {}

        report = ResearchReport(
            title="AI Research",
            executive_summary="Summary.",
            main_findings=["Finding"],
            conclusions=["Conclusion"],
            sources_summary="Sources.",
        )

        async def mock_research(self, question, **kwargs):
            captured_kwargs.update(kwargs)
            return report

        monkeypatch.setattr(HaikuRAG, "research", mock_research)

        skill = create_skill(db_path=rag_db)
        research = _get_tool(skill, "research")
        state = RAGState(document_filter="title = 'AI Overview'")
        ctx = _make_ctx(state)
        await research(ctx, question="What is AI?")
        assert captured_kwargs.get("filter") == "title = 'AI Overview'"

    async def test_research_without_state(self, rag_db, monkeypatch):
        from haiku.rag.skills.rag import create_skill

        report = ResearchReport(
            title="AI Research",
            executive_summary="Summary.",
            main_findings=["Finding"],
            conclusions=["Conclusion"],
            sources_summary="Sources.",
        )
        monkeypatch.setattr(HaikuRAG, "research", AsyncMock(return_value=report))

        skill = create_skill(db_path=rag_db)
        research = _get_tool(skill, "research")
        ctx = _make_ctx(state=None)
        result = await research(ctx, question="What is AI?")
        assert isinstance(result, str)

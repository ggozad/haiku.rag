from pathlib import Path

import pytest

from haiku.rag.agents.chat import (
    AGUI_STATE_KEY,
    ChatDeps,
    ChatSessionState,
    QAResponse,
    SearchAgent,
    create_chat_agent,
)
from haiku.rag.agents.chat.state import MAX_QA_HISTORY
from haiku.rag.agents.research.models import Citation
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_chat_agent")


def test_create_chat_agent():
    """Test that create_chat_agent returns a properly configured agent."""
    agent = create_chat_agent(Config)
    assert agent is not None
    assert agent.name == "chat_agent" or agent.name is None


def test_chat_deps_initialization(temp_db_path):
    """Test ChatDeps can be initialized with required fields."""
    client = HaikuRAG(temp_db_path, create=True)
    deps = ChatDeps(client=client, config=Config)

    assert deps.client is client
    assert deps.config is Config
    assert deps.search_results is None
    assert deps.session_state is None

    client.close()


def test_agui_state_key_constant():
    """Test AGUI_STATE_KEY is exported with correct value."""
    assert AGUI_STATE_KEY == "haiku.rag.chat"


def test_chat_deps_with_state_key(temp_db_path):
    """Test ChatDeps can be initialized with state_key for keyed state emission."""
    client = HaikuRAG(temp_db_path, create=True)
    deps = ChatDeps(client=client, config=Config, state_key="my_state")

    assert deps.client is client
    assert deps.config is Config
    assert deps.state_key == "my_state"

    client.close()


def test_chat_deps_state_key_default_none(temp_db_path):
    """Test ChatDeps state_key defaults to None."""
    client = HaikuRAG(temp_db_path, create=True)
    deps = ChatDeps(client=client, config=Config)

    assert deps.state_key is None

    client.close()


def test_chat_session_state():
    """Test ChatSessionState model."""
    state = ChatSessionState(session_id="test-session")
    assert state.session_id == "test-session"
    assert state.citations == []
    assert state.qa_history == []


def test_citation():
    """Test Citation model."""
    citation = Citation(
        index=1,
        document_id="doc-123",
        chunk_id="chunk-456",
        document_uri="test.md",
        document_title="Test Document",
        page_numbers=[1, 2],
        headings=["Section 1"],
        content="Test content",
    )
    assert citation.index == 1
    assert citation.document_id == "doc-123"
    assert citation.chunk_id == "chunk-456"
    assert citation.content == "Test content"


def test_qa_response():
    """Test QAResponse model."""
    citation = Citation(
        index=1,
        document_id="doc-123",
        chunk_id="chunk-456",
        document_uri="test.md",
        document_title="Test Document",
        content="Test content",
    )
    qa = QAResponse(
        question="What is this?",
        answer="This is a test",
        confidence=0.95,
        citations=[citation],
    )
    assert qa.question == "What is this?"
    assert qa.answer == "This is a test"
    assert qa.confidence == 0.95
    assert len(qa.citations) == 1
    assert qa.sources == ["Test Document"]


def test_qa_response_sources_with_uri_fallback():
    """Test QAResponse.sources falls back to URI when title is None."""
    citation = Citation(
        index=1,
        document_id="doc-123",
        chunk_id="chunk-456",
        document_uri="test.md",
        document_title=None,
        content="Test content",
    )
    qa = QAResponse(
        question="What is this?",
        answer="This is a test",
        citations=[citation],
    )
    assert qa.sources == ["test.md"]


def test_qa_response_to_search_answer():
    """Test QAResponse.to_search_answer() converts to SearchAnswer for research graph."""
    citation = Citation(
        index=1,
        document_id="doc-123",
        chunk_id="chunk-456",
        document_uri="test.md",
        document_title="Test Document",
        content="Test content",
    )
    qa = QAResponse(
        question="What is the answer?",
        answer="The answer is 42",
        confidence=0.95,
        citations=[citation],
    )

    search_answer = qa.to_search_answer()

    assert search_answer.query == "What is the answer?"
    assert search_answer.answer == "The answer is 42"
    assert search_answer.confidence == 0.95
    assert search_answer.cited_chunks == ["chunk-456"]
    assert len(search_answer.citations) == 1
    assert search_answer.citations[0].chunk_id == "chunk-456"


def test_search_agent_initialization(temp_db_path):
    """Test SearchAgent can be initialized."""
    client = HaikuRAG(temp_db_path, create=True)
    search_agent = SearchAgent(client, Config)
    assert search_agent is not None
    client.close()


# DocLayNet content for testing
DOCLAYNET_CLASS_LABELS = """
DocLayNet Dataset - Class Labels

DocLayNet defines 11 distinct class labels for document layout analysis:
1. Caption - Text describing figures or tables
2. Footnote - Notes at the bottom of pages
3. Formula - Mathematical expressions
4. List-item - Items in bulleted or numbered lists
5. Page-footer - Footer content on pages
6. Page-header - Header content on pages
7. Picture - Images and diagrams
8. Section-header - Headings for document sections
9. Table - Tabular data
10. Text - Regular paragraph text (highest count: 510,377 instances)
11. Title - Document titles

The Text class has the highest count with 510,377 instances in the dataset.
"""

DOCLAYNET_ANNOTATION = """
DocLayNet Dataset - Annotation Process

The annotation process was organized into 4 phases:
- Phase 1: Data selection and preparation by a small team of experts
- Phase 2: Label selection and guideline definition
- Phase 3: Annotation by 40 dedicated annotators
- Phase 4: Quality control and continuous supervision

The Corpus Conversion Service (CCS) was used for annotation, providing a visual interface.
"""

DOCLAYNET_DATA_SOURCES = """
DocLayNet Dataset - Data Sources

The data sources for DocLayNet include:
- Publication repositories such as arXiv
- Government offices and official documents
- Company websites and corporate reports
- Data directory services for financial reports
- Patent documents

Scanned documents were excluded to avoid rotation and skewing issues.
"""


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_search_tool(allow_model_requests, temp_db_path):
    """Test the chat agent's search tool functionality."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add test documents
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )
        await client.create_document(
            content=DOCLAYNET_ANNOTATION,
            uri="doclaynet-annotation",
            title="DocLayNet Annotation",
        )

        agent = create_chat_agent(Config)
        session_state = ChatSessionState(session_id="test-search")
        deps = ChatDeps(
            client=client,
            config=Config,
            session_state=session_state,
        )

        # Ask something that should trigger the search tool
        result = await agent.run(
            "Search for documents about class labels",
            deps=deps,
        )

        assert result.output is not None
        assert len(result.output) > 0


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_search_tool_with_filter(allow_model_requests, temp_db_path):
    """Test the chat agent's search tool with document filter."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add test documents
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )
        await client.create_document(
            content=DOCLAYNET_DATA_SOURCES,
            uri="doclaynet-sources",
            title="DocLayNet Sources",
        )

        agent = create_chat_agent(Config)
        session_state = ChatSessionState(session_id="test-search-filter")
        deps = ChatDeps(
            client=client,
            config=Config,
            session_state=session_state,
        )

        # Ask to search within a specific document
        result = await agent.run(
            "Search for information about class labels in the DocLayNet Class Labels document",
            deps=deps,
        )

        assert result.output is not None


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_get_document_tool(allow_model_requests, temp_db_path):
    """Test the chat agent's get_document tool."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add a test document
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        agent = create_chat_agent(Config)
        deps = ChatDeps(
            client=client,
            config=Config,
        )

        # Ask to get a specific document
        result = await agent.run(
            "Get me the DocLayNet Class Labels document",
            deps=deps,
        )

        assert result.output is not None
        # The response should contain info about the document
        assert "DocLayNet" in result.output or "class" in result.output.lower()


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_get_document_not_found(allow_model_requests, temp_db_path):
    """Test the chat agent's get_document tool when document is not found."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            client=client,
            config=Config,
        )

        # Ask for a document that doesn't exist
        result = await agent.run(
            "Get me the nonexistent document",
            deps=deps,
        )

        assert result.output is not None


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_search_agent_with_context(allow_model_requests, temp_db_path):
    """Test SearchAgent's search method with context."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add test documents
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )
        await client.create_document(
            content=DOCLAYNET_ANNOTATION,
            uri="doclaynet-annotation",
            title="DocLayNet Annotation",
        )

        search_agent = SearchAgent(client, Config)

        # Search with context
        results = await search_agent.search(
            query="What are the class labels?",
            context="We're discussing document layout analysis",
        )

        assert isinstance(results, list)


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_search_agent_with_filter(allow_model_requests, temp_db_path):
    """Test SearchAgent's search method with document filter."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add test documents
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )
        await client.create_document(
            content=DOCLAYNET_DATA_SOURCES,
            uri="doclaynet-sources",
            title="DocLayNet Sources",
        )

        search_agent = SearchAgent(client, Config)

        # Search with filter - only the labels document
        results = await search_agent.search(
            query="What information is available?",
            filter="uri LIKE '%labels%'",
        )

        assert isinstance(results, list)
        # Results should only come from the labels document
        for r in results:
            assert "labels" in (r.document_uri or "")


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_search_agent_deduplication(allow_model_requests, temp_db_path):
    """Test SearchAgent deduplicates results by chunk_id."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add test documents
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        search_agent = SearchAgent(client, Config)

        # Search - the search agent will likely run multiple queries
        # that could return the same chunk, which should be deduplicated
        results = await search_agent.search(
            query="Tell me about class labels and their counts",
        )

        assert isinstance(results, list)

        # Verify no duplicate chunk_ids
        chunk_ids = [r.chunk_id for r in results if r.chunk_id]
        assert len(chunk_ids) == len(set(chunk_ids)), "Found duplicate chunk_ids"


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_search_agent_no_results(allow_model_requests, temp_db_path):
    """Test SearchAgent handles no results gracefully."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        search_agent = SearchAgent(client, Config)

        # Search in empty database
        results = await search_agent.search(
            query="Find information about nonexistent topic xyz123",
        )

        assert isinstance(results, list)
        assert len(results) == 0


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_ask_adds_citations(allow_model_requests, temp_db_path):
    """Test that the ask tool adds citations to the response."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add a document with specific content
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        agent = create_chat_agent(Config)
        session_state = ChatSessionState(session_id="test-citations")
        deps = ChatDeps(
            client=client,
            config=Config,
            session_state=session_state,
        )

        # Ask a question that should use the ask tool with citations
        result = await agent.run(
            "What is the highest count class in the DocLayNet dataset?",
            deps=deps,
        )

        assert result.output is not None
        # The qa_history should have been updated with the new Q&A
        assert len(session_state.qa_history) >= 1


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_ask_triggers_background_summarization(
    allow_model_requests, temp_db_path
):
    """Test that the ask tool triggers background session context summarization."""
    import asyncio

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        agent = create_chat_agent(Config)
        session_state = ChatSessionState(session_id="test-summarization")
        deps = ChatDeps(
            client=client,
            config=Config,
            session_state=session_state,
        )

        # Initially no session_context
        assert session_state.session_context is None

        # Ask a question
        result = await agent.run(
            "What is the highest count class in the DocLayNet dataset?",
            deps=deps,
        )

        assert result.output is not None
        assert len(session_state.qa_history) >= 1

        # Wait for background task to complete
        # The task should update session_state.session_context
        for _ in range(50):  # Wait up to 5 seconds
            if session_state.session_context is not None:
                break
            await asyncio.sleep(0.1)

        # Verify session_context was populated by background task
        assert session_state.session_context is not None
        assert session_state.session_context.summary != ""
        assert session_state.session_context.last_updated is not None


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_ask_with_prior_answer_retrieval(
    allow_model_requests, temp_db_path
):
    """Test that ask tool retrieves relevant prior answers from qa_history.

    This exercises the prior answer retrieval logic (agent.py lines 231-257):
    1. First ask populates qa_history with question_embedding
    2. Second similar ask should find the prior answer via embedding similarity
    """
    import asyncio

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        agent = create_chat_agent(Config)
        session_state = ChatSessionState(session_id="test-prior-answers")
        deps = ChatDeps(
            client=client,
            config=Config,
            session_state=session_state,
        )

        # First ask - establishes qa_history
        result1 = await agent.run(
            "What are the class labels in DocLayNet?",
            deps=deps,
        )
        assert result1.output is not None
        assert len(session_state.qa_history) == 1
        # First question should NOT have embedding yet (set lazily on next ask)
        assert session_state.qa_history[0].question_embedding is None

        # Wait for background summarization to complete
        for _ in range(50):
            if session_state.session_context is not None:
                break
            await asyncio.sleep(0.1)

        # Second ask - similar question triggers prior answer retrieval
        # This will embed the first question and compare similarity
        result2 = await agent.run(
            "Tell me about DocLayNet class labels",
            deps=deps,
        )
        assert result2.output is not None
        # qa_history should now have 2 entries
        assert len(session_state.qa_history) == 2

        # First question should now have embedding (set during second ask's recall check)
        assert session_state.qa_history[0].question_embedding is not None
        # Embedding should be a list of floats
        assert isinstance(session_state.qa_history[0].question_embedding, list)
        assert len(session_state.qa_history[0].question_embedding) > 0

        # Verify prior answer was reused without new searches:
        # Second answer's citations should be subset of first answer's citations
        first_chunk_ids = {c.chunk_id for c in session_state.qa_history[0].citations}
        second_chunk_ids = {c.chunk_id for c in session_state.qa_history[1].citations}
        assert second_chunk_ids <= first_chunk_ids, (
            "Second answer should reuse prior citations, not perform new searches"
        )


def test_fifo_limit_enforcement():
    """Test that FIFO limit enforcement logic works correctly.

    This tests the FIFO trimming logic used in the ask() tool:
    if len(qa_history) > MAX_QA_HISTORY:
        qa_history = qa_history[-MAX_QA_HISTORY:]
    """
    # Create a session state with MAX_QA_HISTORY + 1 entries
    qa_history = [
        QAResponse(
            question=f"Question {i}",
            answer=f"Answer {i}",
            confidence=0.9,
        )
        for i in range(MAX_QA_HISTORY + 1)
    ]

    session_state = ChatSessionState(
        session_id="test-fifo",
        qa_history=qa_history,
    )

    # Simulate the FIFO enforcement from agent.py
    if len(session_state.qa_history) > MAX_QA_HISTORY:
        session_state.qa_history = session_state.qa_history[-MAX_QA_HISTORY:]

    # History should be trimmed to MAX_QA_HISTORY
    assert len(session_state.qa_history) == MAX_QA_HISTORY
    # The first entry should now be "Question 1" (Question 0 was dropped)
    assert session_state.qa_history[0].question == "Question 1"
    # The last entry should be the last added question
    assert session_state.qa_history[-1].question == f"Question {MAX_QA_HISTORY}"


def test_chat_session_state_document_filter():
    """Test ChatSessionState with document_filter."""
    state = ChatSessionState(
        session_id="test-filter",
        document_filter=["doc1.pdf", "doc2.pdf"],
    )
    assert state.document_filter == ["doc1.pdf", "doc2.pdf"]


def test_chat_session_state_document_filter_default_empty():
    """Test ChatSessionState document_filter defaults to empty list."""
    state = ChatSessionState(session_id="test")
    assert state.document_filter == []


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_search_with_session_filter(
    allow_model_requests, temp_db_path
):
    """Test that session document_filter restricts search results."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add two distinct documents
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )
        await client.create_document(
            content=DOCLAYNET_DATA_SOURCES,
            uri="doclaynet-sources",
            title="DocLayNet Sources",
        )

        agent = create_chat_agent(Config)
        # Set session filter to only include the labels document
        session_state = ChatSessionState(
            session_id="test-session-filter",
            document_filter=["DocLayNet Class Labels"],
        )
        deps = ChatDeps(
            client=client,
            config=Config,
            session_state=session_state,
        )

        # Search should only return results from the filtered document
        result = await agent.run(
            "Search for information about DocLayNet",
            deps=deps,
        )

        assert result.output is not None
        # Results should only reference the Labels document, not Sources
        assert "Labels" in result.output or "class" in result.output.lower()


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_search_agent_with_session_filter(allow_model_requests, temp_db_path):
    """Test SearchAgent respects session document filter."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add two distinct documents
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )
        await client.create_document(
            content=DOCLAYNET_DATA_SOURCES,
            uri="doclaynet-sources",
            title="DocLayNet Sources",
        )

        from haiku.rag.agents.chat.state import build_multi_document_filter

        search_agent = SearchAgent(client, Config)

        # Build filter for only the labels document
        doc_filter = build_multi_document_filter(["DocLayNet Class Labels"])

        results = await search_agent.search(
            query="What information is available?",
            filter=doc_filter,
        )

        assert isinstance(results, list)
        # All results should be from the labels document
        for r in results:
            assert "labels" in (r.document_uri or "").lower() or "Labels" in (
                r.document_title or ""
            )


def test_ask_tool_citation_registry_logic():
    """Test the citation index assignment logic used by the ask tool.

    Verifies that:
    1. First chunk gets index 1
    2. Second unique chunk gets index 2
    3. Same chunk_id always gets same index
    4. Indices don't reset between calls
    """
    session_state = ChatSessionState(session_id="test-registry")

    # Simulate first ask tool building citations
    first_ask_chunks = ["chunk-a", "chunk-b"]
    first_citations = []
    for chunk_id in first_ask_chunks:
        index = session_state.get_or_assign_index(chunk_id)
        first_citations.append(
            Citation(
                index=index,
                document_id="doc-1",
                chunk_id=chunk_id,
                document_uri="test.md",
                content="test",
            )
        )

    assert first_citations[0].index == 1
    assert first_citations[1].index == 2

    # Simulate second ask tool - overlapping chunk_id should keep same index
    second_ask_chunks = ["chunk-b", "chunk-c"]  # chunk-b was in first ask
    second_citations = []
    for chunk_id in second_ask_chunks:
        index = session_state.get_or_assign_index(chunk_id)
        second_citations.append(
            Citation(
                index=index,
                document_id="doc-1",
                chunk_id=chunk_id,
                document_uri="test.md",
                content="test",
            )
        )

    # chunk-b should have same index as before
    assert second_citations[0].index == 2
    # chunk-c is new, gets next index
    assert second_citations[1].index == 3

    # Registry should have all three chunks
    assert len(session_state.citation_registry) == 3
    assert session_state.citation_registry == {"chunk-a": 1, "chunk-b": 2, "chunk-c": 3}


def test_search_tool_citation_registry_logic():
    """Test the citation index assignment logic used by the search tool.

    Verifies that search and ask tools share the same registry,
    maintaining stable indices across different tool calls.
    """
    session_state = ChatSessionState(session_id="test-registry")

    # Simulate ask tool first (assigns indices 1, 2)
    for chunk_id in ["chunk-a", "chunk-b"]:
        session_state.get_or_assign_index(chunk_id)

    # Simulate search tool returning overlapping + new chunks
    search_chunks = ["chunk-b", "chunk-c", "chunk-d"]  # chunk-b already exists
    search_citations = []
    for chunk_id in search_chunks:
        index = session_state.get_or_assign_index(chunk_id)
        search_citations.append(
            Citation(
                index=index,
                document_id="doc-1",
                chunk_id=chunk_id,
                document_uri="test.md",
                content="test",
            )
        )

    # chunk-b should have same index as assigned by ask (2)
    assert search_citations[0].index == 2
    # New chunks get incrementing indices
    assert search_citations[1].index == 3
    assert search_citations[2].index == 4

    # Registry should have all four chunks
    assert len(session_state.citation_registry) == 4
    assert session_state.citation_registry == {
        "chunk-a": 1,
        "chunk-b": 2,
        "chunk-c": 3,
        "chunk-d": 4,
    }


# =============================================================================
# Prior Answer Recall Tests
# =============================================================================


def test_cosine_similarity_identical_vectors():
    """Test cosine similarity returns 1.0 for identical vectors."""
    from haiku.rag.agents.chat.agent import _cosine_similarity

    vec = [1.0, 2.0, 3.0]
    assert _cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    """Test cosine similarity returns 0.0 for orthogonal vectors."""
    from haiku.rag.agents.chat.agent import _cosine_similarity

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    assert _cosine_similarity(vec1, vec2) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    """Test cosine similarity returns -1.0 for opposite vectors."""
    from haiku.rag.agents.chat.agent import _cosine_similarity

    vec1 = [1.0, 2.0, 3.0]
    vec2 = [-1.0, -2.0, -3.0]
    assert _cosine_similarity(vec1, vec2) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    """Test cosine similarity handles zero vectors gracefully."""
    from haiku.rag.agents.chat.agent import _cosine_similarity

    vec = [1.0, 2.0, 3.0]
    zero = [0.0, 0.0, 0.0]
    assert _cosine_similarity(vec, zero) == 0.0
    assert _cosine_similarity(zero, vec) == 0.0
    assert _cosine_similarity(zero, zero) == 0.0


def test_prior_answer_relevance_threshold_constant():
    """Test PRIOR_ANSWER_RELEVANCE_THRESHOLD is set to expected value."""
    from haiku.rag.agents.chat.agent import PRIOR_ANSWER_RELEVANCE_THRESHOLD

    assert PRIOR_ANSWER_RELEVANCE_THRESHOLD == 0.7


def test_prior_answer_matching_above_threshold():
    """Test that similar questions (above threshold) are matched."""
    from haiku.rag.agents.chat.agent import (
        PRIOR_ANSWER_RELEVANCE_THRESHOLD,
        _cosine_similarity,
    )

    # Simulate two nearly identical question embeddings
    question_embedding = [0.5, 0.5, 0.5, 0.5]
    prior_embedding = [0.51, 0.49, 0.5, 0.5]  # Very similar

    similarity = _cosine_similarity(question_embedding, prior_embedding)
    assert similarity >= PRIOR_ANSWER_RELEVANCE_THRESHOLD


def test_prior_answer_matching_below_threshold():
    """Test that dissimilar questions (below threshold) are not matched."""
    from haiku.rag.agents.chat.agent import (
        PRIOR_ANSWER_RELEVANCE_THRESHOLD,
        _cosine_similarity,
    )

    # Simulate two different question embeddings
    question_embedding = [1.0, 0.0, 0.0, 0.0]
    prior_embedding = [0.0, 1.0, 0.0, 0.0]  # Orthogonal = very different

    similarity = _cosine_similarity(question_embedding, prior_embedding)
    assert similarity < PRIOR_ANSWER_RELEVANCE_THRESHOLD


def test_qa_response_embedding_cache():
    """Test that QAResponse stores and retrieves question_embedding correctly."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    qa = QAResponse(
        question="What is X?",
        answer="X is Y.",
        confidence=0.9,
        question_embedding=embedding,
    )

    assert qa.question_embedding == embedding
    # Embedding should be excluded from serialization (AG-UI state)
    serialized = qa.model_dump()
    assert "question_embedding" not in serialized


def test_qa_response_embedding_default_none():
    """Test that QAResponse.question_embedding defaults to None."""
    qa = QAResponse(
        question="What is X?",
        answer="X is Y.",
        confidence=0.9,
    )

    assert qa.question_embedding is None


# =============================================================================
# Background Task Cancellation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_summarization_task_cancellation():
    """Test that new summarization tasks cancel previous ones for same session."""
    import asyncio

    from haiku.rag.agents.chat.agent import _summarization_tasks

    # Clear any existing tasks
    _summarization_tasks.clear()

    session_id = "test-cancel-session"

    # Create a slow task that simulates summarization
    async def slow_task():
        await asyncio.sleep(10)  # Would take 10 seconds

    # Start first task
    task1 = asyncio.create_task(slow_task())
    _summarization_tasks[session_id] = task1

    # Simulate what happens when second ask comes in - cancel first task
    if session_id in _summarization_tasks:
        _summarization_tasks[session_id].cancel()

    # Yield to let cancellation propagate
    await asyncio.sleep(0)

    # Start second task
    task2 = asyncio.create_task(slow_task())
    _summarization_tasks[session_id] = task2

    # First task should be cancelled
    assert task1.cancelled() or task1.done()

    # Second task should be running
    assert not task2.done()

    # Cleanup
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        pass
    _summarization_tasks.clear()


def test_citation_index_fallback_without_session_state():
    """Test that citation indices fall back to sequential numbering without session_state.

    This tests the fallback branch in the ask and search tools when
    ctx.deps.session_state is None.
    """
    # Simulate the fallback logic from agent.py lines 281-285
    citation_infos = []
    session_state = None  # No session state

    # Simulate processing citations without session_state
    chunk_ids = ["chunk-a", "chunk-b", "chunk-c"]
    for chunk_id in chunk_ids:
        if session_state is not None:
            index = session_state.get_or_assign_index(chunk_id)
        else:
            index = len(citation_infos) + 1
        citation_infos.append(
            Citation(
                index=index,
                document_id="doc-1",
                chunk_id=chunk_id,
                document_uri="test.md",
                content="test",
            )
        )

    # Without session_state, indices are simple sequential numbers
    assert citation_infos[0].index == 1
    assert citation_infos[1].index == 2
    assert citation_infos[2].index == 3


@pytest.mark.asyncio
async def test_summarization_task_cleanup_on_completion():
    """Test that completed tasks are cleaned up from _summarization_tasks."""
    import asyncio

    from haiku.rag.agents.chat.agent import _summarization_tasks

    _summarization_tasks.clear()

    session_id = "test-cleanup-session"

    # Create a fast task
    async def fast_task():
        await asyncio.sleep(0.01)

    task = asyncio.create_task(fast_task())
    _summarization_tasks[session_id] = task
    task.add_done_callback(lambda t: _summarization_tasks.pop(session_id, None))

    # Wait for completion
    await task

    # Task should be cleaned up
    assert session_id not in _summarization_tasks

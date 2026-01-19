from pathlib import Path

import pytest

from haiku.rag.agents.chat import (
    AGUI_STATE_KEY,
    ChatDeps,
    ChatSessionState,
    CitationInfo,
    QAResponse,
    SearchAgent,
    create_chat_agent,
)
from haiku.rag.agents.chat.state import MAX_QA_HISTORY
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


def test_chat_agent_has_dynamic_system_prompt():
    """Test that chat agent registers a dynamic system prompt for background_context."""
    agent = create_chat_agent(Config)
    # The agent should have at least one system prompt function registered
    # (the add_background_context function)
    system_prompt_functions = getattr(agent, "_system_prompt_functions")
    assert len(system_prompt_functions) >= 1
    # Verify it's the add_background_context function
    func_names = [r.function.__name__ for r in system_prompt_functions]
    assert "add_background_context" in func_names


def test_citation_info():
    """Test CitationInfo model."""
    citation = CitationInfo(
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
    citation = CitationInfo(
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
    citation = CitationInfo(
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
async def test_chat_agent_with_qa_history_ranking(allow_model_requests, temp_db_path):
    """Test chat agent uses similarity ranking for qa_history.

    This test verifies that when qa_history has more than 5 entries,
    the ranking function is applied and the agent can still process requests.
    """
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add a simple document
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        agent = create_chat_agent(Config)

        # Build session state with pre-populated qa_history (>5 items to trigger ranking)
        session_state = ChatSessionState(
            session_id="test-ranking",
            qa_history=[
                QAResponse(
                    question="What are the 11 class labels in DocLayNet?",
                    answer="The 11 class labels are: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, and Title.",
                ),
                QAResponse(
                    question="How was the annotation process organized?",
                    answer="The annotation was organized into 4 phases.",
                ),
                QAResponse(
                    question="What data sources were used?",
                    answer="Sources include arXiv and government offices.",
                ),
                QAResponse(
                    question="How were pages selected?",
                    answer="By selective subsampling.",
                ),
                QAResponse(
                    question="What is the agreement metric?",
                    answer="The mAP metric was used.",
                ),
                QAResponse(
                    question="What is machine learning?",
                    answer="A field of AI.",
                ),
            ],
        )

        deps = ChatDeps(
            client=client,
            config=Config,
            session_state=session_state,
        )

        # Ask a question - the key test is that ranking is applied without error
        result = await agent.run(
            "What class labels are defined in the dataset?",
            deps=deps,
        )

        # Verify the agent produced a response (ranking didn't break anything)
        assert result.output is not None
        assert len(result.output) > 0

        # Verify qa_history was updated (new Q&A was added)
        assert len(session_state.qa_history) == 7  # 6 original + 1 new


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
async def test_chat_agent_search_with_state_key(allow_model_requests, temp_db_path):
    """Test search tool emits keyed state when state_key is set."""
    async with HaikuRAG(temp_db_path, create=True) as client:
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
            state_key=AGUI_STATE_KEY,
        )

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
async def test_chat_agent_ask_with_state_key(allow_model_requests, temp_db_path):
    """Test ask tool emits keyed state when state_key is set."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        agent = create_chat_agent(Config)
        session_state = ChatSessionState(session_id="test-ask-keyed")
        deps = ChatDeps(
            client=client,
            config=Config,
            session_state=session_state,
            state_key=AGUI_STATE_KEY,
        )

        result = await agent.run(
            "What is the highest count class in the DocLayNet dataset?",
            deps=deps,
        )

        assert result.output is not None
        assert len(session_state.qa_history) >= 1


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

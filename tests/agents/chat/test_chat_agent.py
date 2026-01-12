from haiku.rag.agents.chat import (
    ChatDeps,
    ChatSessionState,
    CitationInfo,
    QAResponse,
    SearchAgent,
    create_chat_agent,
)
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config


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


def test_chat_session_state():
    """Test ChatSessionState model."""
    state = ChatSessionState(session_id="test-session")
    assert state.session_id == "test-session"
    assert state.citations == []
    assert state.qa_history == []


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

from pathlib import Path

import pytest

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

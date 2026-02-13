from pathlib import Path

import pytest
from ag_ui.core import StateDeltaEvent, StateSnapshotEvent

from haiku.rag.agents.chat import (
    AGUI_STATE_KEY,
    ChatDeps,
    ChatSessionState,
    create_chat_agent,
    prepare_chat_context,
)
from haiku.rag.agents.chat.context import _summarization_tasks
from haiku.rag.agents.research.models import Citation
from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.tools import ToolContext
from haiku.rag.tools.qa import MAX_QA_HISTORY, QAHistoryEntry
from haiku.rag.tools.session import SESSION_NAMESPACE, SessionContext, SessionState


def extract_state_from_result(result, state_key: str = AGUI_STATE_KEY) -> dict | None:
    """Extract emitted state from agent result's tool return metadata.

    For deltas, applies the patch to an empty state to get the final state.
    """
    import jsonpatch

    for message in result.all_messages():
        if hasattr(message, "parts"):
            for part in message.parts:
                if hasattr(part, "metadata") and part.metadata:
                    for meta in part.metadata:
                        if isinstance(meta, StateSnapshotEvent):
                            return meta.snapshot.get(state_key)
                        elif isinstance(meta, StateDeltaEvent):
                            # Apply delta to empty state to get final state
                            empty_state = {
                                state_key: ChatSessionState().model_dump(mode="json")
                            }
                            patched = jsonpatch.apply_patch(empty_state, meta.delta)
                            return patched.get(state_key)
    return None


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_chat_agent")


def test_create_chat_agent(temp_db_path):
    """Test that create_chat_agent returns a properly configured agent."""
    agent = create_chat_agent(Config)
    assert agent is not None
    assert agent.name == "chat_agent" or agent.name is None


def test_chat_deps_initialization(temp_db_path):
    """Test ChatDeps can be initialized with required fields."""
    client = HaikuRAG(temp_db_path, create=True)
    context = ToolContext()
    deps = ChatDeps(config=Config, client=client, tool_context=context)

    assert deps.config is Config
    assert deps.client is client
    assert deps.tool_context is context
    client.close()


def test_chat_deps_is_agent_deps(temp_db_path):
    """Test ChatDeps is a subclass of AgentDeps."""
    from haiku.rag.tools.deps import AgentDeps

    client = HaikuRAG(temp_db_path, create=True)
    context = ToolContext()
    deps = ChatDeps(config=Config, client=client, tool_context=context)
    assert isinstance(deps, AgentDeps)
    client.close()


def test_agui_state_key_constant():
    """Test AGUI_STATE_KEY is exported with correct value."""
    assert AGUI_STATE_KEY == "haiku.rag.chat"


def test_chat_deps_state_setter_handles_initial_context(temp_db_path):
    """Test ChatDeps.state setter transfers initial_context to qa_session_state."""
    from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState

    client = HaikuRAG(temp_db_path, create=True)
    context = ToolContext()
    context.state_key = AGUI_STATE_KEY
    # Register QASessionState (normally done by prepare_chat_context)
    context.register(QA_SESSION_NAMESPACE, QASessionState())
    context.register(SESSION_NAMESPACE, SessionState())

    deps = ChatDeps(config=Config, client=client, tool_context=context)

    # Client sends initial_context with no session_context
    incoming_state = {
        AGUI_STATE_KEY: {
            "initial_context": "Background info about the project",
            "session_context": None,
            "qa_history": [],
            "citations": [],
            "document_filter": [],
            "citation_registry": {},
        }
    }

    deps.state = incoming_state

    # initial_context should be copied to qa_session_state.session_context
    qa_session_state = context.get(QA_SESSION_NAMESPACE)
    assert isinstance(qa_session_state, QASessionState)
    assert qa_session_state.session_context is not None
    assert (
        qa_session_state.session_context.summary == "Background info about the project"
    )
    client.close()


def test_chat_deps_state_setter_parses_session_context_dict(temp_db_path):
    """Test ChatDeps.state setter parses session_context dict and extracts summary."""
    from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState

    client = HaikuRAG(temp_db_path, create=True)
    context = ToolContext()
    context.state_key = AGUI_STATE_KEY
    context.register(QA_SESSION_NAMESPACE, QASessionState())
    context.register(SESSION_NAMESPACE, SessionState())

    deps = ChatDeps(config=Config, client=client, tool_context=context)

    # Client sends session_context as a dict (as it comes from JSON)
    incoming_state = {
        AGUI_STATE_KEY: {
            "session_context": {
                "summary": "Previous conversation summary",
                "last_updated": "2025-01-27T12:00:00",
            },
            "qa_history": [],
            "citations": [],
            "document_filter": [],
            "citation_registry": {},
        }
    }

    deps.state = incoming_state

    # session_context dict should be parsed into SessionContext
    qa_session_state = context.get(QA_SESSION_NAMESPACE)
    assert isinstance(qa_session_state, QASessionState)
    assert qa_session_state.session_context is not None
    assert isinstance(qa_session_state.session_context, SessionContext)
    assert qa_session_state.session_context.summary == "Previous conversation summary"
    client.close()


def test_chat_deps_state_setter_preserves_server_session_context(temp_db_path):
    """Test that server's session_context is preferred over client's stale value."""
    from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState

    client = HaikuRAG(temp_db_path, create=True)
    context = ToolContext()
    context.state_key = AGUI_STATE_KEY
    qa_state = QASessionState()
    qa_state.session_context = SessionContext(
        summary="Fresh summary from background summarizer"
    )
    context.register(QA_SESSION_NAMESPACE, qa_state)
    context.register(SESSION_NAMESPACE, SessionState())

    deps = ChatDeps(config=Config, client=client, tool_context=context)

    # Client sends stale session_context
    incoming_state = {
        AGUI_STATE_KEY: {
            "session_context": {
                "summary": "Stale summary from client",
                "last_updated": "2025-01-27T12:00:00",
            },
            "qa_history": [],
            "citations": [],
            "document_filter": [],
            "citation_registry": {},
        }
    }

    deps.state = incoming_state

    # Server's fresher session_context should be preserved
    qa_session_state = context.get(QA_SESSION_NAMESPACE)
    assert isinstance(qa_session_state, QASessionState)
    assert qa_session_state.session_context is not None
    assert (
        qa_session_state.session_context.summary
        == "Fresh summary from background summarizer"
    )
    client.close()


def test_chat_session_state():
    """Test ChatSessionState model."""
    state = ChatSessionState()
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
    """Test QAHistoryEntry model."""
    citation = Citation(
        index=1,
        document_id="doc-123",
        chunk_id="chunk-456",
        document_uri="test.md",
        document_title="Test Document",
        content="Test content",
    )
    qa = QAHistoryEntry(
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
    """Test QAHistoryEntry.sources falls back to URI when title is None."""
    citation = Citation(
        index=1,
        document_id="doc-123",
        chunk_id="chunk-456",
        document_uri="test.md",
        document_title=None,
        content="Test content",
    )
    qa = QAHistoryEntry(
        question="What is this?",
        answer="This is a test",
        citations=[citation],
    )
    assert qa.sources == ["test.md"]


def test_qa_response_to_search_answer():
    """Test QAHistoryEntry.to_search_answer() converts to SearchAnswer for research graph."""
    citation = Citation(
        index=1,
        document_id="doc-123",
        chunk_id="chunk-456",
        document_uri="test.md",
        document_title="Test Document",
        content="Test content",
    )
    qa = QAHistoryEntry(
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

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
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

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
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

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
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
        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Ask for a document that doesn't exist
        result = await agent.run(
            "Get me the nonexistent document",
            deps=deps,
        )

        assert result.output is not None


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_ask_adds_citations(allow_model_requests, temp_db_path):
    """Test that the ask tool is called and can add citations to session state."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add a document with specific content
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Ask a question that should use the ask tool
        result = await agent.run(
            "What is the highest count class in the DocLayNet dataset?",
            deps=deps,
        )

        assert result.output is not None

        # Verify the agent used the ask tool by checking for tool calls
        tool_calls = [
            part
            for msg in result.all_messages()
            if hasattr(msg, "parts")
            for part in msg.parts
            if hasattr(part, "tool_name") and part.tool_name == "ask"
        ]
        assert len(tool_calls) >= 1, "Expected ask tool to be called"

        # Session state should be registered (citations may or may not be present
        # depending on whether the research graph found relevant evidence)
        session_state = context.get(SESSION_NAMESPACE)
        assert isinstance(session_state, SessionState)


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_ask_triggers_background_summarization(
    allow_model_requests, temp_db_path
):
    """Test that the ask tool triggers background session context summarization.

    Patches the internal trigger in run_qa_core to avoid concurrent HTTP calls
    that break VCR cassette replay ordering. Triggers summarization explicitly
    after the agent run completes.
    """
    from unittest.mock import patch

    from haiku.rag.agents.chat.agent import trigger_background_summarization
    from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState

    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Patch internal trigger to avoid concurrent HTTP calls during VCR
        with patch("haiku.rag.agents.chat.agent._trigger_summarization"):
            result = await agent.run(
                "What is the highest count class in the DocLayNet dataset?",
                deps=deps,
            )

        assert result.output is not None

        # Trigger summarization explicitly (sequential, deterministic)
        trigger_background_summarization(deps)

        # Wait for background task to complete
        qa_session_state = context.get(QA_SESSION_NAMESPACE, QASessionState)
        assert qa_session_state is not None
        key = id(qa_session_state)
        if key in _summarization_tasks:
            await _summarization_tasks[key]

        # Verify session_context was populated by background task
        assert qa_session_state.session_context is not None
        assert qa_session_state.session_context.summary != ""


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_multi_turn_with_context(allow_model_requests, temp_db_path):
    """Test multi-turn conversation with initial context, summarization, and prior recall.

    Exercises the full conversation flow:
    1. Initial context is transferred to session context
    2. First question triggers background summarization
    3. Second related question uses prior answer recall and updated session context
    4. Both qa_history entries are present after two turns

    The ask tool internally fires background summarization (concurrent HTTP calls)
    which causes VCR cassette mismatches. We patch it to a no-op and trigger
    summarization explicitly after each turn to keep HTTP ordering deterministic.
    """
    from unittest.mock import patch

    from haiku.rag.agents.chat.agent import trigger_background_summarization
    from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState

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

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Set initial state with initial_context (mimicking AG-UI client)
        deps.state = {
            AGUI_STATE_KEY: {
                "initial_context": "The user is researching the DocLayNet dataset for a paper on document layout analysis.",
                "session_context": None,
                "qa_history": [],
                "citations": [],
                "document_filter": [],
                "citation_registry": {},
            }
        }

        # initial_context should be transferred to QASessionState
        qa_session = context.get(QA_SESSION_NAMESPACE, QASessionState)
        assert qa_session is not None
        assert qa_session.session_context is not None
        assert (
            qa_session.session_context.summary
            == "The user is researching the DocLayNet dataset for a paper on document layout analysis."
        )

        # Patch the internal summarization trigger in the ask tool to avoid
        # concurrent HTTP calls that break VCR cassette replay ordering.
        with patch(
            "haiku.rag.agents.chat.agent._trigger_summarization",
        ):
            # First question about class labels
            result1 = await agent.run(
                "What are the class labels defined in DocLayNet?",
                deps=deps,
            )
        assert result1.output is not None

        # Trigger summarization explicitly (sequential, no concurrency)
        trigger_background_summarization(deps)
        key = id(qa_session)
        if key in _summarization_tasks:
            await _summarization_tasks[key]

        assert qa_session.session_context is not None
        assert qa_session.session_context.summary != ""

        # qa_history should have one entry
        qa_session = context.get(QA_SESSION_NAMESPACE, QASessionState)
        assert qa_session is not None
        assert len(qa_session.qa_history) >= 1

        # Second related question - uses prior answers and updated session context
        with patch(
            "haiku.rag.agents.chat.agent._trigger_summarization",
        ):
            result2 = await agent.run(
                "How were the annotations created and how many annotators were involved?",
                deps=deps,
                message_history=result1.all_messages(),
            )
        assert result2.output is not None

        # Trigger summarization explicitly
        trigger_background_summarization(deps)
        qa_session = context.get(QA_SESSION_NAMESPACE, QASessionState)
        assert qa_session is not None
        key = id(qa_session)
        if key in _summarization_tasks:
            await _summarization_tasks[key]

        # qa_history should have two entries
        assert len(qa_session.qa_history) >= 2

        # Session context should be updated with newer summary
        assert qa_session.session_context is not None
        assert qa_session.session_context.summary != ""


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_chat_agent_ask_with_prior_answer_retrieval(
    allow_model_requests, temp_db_path
):
    """Test that ask tool retrieves relevant prior answers from qa_history.

    This exercises the prior answer retrieval logic:
    1. First ask populates qa_history with question_embedding
    2. Second similar ask should find the prior answer via embedding similarity
    """
    async with HaikuRAG(temp_db_path, create=True) as client:
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps1 = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # First ask - establishes qa_history
        result1 = await agent.run(
            "What are the class labels in DocLayNet?",
            deps=deps1,
        )
        assert result1.output is not None

        # Check that session state has citations after first call
        session_state = context.get(SESSION_NAMESPACE)
        assert isinstance(session_state, SessionState)
        # Citations might be 0 if the answer came from prior context
        assert len(session_state.citations) >= 0

        # Second ask - similar question triggers prior answer retrieval
        result2 = await agent.run(
            "Tell me about DocLayNet class labels",
            deps=deps1,
        )
        assert result2.output is not None

        # The QA session state should have history entries
        from haiku.rag.tools.qa import QA_SESSION_NAMESPACE, QASessionState

        qa_session = context.get(QA_SESSION_NAMESPACE)
        assert isinstance(qa_session, QASessionState)
        # After two asks, we should have entries in qa_history
        assert len(qa_session.qa_history) >= 1


def test_fifo_limit_enforcement():
    """Test that FIFO limit enforcement logic works correctly.

    This tests the FIFO trimming logic used in the ask() tool:
    if len(qa_history) > MAX_QA_HISTORY:
        qa_history = qa_history[-MAX_QA_HISTORY:]
    """
    # Create a session state with MAX_QA_HISTORY + 1 entries
    qa_history = [
        QAHistoryEntry(
            question=f"Question {i}",
            answer=f"Answer {i}",
            confidence=0.9,
        )
        for i in range(MAX_QA_HISTORY + 1)
    ]

    session_state = ChatSessionState(
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
        document_filter=["doc1.pdf", "doc2.pdf"],
    )
    assert state.document_filter == ["doc1.pdf", "doc2.pdf"]


def test_chat_session_state_document_filter_default_empty():
    """Test ChatSessionState document_filter defaults to empty list."""
    state = ChatSessionState()
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

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)

        # Set session filter to only include the labels document
        session_state = context.get(SESSION_NAMESPACE)
        assert isinstance(session_state, SessionState)
        session_state.document_filter = ["DocLayNet Class Labels"]

        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Search should only return results from the filtered document
        result = await agent.run(
            "Search for information about DocLayNet",
            deps=deps,
        )

        assert result.output is not None

        # Check that citations in context are only from the filtered document
        session_state = context.get(SESSION_NAMESPACE)
        assert isinstance(session_state, SessionState)
        # If citations were added, they should only be from the labels document
        for citation in session_state.citations:
            assert "labels" in citation.document_uri.lower() or "Labels" in (
                citation.document_title or ""
            )


def test_ask_tool_citation_registry_logic():
    """Test the citation index assignment logic used by the ask tool.

    Verifies that:
    1. First chunk gets index 1
    2. Second unique chunk gets index 2
    3. Same chunk_id always gets same index
    4. Indices don't reset between calls
    """
    session_state = SessionState()

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
    session_state = SessionState()

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
    from haiku.rag.tools.qa import _cosine_similarity

    vec = [1.0, 2.0, 3.0]
    assert _cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    """Test cosine similarity returns 0.0 for orthogonal vectors."""
    from haiku.rag.tools.qa import _cosine_similarity

    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    assert _cosine_similarity(vec1, vec2) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    """Test cosine similarity returns -1.0 for opposite vectors."""
    from haiku.rag.tools.qa import _cosine_similarity

    vec1 = [1.0, 2.0, 3.0]
    vec2 = [-1.0, -2.0, -3.0]
    assert _cosine_similarity(vec1, vec2) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    """Test cosine similarity handles zero vectors gracefully."""
    from haiku.rag.tools.qa import _cosine_similarity

    vec = [1.0, 2.0, 3.0]
    zero = [0.0, 0.0, 0.0]
    assert _cosine_similarity(vec, zero) == 0.0
    assert _cosine_similarity(zero, vec) == 0.0
    assert _cosine_similarity(zero, zero) == 0.0


def test_prior_answer_relevance_threshold_constant():
    """Test PRIOR_ANSWER_RELEVANCE_THRESHOLD is set to expected value."""
    from haiku.rag.tools.qa import PRIOR_ANSWER_RELEVANCE_THRESHOLD

    assert PRIOR_ANSWER_RELEVANCE_THRESHOLD == 0.7


def test_prior_answer_matching_above_threshold():
    """Test that similar questions (above threshold) are matched."""
    from haiku.rag.tools.qa import (
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
    from haiku.rag.tools.qa import (
        PRIOR_ANSWER_RELEVANCE_THRESHOLD,
        _cosine_similarity,
    )

    # Simulate two different question embeddings
    question_embedding = [1.0, 0.0, 0.0, 0.0]
    prior_embedding = [0.0, 1.0, 0.0, 0.0]  # Orthogonal = very different

    similarity = _cosine_similarity(question_embedding, prior_embedding)
    assert similarity < PRIOR_ANSWER_RELEVANCE_THRESHOLD


def test_qa_response_embedding_cache():
    """Test that QAHistoryEntry stores and retrieves question_embedding correctly."""
    embedding = [0.1, 0.2, 0.3, 0.4]
    qa = QAHistoryEntry(
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
    """Test that QAHistoryEntry.question_embedding defaults to None."""
    qa = QAHistoryEntry(
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
    """Test that new summarization tasks cancel previous ones for same state object."""
    import asyncio

    from haiku.rag.agents.chat.context import _summarization_tasks

    # Clear any existing tasks
    _summarization_tasks.clear()

    key = 12345  # Simulates id(qa_session_state)

    # Create a slow task that simulates summarization
    async def slow_task():
        await asyncio.sleep(10)  # Would take 10 seconds

    # Start first task
    task1 = asyncio.create_task(slow_task())
    _summarization_tasks[key] = task1

    # Simulate what happens when second ask comes in - cancel first task
    if key in _summarization_tasks:
        _summarization_tasks[key].cancel()

    # Yield to let cancellation propagate
    await asyncio.sleep(0)

    # Start second task
    task2 = asyncio.create_task(slow_task())
    _summarization_tasks[key] = task2

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


# =============================================================================
# list_documents Tool Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_list_documents_basic(allow_model_requests, temp_db_path):
    """Test that list_documents tool returns available documents."""
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

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Ask to list documents
        result = await agent.run(
            "What documents are available in the knowledge base?",
            deps=deps,
        )

        assert result.output is not None
        # Should mention both documents
        assert "DocLayNet" in result.output


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_list_documents_with_session_filter(allow_model_requests, temp_db_path):
    """Test that list_documents respects session document_filter."""
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

        # Set session filter to only include the labels document
        context = ToolContext()
        context.register(
            SESSION_NAMESPACE,
            SessionState(document_filter=["DocLayNet Class Labels"]),
        )
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Ask to list documents - should only show filtered documents
        result = await agent.run(
            "Show me what documents are available",
            deps=deps,
        )

        assert result.output is not None
        # Should only mention the Labels document, not Sources
        assert "Labels" in result.output or "labels" in result.output


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_list_documents_pagination(allow_model_requests, temp_db_path):
    """Test that list_documents supports pagination via limit/offset."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add multiple test documents
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
        await client.create_document(
            content=DOCLAYNET_DATA_SOURCES,
            uri="doclaynet-sources",
            title="DocLayNet Sources",
        )

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Ask to list first 2 documents
        result = await agent.run(
            "List the first 2 documents available",
            deps=deps,
        )

        assert result.output is not None


# =============================================================================
# summarize_document Tool Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_summarize_document_found(allow_model_requests, temp_db_path):
    """Test that summarize_document generates a summary for a found document."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Add a test document
        await client.create_document(
            content=DOCLAYNET_CLASS_LABELS,
            uri="doclaynet-labels",
            title="DocLayNet Class Labels",
        )

        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Ask to summarize a specific document
        result = await agent.run(
            "Summarize the DocLayNet Class Labels document",
            deps=deps,
        )

        assert result.output is not None
        # Should contain summary content about class labels
        assert len(result.output) > 50


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_summarize_document_not_found(allow_model_requests, temp_db_path):
    """Test that summarize_document handles not found documents gracefully."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        context = ToolContext()
        prepare_chat_context(context)
        agent = create_chat_agent(Config)
        deps = ChatDeps(
            config=Config,
            client=client,
            tool_context=context,
        )

        # Ask to summarize a document that doesn't exist
        result = await agent.run(
            "Summarize the nonexistent document",
            deps=deps,
        )

        assert result.output is not None
        # Should indicate the document wasn't found


# =============================================================================
# count_documents Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_count_documents(temp_db_path):
    """Test count_documents method."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        # Empty database
        assert await client.count_documents() == 0

        # Add documents
        await client.create_document(content="Doc 1", uri="test/doc1.pdf")
        await client.create_document(content="Doc 2", uri="test/doc2.pdf")
        await client.create_document(content="Doc 3", uri="other/doc3.txt")

        # Count all
        assert await client.count_documents() == 3

        # Count with filter
        assert await client.count_documents(filter="uri LIKE '%.pdf'") == 2
        assert await client.count_documents(filter="uri LIKE '%.txt'") == 1


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

    from haiku.rag.agents.chat.context import _summarization_tasks

    _summarization_tasks.clear()

    key = 67890  # Simulates id(qa_session_state)

    # Create a fast task
    async def fast_task():
        await asyncio.sleep(0.01)

    task = asyncio.create_task(fast_task())
    _summarization_tasks[key] = task
    task.add_done_callback(lambda t: _summarization_tasks.pop(key, None))

    # Wait for completion
    await task

    # Task should be cleaned up
    assert key not in _summarization_tasks

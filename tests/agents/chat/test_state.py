from pathlib import Path

import pytest

from haiku.rag.agents.chat.state import (
    MAX_QA_HISTORY,
    CitationInfo,
    QAResponse,
    _embedding_cache,
    _qa_cache_key,
    build_document_filter,
    format_conversation_context,
    rank_qa_history_by_similarity,
)
from haiku.rag.client import HaikuRAG


@pytest.fixture(scope="module")
def vcr_cassette_dir():
    return str(Path(__file__).parent.parent.parent / "cassettes" / "test_chat_state")


@pytest.mark.asyncio
async def test_rank_qa_history_empty():
    """Test empty history returns empty list."""
    # Create a mock embedder - we won't actually call it
    result = await rank_qa_history_by_similarity(
        current_question="What is this?",
        qa_history=[],
        embedder=None,  # type: ignore - won't be called for empty list
        top_k=5,
    )
    assert result == []


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_rank_qa_history_small_list(temp_db_path, allow_model_requests):
    """Test with history smaller than top_k returns all entries."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        embedder = client.chunk_repository.embedder

        # Create 3 Q&A pairs (less than top_k=5)
        qa_history = [
            QAResponse(question="What is Python?", answer="A programming language"),
            QAResponse(question="What is Java?", answer="Another programming language"),
            QAResponse(
                question="What is Rust?", answer="A systems programming language"
            ),
        ]

        result = await rank_qa_history_by_similarity(
            current_question="Tell me about Python",
            qa_history=qa_history,
            embedder=embedder,
            top_k=5,
        )

        # Should return all 3 entries since history < top_k
        assert len(result) == 3
        # All original entries should be present
        assert set(qa.question for qa in result) == {
            "What is Python?",
            "What is Java?",
            "What is Rust?",
        }


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_rank_qa_history_returns_top_k(temp_db_path, allow_model_requests):
    """Test ranking returns top-K most similar entries and populates cache."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        embedder = client.chunk_repository.embedder

        # Create 10 Q&A pairs on different topics
        qa_history = [
            QAResponse(
                question="What are the 11 class labels in DocLayNet?",
                answer="Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, and Title",
            ),
            QAResponse(
                question="How was the annotation process organized?",
                answer="The process had 4 phases with 40 dedicated annotators",
            ),
            QAResponse(
                question="What data sources were used?",
                answer="arXiv, government offices, company websites, financial reports and patents",
            ),
            QAResponse(
                question="How were pages selected?",
                answer="By selective subsampling with bias towards pages with figures or tables",
            ),
            QAResponse(
                question="What is the inter-annotator agreement?",
                answer="Computed as mAP@0.5-0.95 metric between pairwise annotations",
            ),
            QAResponse(
                question="What is machine learning?",
                answer="A field of AI that enables systems to learn from data",
            ),
            QAResponse(
                question="How does neural network training work?",
                answer="Through backpropagation and gradient descent",
            ),
            QAResponse(
                question="What is deep learning?",
                answer="A subset of ML using neural networks with many layers",
            ),
        ]

        # Clear cache to verify it gets populated
        _embedding_cache.clear()

        # Ask a question related to class labels (Q1)
        result = await rank_qa_history_by_similarity(
            current_question="Which class label has the highest count in DocLayNet?",
            qa_history=qa_history,
            embedder=embedder,
            top_k=5,
        )

        # Should return exactly 5 entries
        assert len(result) == 5

        # The class labels Q&A should be in the top 5 (it's most semantically similar)
        result_questions = [qa.question for qa in result]
        assert "What are the 11 class labels in DocLayNet?" in result_questions

        # Verify cache is populated for all Q/A pairs
        for qa in qa_history:
            cache_key = _qa_cache_key(qa.question, qa.answer)
            assert cache_key in _embedding_cache
            assert len(_embedding_cache[cache_key]) > 0


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_rank_qa_history_preserves_order(temp_db_path, allow_model_requests):
    """Test that ranking preserves original order among selected items."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        embedder = client.chunk_repository.embedder

        # Create Q&A pairs where multiple are similar
        qa_history = [
            QAResponse(
                question="What is Python?",
                answer="A programming language",
                citations=[
                    CitationInfo(
                        index=1,
                        document_id="doc1",
                        chunk_id="chunk1",
                        document_uri="python.md",
                        content="Python content",
                    )
                ],
            ),
            QAResponse(
                question="What is Java?",
                answer="Another programming language",
            ),
            QAResponse(
                question="How to use Python for data science?",
                answer="Use pandas, numpy, and scikit-learn",
            ),
        ]

        result = await rank_qa_history_by_similarity(
            current_question="Tell me about Python programming",
            qa_history=qa_history,
            embedder=embedder,
            top_k=3,
        )

        # All should be returned
        assert len(result) == 3

        # The two Python-related questions should be in the results
        result_questions = [qa.question for qa in result]
        assert "What is Python?" in result_questions
        assert "How to use Python for data science?" in result_questions


def test_format_conversation_context_empty():
    """Test format_conversation_context with empty history."""
    result = format_conversation_context([])
    assert result == ""


def test_format_conversation_context_with_history():
    """Test format_conversation_context formats qa_history as XML."""
    citation = CitationInfo(
        index=1,
        document_id="doc-123",
        chunk_id="chunk-456",
        document_uri="test.md",
        document_title="Test Document",
        content="Test content",
    )
    qa_history = [
        QAResponse(
            question="What is Python?",
            answer="A programming language",
            citations=[citation],
        ),
        QAResponse(
            question="What is Java?",
            answer="Another programming language",
        ),
    ]

    result = format_conversation_context(qa_history)

    assert "<conversation_context>" in result
    assert "previous_qa" in result
    assert "What is Python?" in result
    assert "A programming language" in result
    assert "What is Java?" in result
    assert "Another programming language" in result
    assert "Test Document" in result  # source from first citation


def test_build_document_filter_simple():
    """Test build_document_filter with simple name."""
    result = build_document_filter("mytest")
    assert "LOWER(uri) LIKE LOWER('%mytest%')" in result
    assert "LOWER(title) LIKE LOWER('%mytest%')" in result


def test_build_document_filter_with_spaces():
    """Test build_document_filter handles spaces correctly."""
    result = build_document_filter("TB MED 593")
    # Should include both the original (with spaces) and without spaces
    assert "LOWER(uri) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(uri) LIKE LOWER('%TBMED593%')" in result
    assert "LOWER(title) LIKE LOWER('%TB MED 593%')" in result
    assert "LOWER(title) LIKE LOWER('%TBMED593%')" in result


def test_build_document_filter_escapes_quotes():
    """Test build_document_filter escapes single quotes."""
    result = build_document_filter("O'Reilly")
    # Single quotes should be doubled for SQL escaping
    assert "O''Reilly" in result


def test_max_qa_history_constant():
    """Test MAX_QA_HISTORY constant value."""
    assert MAX_QA_HISTORY == 50


def test_chat_session_state_background_context():
    """Test ChatSessionState accepts background_context."""
    from haiku.rag.agents.chat.state import ChatSessionState

    state = ChatSessionState(
        session_id="test-session",
        background_context="This is background knowledge about the topic.",
    )
    assert state.background_context == "This is background knowledge about the topic."


def test_chat_session_state_background_context_defaults_to_none():
    """Test ChatSessionState background_context defaults to None."""
    from haiku.rag.agents.chat.state import ChatSessionState

    state = ChatSessionState(session_id="test-session")
    assert state.background_context is None


def test_chat_deps_state_getter_returns_namespaced_state():
    """Test ChatDeps.state getter returns state under namespaced key."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY, ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(
        session_id="test-123",
        qa_history=[
            QAResponse(question="Q1", answer="A1", confidence=0.9),
        ],
        background_context="Background info",
    )

    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    state = deps.state
    assert state is not None
    assert AGUI_STATE_KEY in state
    assert state[AGUI_STATE_KEY]["session_id"] == "test-123"
    assert len(state[AGUI_STATE_KEY]["qa_history"]) == 1
    assert state[AGUI_STATE_KEY]["qa_history"][0]["question"] == "Q1"
    assert state[AGUI_STATE_KEY]["background_context"] == "Background info"


def test_chat_deps_state_getter_without_namespace():
    """Test ChatDeps.state getter returns flat state when no state_key."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="test-123")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=None,
    )

    state = deps.state
    assert state is not None
    assert "session_id" in state
    assert state["session_id"] == "test-123"


def test_chat_deps_state_getter_returns_none_without_session():
    """Test ChatDeps.state getter returns None when no session_state."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import ChatDeps

    mock_client = MagicMock()
    mock_config = MagicMock()

    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=None,
    )

    assert deps.state is None


def test_chat_deps_state_setter_updates_from_namespaced_state():
    """Test ChatDeps.state setter updates session_state from namespaced incoming state."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY, ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="initial")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    # Simulate incoming AG-UI state with namespaced key
    incoming_state = {
        AGUI_STATE_KEY: {
            "session_id": "updated-123",
            "qa_history": [
                {"question": "Q1", "answer": "A1", "confidence": 0.9, "citations": []}
            ],
            "citations": [],
            "background_context": "New context",
        }
    }

    deps.state = incoming_state

    assert deps.session_state is not None
    assert deps.session_state.session_id == "updated-123"
    assert len(deps.session_state.qa_history) == 1
    assert deps.session_state.qa_history[0].question == "Q1"
    assert deps.session_state.background_context == "New context"


def test_chat_deps_state_setter_handles_none():
    """Test ChatDeps.state setter handles None gracefully."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="original")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
    )

    # Setting None should not raise and should not change state
    deps.state = None

    assert deps.session_state is not None
    assert deps.session_state.session_id == "original"


def test_chat_deps_state_setter_without_session_state():
    """Test ChatDeps.state setter does nothing when session_state is None."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import ChatDeps

    mock_client = MagicMock()
    mock_config = MagicMock()

    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=None,
    )

    # Should not raise even with valid incoming state
    deps.state = {"session_id": "test", "qa_history": [], "citations": []}

    assert deps.session_state is None


def test_chat_deps_state_setter_with_citation_dicts():
    """Test ChatDeps.state setter converts citation dicts to CitationInfo."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY, ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="test")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    incoming_state = {
        AGUI_STATE_KEY: {
            "session_id": "test",
            "qa_history": [],
            "citations": [
                {
                    "index": 1,
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "document_uri": "test.md",
                    "document_title": "Test Doc",
                    "page_numbers": [1, 2],
                    "headings": ["Intro"],
                    "content": "Test content",
                }
            ],
            "background_context": None,
        }
    }

    deps.state = incoming_state

    assert deps.session_state is not None
    assert len(deps.session_state.citations) == 1
    citation = deps.session_state.citations[0]
    assert citation.document_id == "doc-1"
    assert citation.chunk_id == "chunk-1"
    assert citation.page_numbers == [1, 2]


def test_chat_deps_state_getter_includes_session_context():
    """Test ChatDeps.state getter includes session_context when present."""
    from datetime import datetime
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import (
        AGUI_STATE_KEY,
        ChatDeps,
        ChatSessionState,
        SessionContext,
    )

    mock_client = MagicMock()
    mock_config = MagicMock()

    now = datetime.now()
    session_state = ChatSessionState(
        session_id="test-123",
        session_context=SessionContext(
            summary="User discussed authentication.",
            last_updated=now,
        ),
    )

    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    state = deps.state
    assert state is not None
    assert AGUI_STATE_KEY in state
    assert state[AGUI_STATE_KEY]["session_context"] is not None
    assert (
        state[AGUI_STATE_KEY]["session_context"]["summary"]
        == "User discussed authentication."
    )


def test_chat_deps_state_setter_restores_session_context():
    """Test ChatDeps.state setter restores session_context from incoming state."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import AGUI_STATE_KEY, ChatDeps, ChatSessionState

    mock_client = MagicMock()
    mock_config = MagicMock()

    session_state = ChatSessionState(session_id="initial")
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    incoming_state = {
        AGUI_STATE_KEY: {
            "session_id": "test-123",
            "qa_history": [],
            "citations": [],
            "background_context": None,
            "session_context": {
                "summary": "Restored context summary.",
                "last_updated": "2025-01-15T10:30:00",
            },
        }
    }

    deps.state = incoming_state

    assert deps.session_state is not None
    assert deps.session_state.session_context is not None
    assert deps.session_state.session_context.summary == "Restored context summary."


def test_chat_deps_state_setter_handles_null_session_context():
    """Test ChatDeps.state setter handles null session_context."""
    from unittest.mock import MagicMock

    from haiku.rag.agents.chat.state import (
        AGUI_STATE_KEY,
        ChatDeps,
        ChatSessionState,
        SessionContext,
    )

    mock_client = MagicMock()
    mock_config = MagicMock()

    # Start with a session_context
    session_state = ChatSessionState(
        session_id="test",
        session_context=SessionContext(summary="Initial summary"),
    )
    deps = ChatDeps(
        client=mock_client,
        config=mock_config,
        session_state=session_state,
        state_key=AGUI_STATE_KEY,
    )

    # Send null to clear it
    incoming_state = {
        AGUI_STATE_KEY: {
            "session_id": "test",
            "qa_history": [],
            "citations": [],
            "session_context": None,
        }
    }

    deps.state = incoming_state

    assert deps.session_state is not None
    assert deps.session_state.session_context is None

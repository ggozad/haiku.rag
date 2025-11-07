import uuid

import pytest

pytest.importorskip("fasta2a")

from fasta2a.schema import Message, TextPart  # noqa: E402
from haiku_rag_a2a.a2a import (
    extract_question_from_task,
    get_agent_skills,
    load_message_history,
    save_message_history,
)
from haiku_rag_a2a.a2a.storage import LRUMemoryStorage
from pydantic_ai.messages import (  # noqa: E402
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.messages import (
    TextPart as AITextPart,
)

from haiku.rag.client import HaikuRAG


@pytest.mark.asyncio
async def test_save_and_load_message_history():
    """Test round-trip of saving and loading message history."""
    # Create sample message history with proper part_kind for ModelRequest
    from pydantic_ai.messages import UserPromptPart

    original_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="What is Python?")]),
        ModelResponse(parts=[AITextPart(content="Python is a programming language")]),
    ]

    # Save to A2A format
    saved_message = save_message_history(original_history)

    # Verify structure
    assert saved_message["role"] == "agent"
    assert saved_message["kind"] == "message"
    assert len(saved_message["parts"]) == 1
    assert saved_message["parts"][0]["kind"] == "data"
    metadata = saved_message["parts"][0].get("metadata")
    assert metadata is not None
    assert metadata.get("type") == "conversation_state"

    # Load it back
    loaded_history = load_message_history([saved_message])

    # Verify it matches
    assert len(loaded_history) == len(original_history)
    # First message is a request with UserPromptPart
    assert isinstance(loaded_history[0], ModelRequest)
    first_part = loaded_history[0].parts[0]
    assert hasattr(first_part, "content")
    assert first_part.content == "What is Python?"  # type: ignore
    # Second message is a response with TextPart
    assert isinstance(loaded_history[1], ModelResponse)
    second_part = loaded_history[1].parts[0]
    assert hasattr(second_part, "content")
    assert second_part.content == "Python is a programming language"  # type: ignore


@pytest.mark.asyncio
async def test_save_and_load_message_history_with_tool_calls():
    """Test saving and loading message history that includes tool calls."""
    from pydantic_ai.messages import UserPromptPart

    original_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="Search for Python")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="search_documents",
                    args={"query": "Python", "limit": 3},
                    tool_call_id="call_1",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="search_documents",
                    content="Python is a high-level programming language",
                    tool_call_id="call_1",
                )
            ]
        ),
        ModelResponse(
            parts=[AITextPart(content="Based on the search, Python is a language")]
        ),
    ]

    # Save and load
    saved_message = save_message_history(original_history)
    loaded_history = load_message_history([saved_message])

    # Verify tool calls are preserved
    assert len(loaded_history) == 4
    assert isinstance(loaded_history[1].parts[0], ToolCallPart)
    assert loaded_history[1].parts[0].tool_name == "search_documents"
    assert isinstance(loaded_history[2].parts[0], ToolReturnPart)
    assert loaded_history[2].parts[0].tool_name == "search_documents"


@pytest.mark.asyncio
async def test_extract_question_from_task():
    """Test extracting user question from task history."""
    task_history: list[Message] = [
        Message(
            role="user",
            parts=[TextPart(kind="text", text="What is Python?")],
            kind="message",
            message_id=str(uuid.uuid4()),
        )
    ]

    question = extract_question_from_task(task_history)
    assert question == "What is Python?"


@pytest.mark.asyncio
async def test_extract_question_from_task_no_text():
    """Test extracting question when no text part exists."""
    task_history: list[Message] = [
        Message(
            role="user",
            parts=[],
            kind="message",
            message_id=str(uuid.uuid4()),
        )
    ]

    question = extract_question_from_task(task_history)
    assert question is None


@pytest.mark.asyncio
async def test_lru_memory_storage_lru_eviction():
    """Test that LRUMemoryStorage evicts least recently used contexts."""
    from fasta2a.storage import InMemoryStorage

    base_storage = InMemoryStorage()
    storage = LRUMemoryStorage(storage=base_storage, max_contexts=3)

    # Add 3 contexts (at limit)
    await storage.update_context("ctx1", [])
    await storage.update_context("ctx2", [])
    await storage.update_context("ctx3", [])

    # All 3 should be tracked
    assert len(storage.context_order) == 3
    assert "ctx1" in storage.context_order
    assert "ctx2" in storage.context_order
    assert "ctx3" in storage.context_order

    # Add 4th context - should evict ctx1 (oldest)
    await storage.update_context("ctx4", [])
    assert len(storage.context_order) == 3
    assert "ctx1" not in storage.context_order
    assert "ctx2" in storage.context_order
    assert "ctx3" in storage.context_order
    assert "ctx4" in storage.context_order

    # Access ctx2 (moves it to end)
    await storage.load_context("ctx2")

    # Add 5th context - should evict ctx3 (now oldest since ctx2 was accessed)
    await storage.update_context("ctx5", [])
    assert len(storage.context_order) == 3
    assert "ctx3" not in storage.context_order
    assert "ctx2" in storage.context_order  # Still present (was accessed)
    assert "ctx4" in storage.context_order
    assert "ctx5" in storage.context_order


@pytest.mark.asyncio
async def test_lru_memory_storage_access_order():
    """Test that accessing contexts updates their order."""
    from fasta2a.storage import InMemoryStorage

    base_storage = InMemoryStorage()
    storage = LRUMemoryStorage(storage=base_storage, max_contexts=2)

    # Add 2 contexts
    await storage.update_context("ctx1", [])
    await storage.update_context("ctx2", [])

    # Order should be: ctx1, ctx2
    assert list(storage.context_order.keys()) == ["ctx1", "ctx2"]

    # Load ctx1 (moves to end)
    await storage.load_context("ctx1")
    # Order should be: ctx2, ctx1
    assert list(storage.context_order.keys()) == ["ctx2", "ctx1"]

    # Add ctx3 - should evict ctx2 (oldest)
    await storage.update_context("ctx3", [])
    assert "ctx2" not in storage.context_order
    assert "ctx1" in storage.context_order
    assert "ctx3" in storage.context_order


@pytest.mark.asyncio
async def test_a2a_app_creation(temp_db_path):
    """Test that A2A app can be created successfully."""
    from haiku_rag_a2a.a2a import create_a2a_app

    # Create a test database
    async with HaikuRAG(temp_db_path) as client:
        await client.create_document(
            content="Python is a high-level programming language known for its simplicity.",
            uri="python_doc",
        )

    # Create A2A app
    app = create_a2a_app(temp_db_path)

    # Verify app properties
    assert app.name == "haiku-rag"
    assert app.description is not None
    assert "conversational" in app.description.lower()


@pytest.mark.asyncio
async def test_a2a_app_has_skills(temp_db_path):
    """Test that A2A app exposes skills describing its capabilities."""
    from haiku_rag_a2a.a2a import create_a2a_app

    # Create a test database
    async with HaikuRAG(temp_db_path) as client:
        await client.create_document(content="Test document", uri="test_doc")

    # Create A2A app
    app = create_a2a_app(temp_db_path)

    # Verify app has skills
    assert app.skills is not None
    assert len(app.skills) > 0

    # Check that at least one skill exists
    skill = app.skills[0]
    assert "id" in skill
    assert "name" in skill
    assert "description" in skill
    assert "tags" in skill
    assert "input_modes" in skill
    assert "output_modes" in skill

    # Verify the skill describes document search/QA capabilities
    skill_text = f"{skill['name']} {skill['description']}".lower()
    assert any(
        keyword in skill_text
        for keyword in ["search", "question", "answer", "document", "knowledge"]
    )


def test_get_agent_skills():
    """Test that agent skills include all three skills."""
    skills = get_agent_skills()

    assert len(skills) == 3

    skill_ids = [skill["id"] for skill in skills]
    assert "document-qa" in skill_ids
    assert "document-search" in skill_ids
    assert "document-retrieve" in skill_ids

    # Check document-qa skill
    doc_qa = next(s for s in skills if s["id"] == "document-qa")
    assert "Document Question Answering" in doc_qa["name"]
    assert "semantic search" in doc_qa["description"]
    assert "question-answering" in doc_qa["tags"]

    # Check document-search skill
    doc_search = next(s for s in skills if s["id"] == "document-search")
    assert "Document Search" in doc_search["name"]
    assert "search" in doc_search["tags"]

    # Check document-retrieve skill
    doc_retrieve = next(s for s in skills if s["id"] == "document-retrieve")
    assert "Document Retrieval" in doc_retrieve["name"]
    assert "retrieval" in doc_retrieve["tags"]


@pytest.mark.asyncio
async def test_build_artifacts_for_search():
    """Test that search operations produce structured search artifacts."""
    from haiku_rag_a2a.a2a.worker import ConversationalWorker
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )

    class MockResult:
        output = "Found 1 relevant results:\n\n1. *Score: 0.9* | **test**\nresult"

        def new_messages(self):
            return [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="search_documents",
                            args={"query": "test", "limit": 3},
                            tool_call_id="call_1",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="search_documents",
                            content=[{"content": "result", "score": 0.9}],
                            tool_call_id="call_1",
                        )
                    ]
                ),
            ]

    from pathlib import Path

    from fasta2a.broker import InMemoryBroker
    from fasta2a.storage import InMemoryStorage

    worker = ConversationalWorker(
        storage=InMemoryStorage(),
        broker=InMemoryBroker(),
        db_path=Path("/tmp/test.db"),
        agent=None,  # type: ignore
    )

    artifacts = worker.build_artifacts(MockResult(), "search", "test query")

    assert len(artifacts) == 1
    assert artifacts[0].get("name") == "search_results"
    assert len(artifacts[0]["parts"]) == 1
    assert artifacts[0]["parts"][0]["kind"] == "data"
    assert "results" in artifacts[0]["parts"][0]["data"]
    assert "query" in artifacts[0]["parts"][0]["data"]


@pytest.mark.asyncio
async def test_build_artifacts_for_retrieve():
    """Test that retrieve operations produce document artifacts."""
    from haiku_rag_a2a.a2a.worker import ConversationalWorker
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )

    class MockResult:
        output = "Document content"

        def new_messages(self):
            return [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="get_full_document",
                            args={"document_uri": "test.txt"},
                            tool_call_id="call_1",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="get_full_document",
                            content="Full document content here",
                            tool_call_id="call_1",
                        )
                    ]
                ),
            ]

    from pathlib import Path

    from fasta2a.broker import InMemoryBroker
    from fasta2a.storage import InMemoryStorage

    worker = ConversationalWorker(
        storage=InMemoryStorage(),
        broker=InMemoryBroker(),
        db_path=Path("/tmp/test.db"),
        agent=None,  # type: ignore
    )

    artifacts = worker.build_artifacts(MockResult(), "retrieve", "test query")

    assert len(artifacts) == 1
    assert artifacts[0].get("name") == "document"
    assert artifacts[0]["parts"][0]["kind"] == "text"
    assert artifacts[0]["parts"][0]["text"] == "Full document content here"


@pytest.mark.asyncio
async def test_build_artifacts_for_multiple_searches():
    """Test that multiple searches each get their own artifact with correct results."""
    from haiku_rag_a2a.a2a.worker import ConversationalWorker
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.messages import TextPart as AITextPart

    class MockResult:
        output = "Answer based on multiple searches"

        def new_messages(self):
            return [
                # First search
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="search_documents",
                            args={"query": "first query", "limit": 2},
                            tool_call_id="call_1",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="search_documents",
                            content=[
                                {"content": "result 1", "score": 0.9},
                                {"content": "result 2", "score": 0.8},
                            ],
                            tool_call_id="call_1",
                        )
                    ]
                ),
                # Second search
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="search_documents",
                            args={"query": "second query", "limit": 2},
                            tool_call_id="call_2",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="search_documents",
                            content=[
                                {"content": "result 3", "score": 0.7},
                                {"content": "result 4", "score": 0.6},
                            ],
                            tool_call_id="call_2",
                        )
                    ]
                ),
                ModelResponse(
                    parts=[AITextPart(content="Answer based on multiple searches")]
                ),
            ]

    from pathlib import Path

    from fasta2a.broker import InMemoryBroker
    from fasta2a.storage import InMemoryStorage

    worker = ConversationalWorker(
        storage=InMemoryStorage(),
        broker=InMemoryBroker(),
        db_path=Path("/tmp/test.db"),
        agent=None,  # type: ignore
    )

    artifacts = worker.build_artifacts(MockResult(), "qa", "What is the answer?")

    # Should have 2 search artifacts + 1 qa_result artifact
    assert len(artifacts) == 3

    # First search artifact
    assert artifacts[0].get("name") == "search_results"
    part_0 = artifacts[0]["parts"][0]
    assert part_0.get("data", {}).get("query") == "first query"
    results_1 = part_0.get("data", {}).get("results", [])
    assert len(results_1) == 2
    assert results_1[0]["content"] == "result 1"
    assert results_1[1]["content"] == "result 2"

    # Second search artifact
    assert artifacts[1].get("name") == "search_results"
    part_1 = artifacts[1]["parts"][0]
    assert part_1.get("data", {}).get("query") == "second query"
    results_2 = part_1.get("data", {}).get("results", [])
    assert len(results_2) == 2
    assert results_2[0]["content"] == "result 3"
    assert results_2[1]["content"] == "result 4"

    # Q&A artifact
    assert artifacts[2].get("name") == "qa_result"


@pytest.mark.asyncio
async def test_qa_artifact_for_conversational_messages():
    """Test that conversational Q&A messages always create qa_result artifacts."""
    from haiku_rag_a2a.a2a.worker import ConversationalWorker
    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.messages import TextPart as AITextPart

    class MockResult:
        output = "Hello! How can I help you?"

        def new_messages(self):
            # No tool calls, just a conversational response
            return [
                ModelResponse(parts=[AITextPart(content="Hello! How can I help you?")]),
            ]

    from pathlib import Path

    from fasta2a.broker import InMemoryBroker
    from fasta2a.storage import InMemoryStorage

    worker = ConversationalWorker(
        storage=InMemoryStorage(),
        broker=InMemoryBroker(),
        db_path=Path("/tmp/test.db"),
        agent=None,  # type: ignore
    )

    artifacts = worker.build_artifacts(MockResult(), "qa", "Hello")

    # Should have qa_result artifact (even without tools, for A2A traceability)
    assert len(artifacts) == 1
    assert artifacts[0].get("name") == "qa_result"
    part = artifacts[0]["parts"][0]
    assert part.get("data", {}).get("question") == "Hello"
    assert part.get("data", {}).get("answer") == "Hello! How can I help you?"


@pytest.mark.asyncio
async def test_build_artifacts_for_qa():
    """Test that Q&A operations produce artifacts for each tool call."""
    from haiku_rag_a2a.a2a.worker import ConversationalWorker
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.messages import TextPart as AITextPart

    class MockResult:
        output = "This is the answer"

        def new_messages(self):
            # Multiple tool calls indicates Q&A workflow
            return [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="search_documents",
                            args={"query": "test", "limit": 3},
                            tool_call_id="call_1",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="search_documents",
                            content=[{"content": "result", "score": 0.9}],
                            tool_call_id="call_1",
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="get_full_document",
                            args={"document_uri": "test.txt"},
                            tool_call_id="call_2",
                        )
                    ]
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="get_full_document",
                            content="Full content",
                            tool_call_id="call_2",
                        )
                    ]
                ),
                ModelResponse(parts=[AITextPart(content="This is the answer")]),
            ]

    from pathlib import Path

    from fasta2a.broker import InMemoryBroker
    from fasta2a.storage import InMemoryStorage

    worker = ConversationalWorker(
        storage=InMemoryStorage(),
        broker=InMemoryBroker(),
        db_path=Path("/tmp/test.db"),
        agent=None,  # type: ignore
    )

    artifacts = worker.build_artifacts(MockResult(), "qa", "What is Python?")

    # Q&A should produce artifacts for each tool call (search + retrieve) + final Q&A artifact
    assert len(artifacts) == 3

    # First artifact is from search_documents
    assert artifacts[0].get("name") == "search_results"
    assert artifacts[0]["parts"][0]["kind"] == "data"
    assert "results" in artifacts[0]["parts"][0]["data"]
    assert artifacts[0]["parts"][0]["data"]["query"] == "test"

    # Second artifact is from get_full_document
    assert artifacts[1].get("name") == "document"
    assert artifacts[1]["parts"][0]["kind"] == "text"

    # Third artifact is the Q&A result
    assert artifacts[2].get("name") == "qa_result"
    assert artifacts[2]["parts"][0]["kind"] == "data"
    assert artifacts[2]["parts"][0]["data"]["question"] == "What is Python?"
    assert artifacts[2]["parts"][0]["data"]["answer"] == "This is the answer"
    assert artifacts[2]["parts"][0]["data"]["skill"] == "document-qa"

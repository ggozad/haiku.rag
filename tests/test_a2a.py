import uuid

import pytest

from haiku.rag.a2a import (
    extract_question_from_task,
    extract_skill_preference,
    get_agent_skills,
    load_message_history,
    save_message_history,
)
from haiku.rag.client import HaikuRAG

pytest.importorskip("fasta2a")

from fasta2a.schema import Message, TextPart  # noqa: E402
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

    from haiku.rag.a2a import LRUMemoryStorage

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

    from haiku.rag.a2a import LRUMemoryStorage

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
    from haiku.rag.a2a import create_a2a_app

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
    from haiku.rag.a2a import create_a2a_app

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
    """Test that agent skills include both document-qa and deep-qa."""
    skills = get_agent_skills()

    assert len(skills) == 2

    skill_ids = [skill["id"] for skill in skills]
    assert "document-qa" in skill_ids
    assert "deep-qa" in skill_ids

    # Check document-qa skill
    doc_qa = next(s for s in skills if s["id"] == "document-qa")
    assert "Document Question Answering" in doc_qa["name"]
    assert "semantic search" in doc_qa["description"]
    assert "question-answering" in doc_qa["tags"]

    # Check deep-qa skill
    deep_qa = next(s for s in skills if s["id"] == "deep-qa")
    assert "Deep Question Answering" in deep_qa["name"]
    assert "Multi-step" in deep_qa["description"]
    assert "research" in deep_qa["tags"]


@pytest.mark.asyncio
async def test_extract_skill_preference_with_metadata():
    """Test extracting skill preference from message metadata."""
    from fasta2a.schema import DataPart

    task_history: list[Message] = [
        Message(
            role="user",
            parts=[
                TextPart(kind="text", text="Complex question"),
                DataPart(
                    kind="data",
                    data={"skill": "deep-qa"},
                    metadata={"type": "skill_preference"},
                ),
            ],
            kind="message",
            message_id=str(uuid.uuid4()),
        )
    ]

    skill = extract_skill_preference(task_history)
    assert skill == "deep-qa"


@pytest.mark.asyncio
async def test_extract_skill_preference_default():
    """Test that skill preference defaults to document-qa."""
    task_history: list[Message] = [
        Message(
            role="user",
            parts=[TextPart(kind="text", text="What is Python?")],
            kind="message",
            message_id=str(uuid.uuid4()),
        )
    ]

    skill = extract_skill_preference(task_history)
    assert skill == "document-qa"


@pytest.mark.asyncio
async def test_extract_skill_preference_no_skill_in_data():
    """Test skill preference when DataPart exists but has no skill."""
    from fasta2a.schema import DataPart

    task_history: list[Message] = [
        Message(
            role="user",
            parts=[
                TextPart(kind="text", text="Question"),
                DataPart(
                    kind="data",
                    data={"other": "value"},
                    metadata={"type": "skill_preference"},
                ),
            ],
            kind="message",
            message_id=str(uuid.uuid4()),
        )
    ]

    skill = extract_skill_preference(task_history)
    assert skill == "document-qa"

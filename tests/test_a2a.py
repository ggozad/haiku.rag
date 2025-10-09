import uuid

import pytest

from haiku.rag.a2a import (
    extract_question_from_task,
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

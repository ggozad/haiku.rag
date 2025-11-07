import uuid

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage
from pydantic_core import to_jsonable_python

try:
    from fasta2a.schema import DataPart, Message  # type: ignore
except ImportError as e:
    raise ImportError(
        "A2A support requires the 'a2a' extra. "
        "Install with: uv pip install 'haiku.rag[a2a]'"
    ) from e

ModelMessagesTypeAdapter = TypeAdapter(list[ModelMessage])


def load_message_history(context: list[Message]) -> list[ModelMessage]:
    """Load pydantic-ai message history from A2A context.

    The context stores serialized pydantic-ai message history directly,
    which we deserialize and return.

    Args:
        context: A2A context messages

    Returns:
        List of pydantic-ai ModelMessage objects
    """
    if not context:
        return []

    # Context should contain a single "state" message with full history
    for msg in context:
        parts = msg.get("parts", [])
        for part in parts:
            if part.get("kind") == "data":
                metadata = part.get("metadata", {})
                if metadata.get("type") == "conversation_state":
                    stored_history = part.get("data", {}).get("message_history", [])
                    if stored_history:
                        return ModelMessagesTypeAdapter.validate_python(stored_history)

    return []


def save_message_history(message_history: list[ModelMessage]) -> Message:
    """Save pydantic-ai message history to A2A context format.

    Args:
        message_history: Full pydantic-ai message history

    Returns:
        A2A Message containing the serialized state (stored as agent role)
    """
    serialized = to_jsonable_python(message_history)
    return Message(
        role="agent",
        parts=[
            DataPart(
                kind="data",
                data={"message_history": serialized},
                metadata={"type": "conversation_state"},
            )
        ],
        kind="message",
        message_id=str(uuid.uuid4()),
    )

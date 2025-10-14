try:
    from fasta2a.schema import Message, Skill  # type: ignore
except ImportError as e:
    raise ImportError(
        "A2A support requires the 'a2a' extra. "
        "Install with: uv pip install 'haiku.rag[a2a]'"
    ) from e


def get_agent_skills() -> list[Skill]:
    """Define the skills exposed by the haiku.rag A2A agent.

    Returns:
        List of skills describing the agent's capabilities
    """
    return [
        Skill(
            id="document-qa",
            name="Document Question Answering",
            description="Answer questions based on a knowledge base of documents using semantic search and retrieval",
            tags=["question-answering", "search", "knowledge-base", "rag"],
            input_modes=["application/json"],
            output_modes=["application/json"],
            examples=[
                "What does the documentation say about authentication?",
                "Find information about Python best practices",
                "Show me the full API documentation",
            ],
        ),
        Skill(
            id="document-search",
            name="Document Search",
            description="Search for relevant document chunks in the knowledge base using hybrid (semantic and BM25) search",
            tags=["search", "retrieval", "semantic-search"],
            input_modes=["application/json"],
            output_modes=["application/json"],
            examples=[
                "Search for Python best practices",
                "Find documents about authentication",
                "Look for API documentation",
            ],
        ),
        Skill(
            id="document-retrieve",
            name="Document Retrieval",
            description="Retrieve the complete content of a specific document by its URI",
            tags=["retrieval", "fetch", "document"],
            input_modes=["application/json"],
            output_modes=["application/json"],
            examples=[
                "Get the full content of document X",
                "Retrieve document by URI",
                "Show me the complete document",
            ],
        ),
    ]


def extract_question_from_task(task_history: list[Message]) -> str | None:
    """Extract the user's question from task history.

    Args:
        task_history: Task history messages

    Returns:
        The question text if found, None otherwise
    """
    for msg in task_history:
        if msg.get("role") == "user":
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    text = part.get("text", "").strip()
                    if text:
                        return text
    return None

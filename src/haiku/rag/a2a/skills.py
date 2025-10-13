"""A2A skill definitions and utilities."""

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
            id="deep-qa",
            name="Deep Question Answering",
            description="Multi-step question decomposition and research for complex queries (can take a long time)",
            tags=["question-answering", "research", "multi-agent", "complex-queries"],
            input_modes=["application/json"],
            output_modes=["application/json"],
            examples=[
                "What are the architectural patterns used in haiku.rag and how do they compare?",
                "Analyze the trade-offs between the simple QA and research agents",
                "What are all the configuration options and their effects?",
            ],
        ),
    ]


def extract_skill_preference(task_history: list[Message]) -> str:
    """Extract skill preference from task history metadata.

    Args:
        task_history: Task history messages

    Returns:
        Skill ID if found in metadata, otherwise "document-qa" (default)
    """
    for msg in task_history:
        if msg.get("role") == "user":
            for part in msg.get("parts", []):
                if part.get("kind") == "data":
                    metadata = part.get("metadata", {})
                    if metadata.get("type") == "skill_preference":
                        skill = part.get("data", {}).get("skill")
                        if skill:
                            return skill
    return "document-qa"


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

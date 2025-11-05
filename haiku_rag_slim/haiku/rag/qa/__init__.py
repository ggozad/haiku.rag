from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.qa.agent import QuestionAnswerAgent


def get_qa_agent(
    client: HaikuRAG,
    config: AppConfig = Config,
    use_citations: bool = False,
    system_prompt: str | None = None,
) -> QuestionAnswerAgent:
    """
    Factory function to get a QA agent based on the configuration.

    Args:
        client: HaikuRAG client instance.
        config: Configuration to use. Defaults to global Config.
        use_citations: Whether to include citations in responses.
        system_prompt: Optional custom system prompt.

    Returns:
        A configured QuestionAnswerAgent instance.
    """
    provider = config.qa.provider
    model_name = config.qa.model

    return QuestionAnswerAgent(
        client=client,
        provider=provider,
        model=model_name,
        use_citations=use_citations,
        system_prompt=system_prompt,
    )

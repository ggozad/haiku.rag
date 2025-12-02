from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.qa.agent import QuestionAnswerAgent


def get_qa_agent(
    client: HaikuRAG,
    config: AppConfig = Config,
    system_prompt: str | None = None,
) -> QuestionAnswerAgent:
    """Factory function to get a QA agent based on the configuration.

    Args:
        client: HaikuRAG client instance.
        config: Configuration to use. Defaults to global Config.
        system_prompt: Optional custom system prompt.

    Returns:
        A configured QuestionAnswerAgent instance.
    """
    return QuestionAnswerAgent(
        client=client,
        model_config=config.qa.model,
        system_prompt=system_prompt,
    )

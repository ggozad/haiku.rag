from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.qa.agent import QuestionAnswerAgent
from haiku.rag.qa.prompts import QA_SYSTEM_PROMPT
from haiku.rag.utils import build_prompt


def get_qa_agent(
    client: HaikuRAG,
    config: AppConfig = Config,
    system_prompt: str | None = None,
) -> QuestionAnswerAgent:
    """Factory function to get a QA agent based on the configuration.

    Args:
        client: HaikuRAG client instance.
        config: Configuration to use. Defaults to global Config.
        system_prompt: Optional custom system prompt (overrides config).

    Returns:
        A configured QuestionAnswerAgent instance.
    """
    # Determine the base prompt: explicit > config > default
    if system_prompt is None:
        system_prompt = config.prompts.qa or QA_SYSTEM_PROMPT

    # Prepend system_context if configured
    system_prompt = build_prompt(system_prompt, config)

    return QuestionAnswerAgent(
        client=client,
        model_config=config.qa.model,
        config=config,
        system_prompt=system_prompt,
    )

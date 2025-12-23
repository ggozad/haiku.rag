from haiku.rag.client import HaikuRAG
from haiku.rag.config import AppConfig, Config
from haiku.rag.qa.agent import QuestionAnswerAgent
from haiku.rag.qa.prompts import QA_SYSTEM_PROMPT, QA_SYSTEM_PROMPT_WITH_RAPTOR
from haiku.rag.utils import build_prompt


async def get_qa_agent(
    client: HaikuRAG,
    config: AppConfig = Config,
    system_prompt: str | None = None,
) -> QuestionAnswerAgent:
    """Factory function to get a QA agent based on the configuration.

    Args:
        client: HaikuRAG client instance.
        config: Configuration to use. Defaults to global Config.
        system_prompt: Optional custom system prompt (overrides config).
            If not provided, uses config.prompts.qa or RAPTOR-aware default.

    Returns:
        A configured QuestionAnswerAgent instance.
    """
    # Determine the base prompt: explicit > config > RAPTOR-aware default
    if system_prompt is None:
        if config.prompts.qa:
            system_prompt = config.prompts.qa
        else:
            has_raptor = await client.raptor_repository.has_nodes()
            system_prompt = QA_SYSTEM_PROMPT_WITH_RAPTOR if has_raptor else QA_SYSTEM_PROMPT

    # Prepend system_context if configured
    system_prompt = build_prompt(system_prompt, config)

    return QuestionAnswerAgent(
        client=client,
        model_config=config.qa.model,
        config=config,
        system_prompt=system_prompt,
    )

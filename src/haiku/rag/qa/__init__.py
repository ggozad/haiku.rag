from haiku.rag.client import HaikuRAG
from haiku.rag.config import Config
from haiku.rag.qa.base import QuestionAnswerAgentBase
from haiku.rag.qa.ollama import QuestionAnswerOllamaAgent


def get_qa_agent(
    client: HaikuRAG, model: str = "", use_citations: bool = False
) -> QuestionAnswerAgentBase:
    """
    Factory function to get the appropriate QA agent based on the configuration.
    """
    if Config.QA_PROVIDER == "ollama":
        return QuestionAnswerOllamaAgent(
            client, model or Config.QA_MODEL, use_citations
        )

    if Config.QA_PROVIDER == "openai":
        try:
            from haiku.rag.qa.openai import QuestionAnswerOpenAIAgent
        except ImportError:
            raise ImportError(
                "OpenAI QA agent requires the 'openai' package. "
                "Please install haiku.rag with the 'openai' extra:"
                "uv pip install haiku.rag[openai]"
            )
        return QuestionAnswerOpenAIAgent(
            client, model or Config.QA_MODEL, use_citations
        )

    if Config.QA_PROVIDER == "anthropic":
        try:
            from haiku.rag.qa.anthropic import QuestionAnswerAnthropicAgent
        except ImportError:
            raise ImportError(
                "Anthropic QA agent requires the 'anthropic' package. "
                "Please install haiku.rag with the 'anthropic' extra:"
                "uv pip install haiku.rag[anthropic]"
            )
        return QuestionAnswerAnthropicAgent(
            client, model or Config.QA_MODEL, use_citations
        )

    raise ValueError(f"Unsupported QA provider: {Config.QA_PROVIDER}")

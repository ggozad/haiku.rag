"""Common utilities for all graph implementations."""

from typing import Any, Protocol

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import Config


class HasEmitLog(Protocol):
    """Protocol for objects that can emit log messages."""

    def emit_log(self, message: str, state: Any = None) -> None: ...


def get_model(provider: str, model: str) -> OpenAIChatModel | str:
    """
    Get a model instance for the specified provider and model name.

    Args:
        provider: The model provider ("ollama", "vllm", or other)
        model: The model name

    Returns:
        A configured model instance

    Raises:
        ValueError: If the provider is unknown
    """
    if provider == "ollama":
        return OpenAIChatModel(
            model_name=model,
            provider=OllamaProvider(base_url=f"{Config.providers.ollama.base_url}/v1"),
        )
    elif provider == "vllm":
        return OpenAIChatModel(
            model_name=model,
            provider=OpenAIProvider(
                base_url=f"{Config.providers.vllm.research_base_url or Config.providers.vllm.qa_base_url}/v1",
                api_key="none",
            ),
        )
    elif provider in ("openai", "anthropic", "gemini", "groq", "bedrock"):
        # These providers use string format
        return f"{provider}:{model}"
    else:
        raise ValueError(
            f"Unknown model provider: {provider}. "
            f"Supported providers: ollama, vllm, openai, anthropic, gemini, groq, bedrock"
        )


def log(deps: HasEmitLog, state: Any, message: str) -> None:
    """
    Emit a log message through the dependencies.

    Args:
        deps: Dependencies object with emit_log method
        state: Current state (passed to emit_log)
        message: The message to log
    """
    deps.emit_log(message, state)

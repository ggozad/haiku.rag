from typing import Any, Protocol

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import Config


class HasEmitLog(Protocol):
    def emit_log(self, message: str, state: Any = None) -> None: ...


def get_model(provider: str, model: str) -> Any:
    if provider == "ollama":
        return OpenAIChatModel(
            model_name=model,
            provider=OllamaProvider(base_url=f"{Config.OLLAMA_BASE_URL}/v1"),
        )
    elif provider == "vllm":
        return OpenAIChatModel(
            model_name=model,
            provider=OpenAIProvider(
                base_url=f"{Config.VLLM_RESEARCH_BASE_URL or Config.VLLM_QA_BASE_URL}/v1",
                api_key="none",
            ),
        )
    else:
        return f"{provider}:{model}"


def log(deps: HasEmitLog, state: Any, message: str) -> None:
    deps.emit_log(message, state)

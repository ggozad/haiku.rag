from typing import Any, Protocol

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import Config
from haiku.rag.qa.deep.models import SearchAnswer


class HasEmitLog(Protocol):
    def emit_log(self, message: str, state: Any = None) -> None: ...


def get_model(provider: str, model: str) -> Any:
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
    else:
        return f"{provider}:{model}"


def log(deps: HasEmitLog, state: Any, message: str) -> None:
    deps.emit_log(message, state)


def collect_answers_reducer(
    acc: list[SearchAnswer], item: SearchAnswer | None
) -> list[SearchAnswer]:
    """Reducer function to collect search answers, filtering out None values."""
    return acc + [item] if item else acc

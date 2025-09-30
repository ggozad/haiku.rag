from typing import TYPE_CHECKING, Any

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from haiku.rag.config import Config

if TYPE_CHECKING:  # pragma: no cover
    from haiku.rag.research.state import ResearchDeps, ResearchState


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


def log(deps: "ResearchDeps", state: "ResearchState", msg: str) -> None:
    deps.emit_log(msg, state)

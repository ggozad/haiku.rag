import logging
from typing import TYPE_CHECKING, Any, Literal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai.models.openai import OpenAIChatModelSettings
    from pydantic_ai.profiles.openai import OpenAIModelProfile

    from haiku.rag.config.models import (
        AppConfig,
        EmbeddingModelConfig,
        ModelConfig,
        ThinkingEffort,
    )


_OPENAI_COMPAT_PROFILE: "OpenAIModelProfile" = {
    "openai_chat_supports_multiple_system_messages": False
}


def parse_model_option(value: str) -> "ModelConfig":
    """Parse a 'provider:name' string into a ModelConfig."""
    from haiku.rag.config.models import ModelConfig

    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid model format '{value}'. Expected 'provider:name' (e.g. 'ollama:qwen3.8')."
        )
    return ModelConfig(provider=parts[0], name=parts[1])


def check_api_key_supported(
    model_config: "ModelConfig | EmbeddingModelConfig", supported: set[str]
) -> None:
    """Reject a configured api_key on a provider whose client we never build."""
    if model_config.api_key and model_config.provider not in supported:
        raise ValueError(
            f"api_key is not supported on the '{model_config.provider}' provider "
            f"(supported: {', '.join(sorted(supported))}). Set that provider's "
            "own API key environment variable instead."
        )


def vllm_base_url(base_url: str | None) -> str:
    """Normalize a vLLM endpoint to its OpenAI-compatible `/v1` root."""
    base_url = base_url or "http://localhost:8000/v1"
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return base_url


def _check_provider_known(provider: str) -> None:
    """Reject a chat provider pydantic-ai cannot resolve."""
    from pydantic_ai.providers import infer_provider_class

    try:
        infer_provider_class(provider)
    except ImportError:
        # pydantic-ai knows the name, its SDK is just not installed here. That
        # failure names the extra to install, so leave it to be raised in place.
        return
    except ValueError:
        raise ValueError(
            f"Unknown model provider '{provider}'. See "
            "https://ai.pydantic.dev/models/ for the providers pydantic-ai "
            "supports. vLLM uses provider 'vllm'; another "
            "OpenAI-compatible server (sglang, LM Studio) uses provider "
            "'openai' with base_url."
        ) from None


def apply_common_settings(
    settings: Any | None,
    model_config: Any,
    *,
    map_thinking: bool = True,
) -> Any | None:
    """Apply the settings every provider shares onto a model settings dict."""
    thinking = model_config.thinking if map_thinking else None

    if (
        model_config.temperature is None
        and model_config.max_tokens is None
        and model_config.extra_body is None
        and thinking is None
    ):
        return settings

    settings_dict = {} if settings is None else settings

    if model_config.temperature is not None:
        settings_dict["temperature"] = model_config.temperature

    if model_config.max_tokens is not None:
        settings_dict["max_tokens"] = model_config.max_tokens

    if model_config.extra_body is not None:
        settings_dict["extra_body"] = model_config.extra_body

    if thinking is not None:
        settings_dict["thinking"] = thinking

    return settings_dict


def reasoning_effort(
    model_config: "ModelConfig",
) -> "ThinkingEffort | Literal['none'] | None":
    """OpenAI `reasoning_effort` for a model config, or None when unset."""
    thinking = model_config.thinking
    if thinking is None:
        return None
    if isinstance(thinking, str):
        return thinking
    return "medium" if thinking else "none"


def reasoning_effort_settings(
    model_config: "ModelConfig",
) -> "OpenAIChatModelSettings | None":
    """Settings carrying `reasoning_effort` for a self-hosted endpoint, or None."""
    from pydantic_ai.models.openai import OpenAIChatModelSettings

    effort = reasoning_effort(model_config)
    if effort is None:
        return None
    return OpenAIChatModelSettings(openai_reasoning_effort=effort)


def get_model(
    model_config: "ModelConfig",
    app_config: "AppConfig | None" = None,
) -> Any:
    """Get a model instance for the specified configuration."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    if app_config is None:
        from haiku.rag.config import get_config

        app_config = get_config()

    provider = model_config.provider
    model = model_config.name
    _check_provider_known(provider)
    check_api_key_supported(model_config, {"openai", "ollama", "openrouter", "vllm"})

    if provider == "ollama":
        model_settings = apply_common_settings(
            reasoning_effort_settings(model_config), model_config, map_thinking=False
        )

        # Ollama's OpenAI-compatible API lives under /v1. Append it if the
        # configured base_url doesn't already include it.
        base_url = model_config.base_url or app_config.providers.ollama.base_url
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        return OpenAIChatModel(
            model_name=model,
            provider=OllamaProvider(base_url=base_url, api_key=model_config.api_key),
            settings=model_settings,
            profile=_OPENAI_COMPAT_PROFILE,
        )

    elif provider == "vllm":
        from pydantic_ai.providers.vllm import VLLMProvider

        # VLLMProvider's profile carries the strict-chat-template flag this
        # module applies elsewhere.
        return OpenAIChatModel(
            model_name=model,
            provider=VLLMProvider(
                base_url=vllm_base_url(model_config.base_url),
                api_key=model_config.api_key,
            ),
            settings=apply_common_settings(
                reasoning_effort_settings(model_config),
                model_config,
                map_thinking=False,
            ),
        )

    elif provider == "openai":
        # A base_url names a self-hosted server (vLLM, LM Studio, sglang).
        if model_config.base_url:
            return OpenAIChatModel(
                model_name=model,
                provider=OpenAIProvider(
                    base_url=model_config.base_url, api_key=model_config.api_key
                ),
                settings=apply_common_settings(
                    reasoning_effort_settings(model_config),
                    model_config,
                    map_thinking=False,
                ),
                profile=_OPENAI_COMPAT_PROFILE,
            )

        # api.openai.com: pydantic-ai's profile knows these models, so the
        # unified setting maps `thinking` per model, always-on ones included.
        return OpenAIChatModel(
            model_name=model,
            provider=(
                OpenAIProvider(api_key=model_config.api_key)
                if model_config.api_key
                else "openai"
            ),
            settings=apply_common_settings(None, model_config),
        )

    elif provider == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        # OpenRouterModel maps the unified `thinking` onto its own `reasoning`
        # field, with the profile inferred from the vendor prefix of the name.
        return OpenRouterModel(
            model,
            provider=(
                OpenRouterProvider(api_key=model_config.api_key)
                if model_config.api_key
                else "openrouter"
            ),
            settings=apply_common_settings(None, model_config),
        )

    elif provider == "anthropic":
        from anthropic.types.beta import BetaThinkingConfigDisabledParam
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

        anthropic_settings: Any = None

        # Unified `thinking=False` omits the request field, which leaves the
        # adaptive-thinking models (Sonnet 4.6+, Opus 4.6+) thinking by default.
        disable_thinking = model_config.thinking is False
        if disable_thinking:
            thinking_disabled: BetaThinkingConfigDisabledParam = {"type": "disabled"}
            anthropic_settings = AnthropicModelSettings(
                anthropic_thinking=thinking_disabled
            )

        anthropic_settings = apply_common_settings(
            anthropic_settings, model_config, map_thinking=not disable_thinking
        )

        return AnthropicModel(model_name=model, settings=anthropic_settings)

    elif provider == "google":
        from pydantic_ai.models.google import GoogleModel

        return GoogleModel(
            model_name=model,
            settings=apply_common_settings(None, model_config),
        )

    elif provider == "groq":
        from pydantic_ai.models.groq import GroqModel

        return GroqModel(
            model_name=model,
            settings=apply_common_settings(None, model_config),
        )

    elif provider == "bedrock":
        from pydantic_ai.models.bedrock import (
            BedrockConverseModel,
            BedrockModelSettings,
        )

        bedrock_settings: Any = None

        # Same omission as the direct Anthropic branch: unified `thinking=False`
        # leaves the adaptive-thinking Claude models thinking. Bedrock ids are
        # `[<geo>.]<family>.<model>`, as in `us.anthropic.claude-...`.
        disable_claude_thinking = (
            model_config.thinking is False and "anthropic." in model
        )
        if disable_claude_thinking:
            bedrock_settings = BedrockModelSettings(
                bedrock_additional_model_requests_fields={
                    "thinking": {"type": "disabled"}
                }
            )

        return BedrockConverseModel(
            model_name=model,
            settings=apply_common_settings(
                bedrock_settings, model_config, map_thinking=not disable_claude_thinking
            ),
        )

    elif provider == "mistral":
        from pydantic_ai.models.mistral import MistralModel

        return MistralModel(
            model_name=model,
            settings=apply_common_settings(None, model_config),
        )

    else:
        # Pydantic AI builds the model from the string, which carries no settings.
        dropped = [
            name
            for name in ("temperature", "max_tokens", "thinking", "extra_body")
            if getattr(model_config, name) is not None
        ]
        if dropped:
            logger.warning(
                "Provider %r is passed to Pydantic AI by name, so %s is not applied.",
                provider,
                ", ".join(dropped),
            )
        return f"{provider}:{model}"

import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.version import Version, parse

if TYPE_CHECKING:
    from haiku.rag.graph.common.models import Citation


def apply_common_settings(
    settings: Any | None,
    settings_class: type[Any],
    model_config: Any,
) -> Any | None:
    """Apply common settings (temperature, max_tokens) to model settings.

    Args:
        settings: Existing settings instance or None
        settings_class: Settings class to instantiate if needed
        model_config: ModelConfig with temperature and max_tokens

    Returns:
        Updated settings instance or None if no settings to apply
    """
    if model_config.temperature is None and model_config.max_tokens is None:
        return settings

    if settings is None:
        settings_dict = settings_class()
    else:
        settings_dict = settings

    if model_config.temperature is not None:
        settings_dict["temperature"] = model_config.temperature

    if model_config.max_tokens is not None:
        settings_dict["max_tokens"] = model_config.max_tokens

    return settings_dict


def get_model(
    model_config: Any,
    app_config: Any | None = None,
) -> Any:
    """
    Get a model instance for the specified configuration.

    Args:
        model_config: ModelConfig with provider, model, and settings
        app_config: AppConfig for provider base URLs (defaults to global Config)

    Returns:
        A configured model instance
    """
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    if app_config is None:
        from haiku.rag.config import Config

        app_config = Config

    provider = model_config.provider
    model = model_config.name

    if provider == "ollama":
        model_settings = None

        # Apply thinking control for gpt-oss
        if model == "gpt-oss" and model_config.enable_thinking is not None:
            if model_config.enable_thinking is False:
                model_settings = OpenAIChatModelSettings(openai_reasoning_effort="low")
            else:
                model_settings = OpenAIChatModelSettings(openai_reasoning_effort="high")

        model_settings = apply_common_settings(
            model_settings, OpenAIChatModelSettings, model_config
        )

        return OpenAIChatModel(
            model_name=model,
            provider=OllamaProvider(
                base_url=f"{app_config.providers.ollama.base_url}/v1"
            ),
            settings=model_settings,
        )

    elif provider == "openai":
        openai_settings: Any = None

        # Apply thinking control
        if model_config.enable_thinking is not None:
            if model_config.enable_thinking is False:
                openai_settings = OpenAIChatModelSettings(openai_reasoning_effort="low")
            else:
                openai_settings = OpenAIChatModelSettings(
                    openai_reasoning_effort="high"
                )

        openai_settings = apply_common_settings(
            openai_settings, OpenAIChatModelSettings, model_config
        )

        return OpenAIChatModel(model_name=model, settings=openai_settings)

    elif provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

        anthropic_settings: Any = None

        # Apply thinking control
        if model_config.enable_thinking is not None:
            if model_config.enable_thinking:
                anthropic_settings = AnthropicModelSettings(
                    anthropic_thinking={"type": "enabled", "budget_tokens": 4096}
                )
            else:
                anthropic_settings = AnthropicModelSettings(
                    anthropic_thinking={"type": "disabled"}
                )

        anthropic_settings = apply_common_settings(
            anthropic_settings, AnthropicModelSettings, model_config
        )

        return AnthropicModel(model_name=model, settings=anthropic_settings)

    elif provider == "gemini":
        from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

        gemini_settings: Any = None

        # Apply thinking control
        if model_config.enable_thinking is not None:
            gemini_settings = GoogleModelSettings(
                google_thinking_config={
                    "include_thoughts": model_config.enable_thinking
                }
            )

        gemini_settings = apply_common_settings(
            gemini_settings, GoogleModelSettings, model_config
        )

        return GoogleModel(model_name=model, settings=gemini_settings)

    elif provider == "groq":
        from pydantic_ai.models.groq import GroqModel, GroqModelSettings

        groq_settings: Any = None

        # Apply thinking control
        if model_config.enable_thinking is not None:
            if model_config.enable_thinking:
                groq_settings = GroqModelSettings(groq_reasoning_format="parsed")
            else:
                groq_settings = GroqModelSettings(groq_reasoning_format="hidden")

        groq_settings = apply_common_settings(
            groq_settings, GroqModelSettings, model_config
        )

        return GroqModel(model_name=model, settings=groq_settings)

    elif provider == "bedrock":
        from pydantic_ai.models.bedrock import (
            BedrockConverseModel,
            BedrockModelSettings,
        )

        bedrock_settings: Any = None

        # Apply thinking control for Claude models
        if model_config.enable_thinking is not None:
            additional_fields: dict[str, Any] = {}
            if model.startswith("anthropic.claude"):
                if model_config.enable_thinking:
                    additional_fields = {
                        "thinking": {"type": "enabled", "budget_tokens": 4096}
                    }
                else:
                    additional_fields = {"thinking": {"type": "disabled"}}
            elif "gpt" in model or "o1" in model or "o3" in model:
                # OpenAI models on Bedrock
                additional_fields = {
                    "reasoning_effort": "high"
                    if model_config.enable_thinking
                    else "low"
                }
            elif "qwen" in model:
                # Qwen models on Bedrock
                additional_fields = {
                    "reasoning_config": "high"
                    if model_config.enable_thinking
                    else "low"
                }

            if additional_fields:
                bedrock_settings = BedrockModelSettings(
                    bedrock_additional_model_requests_fields=additional_fields
                )

        bedrock_settings = apply_common_settings(
            bedrock_settings, BedrockModelSettings, model_config
        )

        return BedrockConverseModel(model_name=model, settings=bedrock_settings)

    elif provider == "vllm":
        vllm_settings = None

        # Apply thinking control for gpt-oss
        if model == "gpt-oss" and model_config.enable_thinking is not None:
            if model_config.enable_thinking is False:
                vllm_settings = OpenAIChatModelSettings(openai_reasoning_effort="low")
            else:
                vllm_settings = OpenAIChatModelSettings(openai_reasoning_effort="high")

        vllm_settings = apply_common_settings(
            vllm_settings, OpenAIChatModelSettings, model_config
        )

        return OpenAIChatModel(
            model_name=model,
            provider=OpenAIProvider(
                base_url=f"{app_config.providers.vllm.research_base_url or app_config.providers.vllm.qa_base_url}/v1",
                api_key="none",
            ),
            settings=vllm_settings,
        )

    elif provider == "lm_studio":
        model_settings = None

        # Apply thinking control for gpt-oss
        if model == "gpt-oss" and model_config.enable_thinking is not None:
            if model_config.enable_thinking is False:
                model_settings = OpenAIChatModelSettings(openai_reasoning_effort="low")
            else:
                model_settings = OpenAIChatModelSettings(openai_reasoning_effort="high")

        model_settings = apply_common_settings(
            model_settings, OpenAIChatModelSettings, model_config
        )

        return OpenAIChatModel(
            model_name=model,
            provider=OpenAIProvider(
                base_url=f"{app_config.providers.lm_studio.base_url}/v1",
                api_key="dummy",
            ),
            settings=model_settings,
        )

    else:
        # For any other provider, use string format and let Pydantic AI handle it
        return f"{provider}:{model}"


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_citations(citations: "list[Citation]") -> str:
    """Format citations as markdown string."""
    if not citations:
        return ""
    lines = ["## Citations\n"]
    for c in citations:
        # Build citation header with document_id and chunk_id
        parts = [
            f"- document_id: `{c.document_id}` chunk_id: `{c.chunk_id}` "
            f"uri: **{c.document_uri}**"
        ]
        if c.document_title:
            parts.append(f' - "{c.document_title}"')
        location_parts = []
        if c.page_numbers:
            if len(c.page_numbers) == 1:
                location_parts.append(f"p. {c.page_numbers[0]}")
            else:
                location_parts.append(f"pp. {c.page_numbers[0]}-{c.page_numbers[-1]}")
        if c.headings:
            location_parts.append(f"Section: {c.headings[-1]}")
        if location_parts:
            parts.append(f" ({', '.join(location_parts)})")
        lines.append("".join(parts))
        # Add truncated content excerpt
        excerpt = c.content[:500] + "…" if len(c.content) > 500 else c.content
        excerpt = excerpt.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        lines.append(f"\n  {excerpt}\n")
    return "\n".join(lines)


def get_default_data_dir() -> Path:
    """Get the user data directory for the current system platform.

    Linux: ~/.local/share/haiku.rag
    macOS: ~/Library/Application Support/haiku.rag
    Windows: C:/Users/<USER>/AppData/Roaming/haiku.rag

    Returns:
        User Data Path.
    """
    home = Path.home()

    system_paths = {
        "win32": home / "AppData/Roaming/haiku.rag",
        "linux": home / ".local/share/haiku.rag",
        "darwin": home / "Library/Application Support/haiku.rag",
    }

    data_path = system_paths[sys.platform]
    return data_path


async def is_up_to_date() -> tuple[bool, Version, Version]:
    """Check whether haiku.rag is current.

    Returns:
        A tuple containing a boolean indicating whether haiku.rag is current,
        the running version and the latest version.
    """

    # Lazy import to avoid pulling httpx (and its deps) on module import
    import httpx

    async with httpx.AsyncClient() as client:
        running_version = parse(metadata.version("haiku.rag-slim"))
        try:
            response = await client.get("https://pypi.org/pypi/haiku.rag/json")
            data = response.json()
            pypi_version = parse(data["info"]["version"])
        except Exception:
            # If no network connection, do not raise alarms.
            pypi_version = running_version
    return running_version >= pypi_version, running_version, pypi_version



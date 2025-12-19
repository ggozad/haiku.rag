import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dateutil import parser as dateutil_parser
from packaging.version import Version, parse

if TYPE_CHECKING:
    from rich.console import RenderableType

    from haiku.rag.config.models import AppConfig, ModelConfig
    from haiku.rag.graph.research.models import Citation


def parse_datetime(s: str) -> datetime:
    """Parse a datetime string into a datetime object.

    Supports:
    - ISO 8601 format: "2025-01-15T14:30:00", "2025-01-15T14:30:00Z", "2025-01-15T14:30:00+00:00"
    - Date only: "2025-01-15" (interpreted as 00:00:00)
    - Various other formats via dateutil

    Args:
        s: String to parse

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If the string cannot be parsed
    """
    try:
        return dateutil_parser.parse(s)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Could not parse datetime: {s}. "
            "Use ISO 8601 format (e.g., 2025-01-15T14:30:00) or date (e.g., 2025-01-15)"
        ) from e


def to_utc(dt: datetime) -> datetime:
    """Convert a datetime to UTC.

    - Naive datetimes are assumed to be local time and converted to UTC
    - Datetimes with timezone info are converted to UTC
    - UTC datetimes are returned as-is

    Args:
        dt: Datetime to convert

    Returns:
        Datetime in UTC timezone
    """
    if dt.tzinfo is None:
        # Naive datetime - assume local time
        local_dt = dt.astimezone()  # Adds local timezone
        return local_dt.astimezone(UTC)
    elif dt.tzinfo == UTC:
        return dt
    else:
        return dt.astimezone(UTC)


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
    model_config: "ModelConfig",
    app_config: "AppConfig | None" = None,
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

        # Use model-level base_url if set, otherwise fall back to providers config
        base_url = model_config.base_url or f"{app_config.providers.ollama.base_url}/v1"

        return OpenAIChatModel(
            model_name=model,
            provider=OllamaProvider(base_url=base_url),
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

        # Use model-level base_url if set (for vLLM, LM Studio, etc.)
        if model_config.base_url:
            return OpenAIChatModel(
                model_name=model,
                provider=OpenAIProvider(base_url=model_config.base_url),
                settings=openai_settings,
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
    """Format citations as plain text with preserved formatting.

    Used by things like the MCP server where Rich renderables are not available.
    """
    if not citations:
        return ""

    lines = ["## Citations\n"]

    for c in citations:
        # Header line
        header = f"[{c.document_id}:{c.chunk_id}]"

        # Location info
        location_parts = []
        if c.page_numbers:
            if len(c.page_numbers) == 1:
                location_parts.append(f"p. {c.page_numbers[0]}")
            else:
                location_parts.append(f"pp. {c.page_numbers[0]}-{c.page_numbers[-1]}")
        if c.headings:
            location_parts.append(f"Section: {c.headings[-1]}")

        source = c.document_uri
        if c.document_title:
            source = f"{c.document_title} ({c.document_uri})"
        if location_parts:
            source += f" - {', '.join(location_parts)}"

        lines.append(f"{header} {source}")
        lines.append(c.content)
        lines.append("")

    return "\n".join(lines)


def format_citations_rich(citations: "list[Citation]") -> "list[RenderableType]":
    """Format citations as Rich renderables.

    Returns a list of Rich Panel objects for direct console printing,
    with content rendered as markdown for syntax highlighting.
    """
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    if not citations:
        return []

    renderables: list[RenderableType] = []
    renderables.append(Text("Citations", style="bold"))

    for c in citations:
        # Build header with IDs
        header = Text()
        header.append("doc: ", style="dim")
        header.append(c.document_id, style="cyan")
        header.append("  chunk: ", style="dim")
        header.append(c.chunk_id, style="cyan")

        # Location info for subtitle
        location_parts = []
        if c.page_numbers:
            if len(c.page_numbers) == 1:
                location_parts.append(f"p. {c.page_numbers[0]}")
            else:
                location_parts.append(f"pp. {c.page_numbers[0]}-{c.page_numbers[-1]}")
        if c.headings:
            location_parts.append(f"Section: {c.headings[-1]}")

        subtitle = c.document_uri
        if c.document_title:
            subtitle = f"{c.document_title} ({c.document_uri})"
        if location_parts:
            subtitle += f" - {', '.join(location_parts)}"
        panel = Panel(
            Markdown(c.content),
            title=header,
            subtitle=subtitle,
            subtitle_align="left",
            border_style="dim",
        )
        renderables.append(panel)

    return renderables


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


def build_prompt(base_prompt: str, config: "AppConfig") -> str:
    """Build a prompt with domain_preamble prepended if configured.

    Args:
        base_prompt: The base prompt to use
        config: AppConfig with prompts.domain_preamble

    Returns:
        Prompt with domain_preamble prepended if configured
    """
    if config.prompts.domain_preamble:
        return f"{config.prompts.domain_preamble}\n\n{base_prompt}"
    return base_prompt


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

import math
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from packaging.version import Version, parse

if TYPE_CHECKING:
    from pydantic_ai.messages import BinaryContent
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from rich.console import RenderableType

    from haiku.rag.client import HaikuRAG
    from haiku.rag.config.models import AppConfig, EmbeddingModelConfig, ModelConfig
    from haiku.rag.store.models.citation import Citation


def parse_model_option(value: str) -> "ModelConfig":
    """Parse a 'provider:name' string into a ModelConfig."""
    from haiku.rag.config.models import ModelConfig

    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid model format '{value}'. Expected 'provider:name' (e.g. 'ollama:gpt-oss')."
        )
    return ModelConfig(provider=parts[0], name=parts[1])


def check_api_key_supported(
    model_config: "ModelConfig | EmbeddingModelConfig", supported: set[str]
) -> None:
    """Reject a configured api_key on a provider whose client we never build.

    Those providers reach their vendor SDK by name and read their own
    environment variable, so a key in the config would be dropped silently.
    """
    if model_config.api_key and model_config.provider not in supported:
        raise ValueError(
            f"api_key is not supported on the '{model_config.provider}' provider "
            f"(supported: {', '.join(sorted(supported))}). Set that provider's "
            "own API key environment variable instead."
        )


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def image_binary_content(data: bytes) -> "BinaryContent":
    """Wrap raw image bytes as BinaryContent with the sniffed media type."""
    from io import BytesIO

    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError
    from pydantic_ai.messages import BinaryContent

    try:
        fmt = PILImage.open(BytesIO(data)).format or "PNG"
    except UnidentifiedImageError as e:
        raise ValueError("data is not a recognizable image") from e
    return BinaryContent(data=data, media_type=f"image/{fmt.lower()}")


def apply_common_settings(
    settings: Any | None,
    model_config: Any,
    *,
    map_thinking: bool = True,
) -> Any | None:
    """Apply the settings every provider shares onto a model settings dict.

    Args:
        settings: Existing settings instance or None
        model_config: ModelConfig with temperature and max_tokens
        map_thinking: Whether to map `enable_thinking` onto the unified
            `thinking` setting. The OpenAI-compatible branches opt out and set
            `openai_reasoning_effort` themselves, so that models whose profile
            advertises thinking without OpenAI reasoning support (Ollama's
            deepseek-r1, for one) keep receiving no `reasoning_effort`.

    Returns:
        Updated settings instance or None if no settings to apply
    """
    thinking = model_config.enable_thinking if map_thinking else None

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


# Strict OpenAI-compatible backends (some vLLM chat templates, e.g. Qwen's)
# reject more than one leading system message. Instructions from multiple
# sources (agent preamble, capability instructions, dynamic notices) map to
# one system message each, so have pydantic-ai merge them for any endpoint
# that is not api.openai.com. Harmless on backends that allow multiples.
_OPENAI_COMPAT_PROFILE: "OpenAIModelProfile" = {
    "openai_chat_supports_multiple_system_messages": False
}


def get_model(
    model_config: "ModelConfig",
    app_config: "AppConfig | None" = None,
) -> Any:
    """
    Get a model instance for the specified configuration.

    Args:
        model_config: ModelConfig with provider, model, and settings
        app_config: AppConfig for provider base URLs (defaults to the current global config)

    Returns:
        A configured model instance
    """
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    if app_config is None:
        from haiku.rag.config import get_config

        app_config = get_config()

    provider = model_config.provider
    model = model_config.name
    check_api_key_supported(model_config, {"openai", "ollama"})

    if provider == "ollama":
        model_settings = None

        # Apply thinking control for gpt-oss
        if model == "gpt-oss" and model_config.enable_thinking is not None:
            if model_config.enable_thinking is False:
                model_settings = OpenAIChatModelSettings(openai_reasoning_effort="low")
            else:
                model_settings = OpenAIChatModelSettings(openai_reasoning_effort="high")

        model_settings = apply_common_settings(
            model_settings, model_config, map_thinking=False
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

    elif provider == "openai":
        from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile

        openai_settings: Any = None

        # Apply thinking control only for reasoning models (o-series, gpt-5)
        profile = cast(OpenAIModelProfile, openai_model_profile(model))
        if model_config.enable_thinking is not None and profile.get(
            "openai_supports_reasoning", False
        ):
            if model_config.enable_thinking is False:
                openai_settings = OpenAIChatModelSettings(openai_reasoning_effort="low")
            else:
                openai_settings = OpenAIChatModelSettings(
                    openai_reasoning_effort="high"
                )

        openai_settings = apply_common_settings(
            openai_settings, model_config, map_thinking=False
        )

        # Use model-level base_url if set (for vLLM, LM Studio, etc.)
        if model_config.base_url:
            return OpenAIChatModel(
                model_name=model,
                provider=OpenAIProvider(
                    base_url=model_config.base_url, api_key=model_config.api_key
                ),
                settings=openai_settings,
                profile=_OPENAI_COMPAT_PROFILE,
            )

        return OpenAIChatModel(
            model_name=model,
            provider=(
                OpenAIProvider(api_key=model_config.api_key)
                if model_config.api_key
                else "openai"
            ),
            settings=openai_settings,
        )

    elif provider == "anthropic":
        from anthropic.types.beta import BetaThinkingConfigDisabledParam
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

        anthropic_settings: Any = None

        # Unified `thinking=False` omits the request field, which leaves the
        # adaptive-thinking models (Sonnet 4.6+, Opus 4.6+) thinking by default.
        disable_thinking = model_config.enable_thinking is False
        if disable_thinking:
            thinking_disabled: BetaThinkingConfigDisabledParam = {"type": "disabled"}
            anthropic_settings = AnthropicModelSettings(
                anthropic_thinking=thinking_disabled
            )

        anthropic_settings = apply_common_settings(
            anthropic_settings, model_config, map_thinking=not disable_thinking
        )

        return AnthropicModel(model_name=model, settings=anthropic_settings)

    elif provider == "gemini":
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
            model_config.enable_thinking is False and "anthropic." in model
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


CITATION_PREVIEW_CHARS = 300


def _citation_pages(c: "Citation") -> str | None:
    if not c.page_numbers:
        return None
    if len(c.page_numbers) == 1:
        return f"p. {c.page_numbers[0]}"
    return f"pp. {c.page_numbers[0]}-{c.page_numbers[-1]}"


def _citation_section(c: "Citation") -> str | None:
    if c.headings:
        return c.headings[-1]
    return None


def _citation_label(c: "Citation") -> str:
    if c.document_title and c.document_uri:
        return f"{c.document_title} ({c.document_uri})"
    return c.document_title or c.document_uri


def format_citations(citations: "list[Citation]") -> str:
    """Format citations as plain text with preserved formatting.

    Used by things like the MCP server where Rich renderables are not available.
    Pictures referenced by the chunk are surfaced as ``[Figure: <ref>]`` markers.
    """
    if not citations:
        return ""

    lines = ["## Citations\n"]

    for i, c in enumerate(citations):
        idx = c.index if c.index is not None else (i + 1)
        title = c.document_title or c.document_uri
        header = f"[{idx}] {title}"

        location_parts = []
        pages = _citation_pages(c)
        if pages:
            location_parts.append(pages)
        section = _citation_section(c)
        if section:
            location_parts.append(f"Section: {section}")

        source = c.document_uri
        if location_parts:
            source += f" - {', '.join(location_parts)}"

        lines.append(f"{header} {source}")
        for ref in c.picture_refs:
            lines.append(f"[Figure: {ref}]")
        lines.append(c.content)
        lines.append("")

    return "\n".join(lines)


async def format_citations_rich(
    citations: "list[Citation]",
    client: "HaikuRAG | None" = None,
) -> "list[RenderableType]":
    """Format citations as Rich renderables for terminal display.

    Each citation becomes a Panel with a compact header (``[N] Title (URI) — locator``),
    a body holding any referenced figures followed by a truncated text preview, and
    a dimmed footer that exposes the document and chunk IDs.

    When ``client`` is supplied, picture bytes for ``picture_refs`` are fetched and
    rendered inline via ``textual_image``. Without a client, picture refs appear as
    ``[Figure: <ref>]`` text markers.
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    if not citations:
        return []

    renderables: list[RenderableType] = []
    renderables.append(Text(""))
    renderables.append(Text("Citations", style="bold green"))
    renderables.append(Text(""))

    for i, c in enumerate(citations):
        if i > 0:
            renderables.append(Text(""))
        idx = c.index if c.index is not None else (i + 1)

        header_parts: list[str] = [f"[{idx}] {_citation_label(c)}"]
        if c.source and client is not None and client._federated:
            header_parts.append(c.source)
        pages = _citation_pages(c)
        if pages:
            header_parts.append(pages)
        section = _citation_section(c)
        if section:
            header_parts.append(f"§{section}")
        header = Text(" — ".join(header_parts), style="bold")

        body: list[RenderableType] = []
        for ref in c.picture_refs:
            image_renderable = await _render_picture(
                client, c.document_id, ref, c.source
            )
            body.append(
                image_renderable
                if image_renderable
                else Text(f"[Figure: {ref}]", style="italic dim")
            )

        preview = c.content
        if len(preview) > CITATION_PREVIEW_CHARS:
            preview = preview[:CITATION_PREVIEW_CHARS].rstrip() + "…"
        body.append(Text(preview))

        footer = Text()
        footer.append("doc: ", style="dim")
        footer.append(c.document_id, style="dim cyan")
        footer.append("  chunk: ", style="dim")
        footer.append(c.chunk_id, style="dim cyan")

        panel = Panel(
            Group(*body),
            title=header,
            title_align="left",
            subtitle=footer,
            subtitle_align="left",
            border_style="dim",
        )
        renderables.append(panel)

    return renderables


async def _render_picture(
    client: "HaikuRAG | None", document_id: str, ref: str, source: str | None = None
) -> "RenderableType | None":
    """Fetch a picture and return a Rich renderable, or None on failure/no client."""
    if client is None:
        return None
    from io import BytesIO

    from PIL import Image as PILImage
    from textual_image.renderable import Image as RichImage

    data = await client.get_picture_bytes(document_id, ref, source)
    if not data:
        return None
    try:
        pil = PILImage.open(BytesIO(data))
        pil.load()
    except Exception:
        return None
    return RichImage(pil)


def raise_missing_extra(module: str, extra: str, exc: ModuleNotFoundError) -> NoReturn:
    """Report `module` as a missing optional dependency, naming its extra.

    Re-raises `exc` untouched when the failure came from inside an installed
    package rather than from `module` itself, so a broken transitive import is
    not misreported as "not installed".
    """
    if exc.name != module:
        raise exc
    raise ImportError(
        f"{module} is not installed. Install it with "
        f"`uv pip install 'haiku.rag-slim[{extra}]'`."
    ) from exc


def locate_database(location: str) -> tuple[str, Path | None]:
    """Split a configured location into (uri, db_path).

    A value with a scheme is a `lancedb.uri`; anything else is a local path.
    Routing a local path through `uri` would have `ConnectionMode` classify it as
    object storage, which opens it without the existence check a local database
    gets.
    """
    if "://" in location:
        return location, None
    return "", Path(location)


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


def escape_sql_string(value: str) -> str:
    """Escape single quotes in SQL string literals."""
    return value.replace("'", "''")


def get_package_versions() -> dict[str, str]:
    """Get versions of haiku.rag and its dependencies.

    Returns:
        Dict with keys: haiku_rag, lancedb, docling, pydantic_ai, docling_document_schema
    """
    from docling_core.types.doc.document import DoclingDocument

    versions = {
        "haiku_rag": metadata.version("haiku.rag-slim"),
        "lancedb": metadata.version("lancedb"),
        "pydantic_ai": metadata.version("pydantic-ai-slim"),
        "docling_document_schema": DoclingDocument.model_construct().version,
    }
    try:
        versions["docling"] = metadata.version("docling")
    except metadata.PackageNotFoundError:
        versions["docling"] = "not installed"
    return versions


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
        except Exception:  # pragma: no cover
            # If no network connection, do not raise alarms.
            pypi_version = running_version
    return running_version >= pypi_version, running_version, pypi_version

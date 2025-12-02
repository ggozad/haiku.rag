import importlib
import importlib.util
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

from packaging.version import Version, parse


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


def load_callable(path: str):
    """Load a callable from a dotted path or file path.

    Supported formats:
    - "package.module:func" or "package.module.func"
    - "path/to/file.py:func"

    Returns the loaded callable. Raises ValueError on failure.
    """
    if not path:
        raise ValueError("Empty callable path provided")

    module_part = None
    func_name = None

    if ":" in path:
        module_part, func_name = path.split(":", 1)
    else:
        # split by last dot for module.attr
        if "." in path:
            module_part, func_name = path.rsplit(".", 1)
        else:
            raise ValueError(
                "Invalid callable path format. Use 'module:func' or 'module.func' or 'file.py:func'."
            )

    # Try file path first
    mod: ModuleType | None = None
    module_path = Path(module_part)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    else:
        # Import as a module path
        try:
            mod = importlib.import_module(module_part)
        except Exception as e:
            raise ValueError(f"Failed to import module '{module_part}': {e}")

    if not hasattr(mod, func_name):
        raise ValueError(f"Callable '{func_name}' not found in module '{module_part}'")
    func = getattr(mod, func_name)
    if not callable(func):
        raise ValueError(
            f"Attribute '{func_name}' in module '{module_part}' is not callable"
        )
    return func


async def prefetch_models():
    """Prefetch runtime models (Docling + Ollama + HuggingFace tokenizer as configured)."""
    import asyncio

    import httpx

    from haiku.rag.config import Config

    try:
        from docling.utils.model_downloader import download_models

        await asyncio.to_thread(download_models)
    except ImportError:
        # Docling not installed, skip downloading docling models
        pass

    # Download HuggingFace tokenizer
    from transformers import AutoTokenizer

    await asyncio.to_thread(
        AutoTokenizer.from_pretrained, Config.processing.chunking_tokenizer
    )

    # Collect Ollama models from config
    required_models: set[str] = set()
    if Config.embeddings.model.provider == "ollama":
        required_models.add(Config.embeddings.model.name)
    if Config.qa.model.provider == "ollama":
        required_models.add(Config.qa.model.name)
    if Config.research.model.provider == "ollama":
        required_models.add(Config.research.model.name)
    if Config.reranking.model and Config.reranking.model.provider == "ollama":
        required_models.add(Config.reranking.model.name)

    if not required_models:
        return

    base_url = Config.providers.ollama.base_url

    async with httpx.AsyncClient(timeout=None) as client:
        for model in sorted(required_models):
            async with client.stream(
                "POST", f"{base_url}/api/pull", json={"model": model}
            ) as r:
                async for _ in r.aiter_lines():
                    pass

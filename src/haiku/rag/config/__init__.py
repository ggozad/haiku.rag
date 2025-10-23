import os

from haiku.rag.config.loader import (
    check_for_deprecated_env,
    find_config_file,
    generate_default_config,
    load_config_from_env,
    load_yaml_config,
)
from haiku.rag.config.models import (
    A2AConfig,
    AppConfig,
    EmbeddingsConfig,
    LanceDBConfig,
    OllamaConfig,
    ProcessingConfig,
    ProvidersConfig,
    QAConfig,
    RerankingConfig,
    ResearchConfig,
    StorageConfig,
    VLLMConfig,
)

__all__ = [
    "Config",
    "AppConfig",
    "StorageConfig",
    "LanceDBConfig",
    "EmbeddingsConfig",
    "RerankingConfig",
    "QAConfig",
    "ResearchConfig",
    "ProcessingConfig",
    "OllamaConfig",
    "VLLMConfig",
    "ProvidersConfig",
    "A2AConfig",
    "find_config_file",
    "load_yaml_config",
    "generate_default_config",
    "load_config_from_env",
    "set_config",
]


class ConfigProxy:
    """Proxy for the global configuration that allows runtime updates."""

    def __init__(self):
        # Load config from YAML file or use defaults
        config_path = find_config_file(None)
        if config_path:
            yaml_data = load_yaml_config(config_path)
            self._config = AppConfig.model_validate(yaml_data)
        else:
            self._config = AppConfig()

    def __getattr__(self, name):
        """Proxy attribute access to the underlying config."""
        return getattr(self._config, name)

    def set(self, config: AppConfig) -> None:
        """Replace the current configuration."""
        self._config = config


# Create the global Config instance
Config = ConfigProxy()

# Check for deprecated .env file
check_for_deprecated_env()


def set_config(config: AppConfig) -> None:
    """Set the global configuration programmatically.

    This allows library users to configure haiku.rag without needing
    a YAML file or environment variables.

    Args:
        config: The AppConfig instance to use globally.

    Example:
        >>> from haiku.rag.config import set_config, AppConfig
        >>> custom_config = AppConfig(
        ...     qa={"provider": "openai", "model": "gpt-4o"},
        ...     embeddings={"provider": "voyage", "model": "voyage-3"}
        ... )
        >>> set_config(custom_config)
    """
    Config.set(config)

import os

from haiku.rag.config.loader import (
    find_config_file,
    generate_default_config,
    load_yaml_config,
)
from haiku.rag.config.models import (
    AGUIConfig,
    AppConfig,
    ConversionOptions,
    EmbeddingsConfig,
    LanceDBConfig,
    MonitorConfig,
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
    "AGUIConfig",
    "AppConfig",
    "ConversionOptions",
    "StorageConfig",
    "MonitorConfig",
    "LanceDBConfig",
    "EmbeddingsConfig",
    "RerankingConfig",
    "QAConfig",
    "ResearchConfig",
    "ProcessingConfig",
    "OllamaConfig",
    "VLLMConfig",
    "ProvidersConfig",
    "find_config_file",
    "load_yaml_config",
    "generate_default_config",
    "get_config",
    "set_config",
]

# Global config instance - initially loads from default locations
_config: AppConfig | None = None


def _load_default_config() -> AppConfig:
    """Load config from default locations (used at import time)."""
    config_path = find_config_file(None)
    if config_path:
        yaml_data = load_yaml_config(config_path)
        return AppConfig.model_validate(yaml_data)
    return AppConfig()


def set_config(config: AppConfig) -> None:
    """Set the global config instance (used by CLI to override)."""
    global _config
    _config = config


def get_config() -> AppConfig:
    """Get the current config instance."""
    global _config
    if _config is None:
        _config = _load_default_config()
    return _config


# Legacy compatibility - Config is the default instance
Config = _load_default_config()

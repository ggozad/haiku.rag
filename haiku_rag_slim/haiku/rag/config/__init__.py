from typing import Any

from haiku.rag.config.loader import (
    MissingEnvVarError,
    expand_env_vars,
    find_config_file,
    generate_default_config,
    load_yaml_config,
    redact_secrets,
)
from haiku.rag.config.models import (
    APIConfig,
    AppConfig,
    CircuitBreakerConfig,
    ConversionOptions,
    DatabaseEntry,
    DoclingServeConfig,
    EmbeddingModelConfig,
    EmbeddingsConfig,
    FSSourceConfig,
    HTTPSourceConfig,
    IngestedDatabase,
    IngesterConfig,
    LanceDBConfig,
    MaintenanceConfig,
    MetadataProviderConfig,
    ModelConfig,
    OllamaConfig,
    PluginSourceConfig,
    ProcessingConfig,
    PromptsConfig,
    ProvidersConfig,
    QAConfig,
    QueueConfig,
    RerankingConfig,
    RetryPolicyConfig,
    S3SourceConfig,
    SourceConfig,
    StorageConfig,
    WebDAVSourceConfig,
    WorkerConfig,
)

__all__ = [
    "APIConfig",
    "AppConfig",
    "CircuitBreakerConfig",
    "ConversionOptions",
    "DatabaseEntry",
    "DoclingServeConfig",
    "EmbeddingModelConfig",
    "EmbeddingsConfig",
    "FSSourceConfig",
    "HTTPSourceConfig",
    "IngestedDatabase",
    "IngesterConfig",
    "MaintenanceConfig",
    "LanceDBConfig",
    "MetadataProviderConfig",
    "ModelConfig",
    "OllamaConfig",
    "PluginSourceConfig",
    "ProcessingConfig",
    "PromptsConfig",
    "ProvidersConfig",
    "QAConfig",
    "QueueConfig",
    "RerankingConfig",
    "RetryPolicyConfig",
    "S3SourceConfig",
    "SourceConfig",
    "StorageConfig",
    "WebDAVSourceConfig",
    "WorkerConfig",
    "MissingEnvVarError",
    "changed_sections",
    "expand_env_vars",
    "find_config_file",
    "generate_default_config",
    "get_config",
    "load_yaml_config",
    "redact_secrets",
    "set_config",
]

# Global config instance - initially loads from default locations
_config: AppConfig | None = None


def _load_default_config() -> AppConfig:
    """Load config from the default locations."""
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
    """Get the current config instance, loading it on first use."""
    global _config
    if _config is None:
        _config = _load_default_config()
    return _config


def changed_sections(
    running: AppConfig, read: AppConfig, ignored: dict[str, Any]
) -> list[str]:
    """The sections of `read` that differ from `running`, named to their second
    level. `ignored` is a pydantic `exclude` of the sections not compared."""
    before = running.model_dump(mode="json", exclude=ignored)
    after = read.model_dump(mode="json", exclude=ignored)
    changed = []
    for section, value in before.items():
        if value == after[section]:
            continue
        if isinstance(value, dict):
            changed.extend(
                f"{section}.{key}" for key in value if value[key] != after[section][key]
            )
        else:
            changed.append(section)
    return changed

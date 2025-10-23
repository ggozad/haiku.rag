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
    APIKeysConfig,
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
    "APIKeysConfig",
    "ProvidersConfig",
    "A2AConfig",
    "find_config_file",
    "load_yaml_config",
    "generate_default_config",
    "load_config_from_env",
]

# Load config from YAML file or use defaults
config_path = find_config_file(None)
if config_path:
    yaml_data = load_yaml_config(config_path)
    Config = AppConfig.model_validate(yaml_data)
else:
    Config = AppConfig()

# Check for deprecated .env file
check_for_deprecated_env()

# Export API keys to os.environ for provider libraries
if Config.providers.api_keys.openai:
    os.environ["OPENAI_API_KEY"] = Config.providers.api_keys.openai
if Config.providers.api_keys.voyage:
    os.environ["VOYAGE_API_KEY"] = Config.providers.api_keys.voyage
if Config.providers.api_keys.anthropic:
    os.environ["ANTHROPIC_API_KEY"] = Config.providers.api_keys.anthropic
if Config.providers.api_keys.cohere:
    # Cohere SDK expects CO_API_KEY (not COHERE_API_KEY)
    os.environ["CO_API_KEY"] = Config.providers.api_keys.cohere

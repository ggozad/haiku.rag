import os
import warnings
from pathlib import Path

import yaml


def find_config_file(cli_path: Path | None = None) -> Path | None:
    """Find the YAML config file using the search path.

    Search order:
    1. CLI-provided path (via HAIKU_RAG_CONFIG_PATH env var or parameter)
    2. ./haiku.rag.yaml (current directory)
    3. ~/.config/haiku.rag/config.yaml (user config)

    Returns None if no config file is found.
    """
    # Check environment variable first (set by CLI --config flag)
    if not cli_path:
        env_path = os.getenv("HAIKU_RAG_CONFIG_PATH")
        if env_path:
            cli_path = Path(env_path)

    if cli_path:
        if cli_path.exists():
            return cli_path
        raise FileNotFoundError(f"Config file not found: {cli_path}")

    cwd_config = Path.cwd() / "haiku.rag.yaml"
    if cwd_config.exists():
        return cwd_config

    user_config_dir = Path.home() / ".config" / "haiku.rag"
    user_config = user_config_dir / "config.yaml"
    if user_config.exists():
        return user_config

    return None


def load_yaml_config(path: Path) -> dict:
    """Load and parse a YAML config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def check_for_deprecated_env() -> None:
    """Check for .env file and warn if found."""
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        warnings.warn(
            ".env file detected but YAML configuration is now preferred. "
            "Environment variable configuration is deprecated and will be removed in future versions."
            "Run 'haiku-rag init-config' to generate a YAML config file.",
            DeprecationWarning,
            stacklevel=2,
        )


def generate_default_config() -> dict:
    """Generate a default YAML config structure with documentation."""
    return {
        "environment": "production",
        "storage": {
            "data_dir": "",
            "monitor_directories": [],
            "disable_autocreate": False,
            "vacuum_retention_seconds": 60,
        },
        "lancedb": {"uri": "", "api_key": "", "region": ""},
        "embeddings": {
            "provider": "ollama",
            "model": "qwen3-embedding",
            "vector_dim": 4096,
        },
        "reranking": {"provider": "", "model": ""},
        "qa": {"provider": "ollama", "model": "gpt-oss"},
        "research": {"provider": "", "model": ""},
        "processing": {
            "chunk_size": 256,
            "context_chunk_radius": 0,
            "markdown_preprocessor": "",
        },
        "providers": {
            "ollama": {"base_url": "http://localhost:11434"},
            "vllm": {
                "embeddings_base_url": "",
                "rerank_base_url": "",
                "qa_base_url": "",
                "research_base_url": "",
            },
        },
        "a2a": {"max_contexts": 1000},
    }


def load_config_from_env() -> dict:
    """Load current config from environment variables (for migration)."""
    result = {}

    env_mappings = {
        "ENV": "environment",
        "DEFAULT_DATA_DIR": ("storage", "data_dir"),
        "MONITOR_DIRECTORIES": ("storage", "monitor_directories"),
        "DISABLE_DB_AUTOCREATE": ("storage", "disable_autocreate"),
        "VACUUM_RETENTION_SECONDS": ("storage", "vacuum_retention_seconds"),
        "LANCEDB_URI": ("lancedb", "uri"),
        "LANCEDB_API_KEY": ("lancedb", "api_key"),
        "LANCEDB_REGION": ("lancedb", "region"),
        "EMBEDDINGS_PROVIDER": ("embeddings", "provider"),
        "EMBEDDINGS_MODEL": ("embeddings", "model"),
        "EMBEDDINGS_VECTOR_DIM": ("embeddings", "vector_dim"),
        "RERANK_PROVIDER": ("reranking", "provider"),
        "RERANK_MODEL": ("reranking", "model"),
        "QA_PROVIDER": ("qa", "provider"),
        "QA_MODEL": ("qa", "model"),
        "RESEARCH_PROVIDER": ("research", "provider"),
        "RESEARCH_MODEL": ("research", "model"),
        "CHUNK_SIZE": ("processing", "chunk_size"),
        "CONTEXT_CHUNK_RADIUS": ("processing", "context_chunk_radius"),
        "MARKDOWN_PREPROCESSOR": ("processing", "markdown_preprocessor"),
        "OLLAMA_BASE_URL": ("providers", "ollama", "base_url"),
        "VLLM_EMBEDDINGS_BASE_URL": ("providers", "vllm", "embeddings_base_url"),
        "VLLM_RERANK_BASE_URL": ("providers", "vllm", "rerank_base_url"),
        "VLLM_QA_BASE_URL": ("providers", "vllm", "qa_base_url"),
        "VLLM_RESEARCH_BASE_URL": ("providers", "vllm", "research_base_url"),
        "A2A_MAX_CONTEXTS": ("a2a", "max_contexts"),
    }

    for env_var, path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            if isinstance(path, tuple):
                current = result
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[path[-1]] = value
            else:
                result[path] = value

    return result

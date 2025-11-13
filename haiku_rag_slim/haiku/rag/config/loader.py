import os
from pathlib import Path

import yaml


def find_config_file(cli_path: Path | None = None) -> Path | None:
    """Find the YAML config file using the search path.

    Search order:
    1. CLI-provided path (via HAIKU_RAG_CONFIG_PATH env var or parameter)
    2. ./haiku.rag.yaml (current directory)
    3. Platform-specific user config directory

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

    # Use same directory as data storage for config
    from haiku.rag.utils import get_default_data_dir

    data_dir = get_default_data_dir()
    user_config = data_dir / "haiku.rag.yaml"
    if user_config.exists():
        return user_config

    return None


def load_yaml_config(path: Path) -> dict:
    """Load and parse a YAML config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def generate_default_config() -> dict:
    """Generate a default YAML config structure with documentation."""
    return {
        "environment": "production",
        "storage": {
            "data_dir": "",
            "vacuum_retention_seconds": 86400,
        },
        "monitor": {
            "directories": [],
            "ignore_patterns": [],
            "include_patterns": [],
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
        "agui": {
            "host": "0.0.0.0",
            "port": 8000,
            "cors_origins": ["*"],
            "cors_credentials": True,
            "cors_methods": ["GET", "POST", "OPTIONS"],
            "cors_headers": ["*"],
        },
    }

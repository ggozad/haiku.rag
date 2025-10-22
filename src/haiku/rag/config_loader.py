import os
import warnings
from pathlib import Path

import yaml


def find_config_file(cli_path: Path | None = None) -> Path | None:
    """Find the YAML config file using the search path.

    Search order:
    1. CLI-provided path (if given)
    2. ./haiku.rag.yaml (current directory)
    3. ~/.config/haiku.rag/config.yaml (user config)

    Returns None if no config file is found.
    """
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


def flatten_yaml_to_env_dict(yaml_dict: dict) -> dict:
    """Convert nested YAML structure to flat environment variable dict.

    Maps YAML structure like:
        embeddings:
          provider: ollama
          model: qwen3

    To flat dict like:
        EMBEDDINGS_PROVIDER: ollama
        EMBEDDINGS_MODEL: qwen3
    """
    result = {}

    # Top-level simple fields
    if "environment" in yaml_dict:
        result["ENV"] = yaml_dict["environment"]

    # Storage section
    if "storage" in yaml_dict:
        storage = yaml_dict["storage"]
        if "data_dir" in storage:
            result["DEFAULT_DATA_DIR"] = storage["data_dir"]
        if "monitor_directories" in storage:
            dirs = storage["monitor_directories"]
            if isinstance(dirs, list):
                result["MONITOR_DIRECTORIES"] = ",".join(str(d) for d in dirs)
            else:
                result["MONITOR_DIRECTORIES"] = str(dirs)
        if "disable_autocreate" in storage:
            result["DISABLE_DB_AUTOCREATE"] = storage["disable_autocreate"]
        if "vacuum_retention_seconds" in storage:
            result["VACUUM_RETENTION_SECONDS"] = storage["vacuum_retention_seconds"]

    # LanceDB section
    if "lancedb" in yaml_dict:
        lancedb = yaml_dict["lancedb"]
        if "uri" in lancedb:
            result["LANCEDB_URI"] = lancedb["uri"]
        if "api_key" in lancedb:
            result["LANCEDB_API_KEY"] = lancedb["api_key"]
        if "region" in lancedb:
            result["LANCEDB_REGION"] = lancedb["region"]

    # Embeddings section
    if "embeddings" in yaml_dict:
        embeddings = yaml_dict["embeddings"]
        if "provider" in embeddings:
            result["EMBEDDINGS_PROVIDER"] = embeddings["provider"]
        if "model" in embeddings:
            result["EMBEDDINGS_MODEL"] = embeddings["model"]
        if "vector_dim" in embeddings:
            result["EMBEDDINGS_VECTOR_DIM"] = embeddings["vector_dim"]

    # Reranking section
    if "reranking" in yaml_dict:
        reranking = yaml_dict["reranking"]
        if "provider" in reranking:
            result["RERANK_PROVIDER"] = reranking["provider"]
        if "model" in reranking:
            result["RERANK_MODEL"] = reranking["model"]

    # QA section
    if "qa" in yaml_dict:
        qa = yaml_dict["qa"]
        if "provider" in qa:
            result["QA_PROVIDER"] = qa["provider"]
        if "model" in qa:
            result["QA_MODEL"] = qa["model"]

    # Research section
    if "research" in yaml_dict:
        research = yaml_dict["research"]
        if "provider" in research:
            result["RESEARCH_PROVIDER"] = research["provider"]
        if "model" in research:
            result["RESEARCH_MODEL"] = research["model"]

    # Processing section
    if "processing" in yaml_dict:
        processing = yaml_dict["processing"]
        if "chunk_size" in processing:
            result["CHUNK_SIZE"] = processing["chunk_size"]
        if "context_chunk_radius" in processing:
            result["CONTEXT_CHUNK_RADIUS"] = processing["context_chunk_radius"]
        if "markdown_preprocessor" in processing:
            result["MARKDOWN_PREPROCESSOR"] = processing["markdown_preprocessor"]

    # Providers section
    if "providers" in yaml_dict:
        providers = yaml_dict["providers"]

        if "ollama" in providers:
            ollama = providers["ollama"]
            if "base_url" in ollama:
                result["OLLAMA_BASE_URL"] = ollama["base_url"]

        if "vllm" in providers:
            vllm = providers["vllm"]
            if "embeddings_base_url" in vllm:
                result["VLLM_EMBEDDINGS_BASE_URL"] = vllm["embeddings_base_url"]
            if "rerank_base_url" in vllm:
                result["VLLM_RERANK_BASE_URL"] = vllm["rerank_base_url"]
            if "qa_base_url" in vllm:
                result["VLLM_QA_BASE_URL"] = vllm["qa_base_url"]
            if "research_base_url" in vllm:
                result["VLLM_RESEARCH_BASE_URL"] = vllm["research_base_url"]

        if "api_keys" in providers:
            api_keys = providers["api_keys"]
            if "voyage" in api_keys:
                result["VOYAGE_API_KEY"] = api_keys["voyage"]
            if "openai" in api_keys:
                result["OPENAI_API_KEY"] = api_keys["openai"]
            if "anthropic" in api_keys:
                result["ANTHROPIC_API_KEY"] = api_keys["anthropic"]
            if "cohere" in api_keys:
                result["COHERE_API_KEY"] = api_keys["cohere"]

    # A2A section
    if "a2a" in yaml_dict:
        a2a = yaml_dict["a2a"]
        if "max_contexts" in a2a:
            result["A2A_MAX_CONTEXTS"] = a2a["max_contexts"]

    return result


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
            "api_keys": {"voyage": "", "openai": "", "anthropic": "", "cohere": ""},
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
        "VOYAGE_API_KEY": ("providers", "api_keys", "voyage"),
        "OPENAI_API_KEY": ("providers", "api_keys", "openai"),
        "ANTHROPIC_API_KEY": ("providers", "api_keys", "anthropic"),
        "COHERE_API_KEY": ("providers", "api_keys", "cohere"),
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

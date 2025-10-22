import os
from pathlib import Path

import pytest

from haiku.rag.config_loader import (
    find_config_file,
    flatten_yaml_to_env_dict,
    generate_default_config,
    load_config_from_env,
    load_yaml_config,
)


def test_load_yaml_config(tmp_path):
    """Test loading a YAML config file."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
environment: production
embeddings:
  provider: ollama
  model: test-model
  vector_dim: 1024
""")

    config = load_yaml_config(config_file)
    assert config["environment"] == "production"
    assert config["embeddings"]["provider"] == "ollama"
    assert config["embeddings"]["model"] == "test-model"
    assert config["embeddings"]["vector_dim"] == 1024


def test_flatten_yaml_to_env_dict():
    """Test converting nested YAML to flat env dict."""
    yaml_dict = {
        "environment": "development",
        "storage": {
            "data_dir": "/tmp/data",
            "monitor_directories": ["/path/one", "/path/two"],
            "disable_autocreate": True,
            "vacuum_retention_seconds": 30,
        },
        "embeddings": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "vector_dim": 1536,
        },
        "qa": {"provider": "anthropic", "model": "claude-3-haiku"},
        "processing": {"chunk_size": 512, "context_chunk_radius": 1},
        "providers": {
            "ollama": {"base_url": "http://localhost:11434"},
            "api_keys": {"openai": "test-key", "anthropic": "test-key-2"},
        },
    }

    result = flatten_yaml_to_env_dict(yaml_dict)

    assert result["ENV"] == "development"
    assert result["DEFAULT_DATA_DIR"] == "/tmp/data"
    assert result["MONITOR_DIRECTORIES"] == "/path/one,/path/two"
    assert result["DISABLE_DB_AUTOCREATE"] is True
    assert result["VACUUM_RETENTION_SECONDS"] == 30
    assert result["EMBEDDINGS_PROVIDER"] == "openai"
    assert result["EMBEDDINGS_MODEL"] == "text-embedding-3-small"
    assert result["EMBEDDINGS_VECTOR_DIM"] == 1536
    assert result["QA_PROVIDER"] == "anthropic"
    assert result["QA_MODEL"] == "claude-3-haiku"
    assert result["CHUNK_SIZE"] == 512
    assert result["CONTEXT_CHUNK_RADIUS"] == 1
    assert result["OLLAMA_BASE_URL"] == "http://localhost:11434"
    assert result["OPENAI_API_KEY"] == "test-key"
    assert result["ANTHROPIC_API_KEY"] == "test-key-2"


def test_flatten_yaml_empty():
    """Test flattening empty YAML dict returns empty dict."""
    result = flatten_yaml_to_env_dict({})
    assert result == {}


def test_find_config_file_cwd(tmp_path, monkeypatch):
    """Test finding config in current directory."""
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "haiku.rag.yaml"
    config_file.write_text("environment: production")

    found = find_config_file()
    assert found == config_file


def test_find_config_file_user_config(tmp_path, monkeypatch):
    """Test finding config in user config directory."""
    monkeypatch.chdir(tmp_path)
    user_config_dir = tmp_path / ".config" / "haiku.rag"
    user_config_dir.mkdir(parents=True)
    config_file = user_config_dir / "config.yaml"
    config_file.write_text("environment: production")

    # Mock home directory
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    found = find_config_file()
    assert found == config_file


def test_find_config_file_cli_path(tmp_path):
    """Test finding config via CLI path parameter."""
    config_file = tmp_path / "custom.yaml"
    config_file.write_text("environment: production")

    found = find_config_file(config_file)
    assert found == config_file


def test_find_config_file_env_var(tmp_path, monkeypatch):
    """Test finding config via HAIKU_RAG_CONFIG_PATH env var."""
    config_file = tmp_path / "from-env.yaml"
    config_file.write_text("environment: production")

    monkeypatch.setenv("HAIKU_RAG_CONFIG_PATH", str(config_file))

    found = find_config_file()
    assert found == config_file


def test_find_config_file_not_found(tmp_path, monkeypatch):
    """Test returning None when no config found."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    found = find_config_file()
    assert found is None


def test_find_config_file_cli_path_not_exists(tmp_path):
    """Test error when CLI path doesn't exist."""
    config_file = tmp_path / "nonexistent.yaml"

    with pytest.raises(FileNotFoundError):
        find_config_file(config_file)


def test_generate_default_config():
    """Test generating default config structure."""
    config = generate_default_config()

    assert config["environment"] == "production"
    assert "storage" in config
    assert "embeddings" in config
    assert "qa" in config
    assert "providers" in config
    assert config["embeddings"]["provider"] == "ollama"
    assert config["embeddings"]["vector_dim"] == 4096


def test_load_config_from_env(monkeypatch):
    """Test loading config from environment variables."""
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDINGS_VECTOR_DIM", "1536")
    monkeypatch.setenv("QA_PROVIDER", "anthropic")
    monkeypatch.setenv("QA_MODEL", "claude-3-haiku")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = load_config_from_env()

    assert config["environment"] == "development"
    assert config["embeddings"]["provider"] == "openai"
    assert config["embeddings"]["model"] == "text-embedding-3-small"
    assert config["embeddings"]["vector_dim"] == "1536"
    assert config["qa"]["provider"] == "anthropic"
    assert config["qa"]["model"] == "claude-3-haiku"
    assert config["providers"]["api_keys"]["openai"] == "test-key"


def test_load_config_from_env_empty():
    """Test loading from env when no relevant vars set."""
    # Clear any env vars that might be set
    env_vars = [
        "ENV",
        "EMBEDDINGS_PROVIDER",
        "QA_PROVIDER",
        "OPENAI_API_KEY",
    ]
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    try:
        config = load_config_from_env()
        # Should return empty or minimal dict
        assert isinstance(config, dict)
    finally:
        # Restore original values
        for var, value in original_values.items():
            if value is not None:
                os.environ[var] = value


def test_config_precedence_cwd_over_user(tmp_path, monkeypatch):
    """Test that cwd config takes precedence over user config."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Create both configs
    cwd_config = tmp_path / "haiku.rag.yaml"
    cwd_config.write_text("environment: from-cwd")

    user_config_dir = tmp_path / ".config" / "haiku.rag"
    user_config_dir.mkdir(parents=True)
    user_config = user_config_dir / "config.yaml"
    user_config.write_text("environment: from-user")

    found = find_config_file()
    assert found == cwd_config
    assert found is not None

    config = load_yaml_config(found)
    assert config["environment"] == "from-cwd"


def test_config_precedence_env_var_over_cwd(tmp_path, monkeypatch):
    """Test that HAIKU_RAG_CONFIG_PATH env var takes precedence."""
    monkeypatch.chdir(tmp_path)

    # Create cwd config
    cwd_config = tmp_path / "haiku.rag.yaml"
    cwd_config.write_text("environment: from-cwd")

    # Create env var config
    env_config = tmp_path / "from-env.yaml"
    env_config.write_text("environment: from-env-var")
    monkeypatch.setenv("HAIKU_RAG_CONFIG_PATH", str(env_config))

    found = find_config_file()
    assert found == env_config
    assert found is not None

    config = load_yaml_config(found)
    assert config["environment"] == "from-env-var"

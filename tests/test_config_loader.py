import pytest

from haiku.rag.config.loader import (
    find_config_file,
    generate_default_config,
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

    # Mock get_default_data_dir to return tmp_path
    def mock_get_default_data_dir():
        return tmp_path

    monkeypatch.setattr(
        "haiku.rag.utils.get_default_data_dir", mock_get_default_data_dir
    )

    config_file = tmp_path / "haiku.rag.yaml"
    config_file.write_text("environment: production")

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

    # Mock get_default_data_dir to return tmp_path
    def mock_get_default_data_dir():
        return tmp_path

    monkeypatch.setattr(
        "haiku.rag.utils.get_default_data_dir", mock_get_default_data_dir
    )

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


def test_config_precedence_cwd_over_user(tmp_path, monkeypatch):
    """Test that cwd config takes precedence over user config."""
    # Create separate directories for cwd and user config
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    user_dir = tmp_path / "user"
    user_dir.mkdir()

    monkeypatch.chdir(cwd_dir)

    # Mock get_default_data_dir to return user_dir
    def mock_get_default_data_dir():
        return user_dir

    monkeypatch.setattr(
        "haiku.rag.utils.get_default_data_dir", mock_get_default_data_dir
    )

    # Create both configs
    cwd_config = cwd_dir / "haiku.rag.yaml"
    cwd_config.write_text("environment: from-cwd")

    user_config = user_dir / "haiku.rag.yaml"
    user_config.write_text("environment: from-user")

    found = find_config_file()
    assert found == cwd_config
    assert found is not None

    config_data = load_yaml_config(found)
    assert config_data["environment"] == "from-cwd"


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

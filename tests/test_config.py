import pytest
import yaml

from haiku.rag.config import AppConfig
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
  model:
    provider: ollama
    name: test-model
    vector_dim: 1024
""")

    config = load_yaml_config(config_file)
    assert config["environment"] == "production"
    assert config["embeddings"]["model"]["provider"] == "ollama"
    assert config["embeddings"]["model"]["name"] == "test-model"
    assert config["embeddings"]["model"]["vector_dim"] == 1024


def test_find_config_file_cwd(tmp_path, monkeypatch):
    """Test finding config in current directory."""
    monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "haiku.rag.yaml"
    config_file.write_text("environment: production")

    found = find_config_file()
    assert found == config_file


def test_find_config_file_user_config(tmp_path, monkeypatch):
    """Test finding config in user config directory."""
    monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)
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
    monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)
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


def test_config_precedence_cwd_over_user(tmp_path, monkeypatch):
    """Test that cwd config takes precedence over user config."""
    monkeypatch.delenv("HAIKU_RAG_CONFIG_PATH", raising=False)

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


def test_generate_default_config_completeness():
    """Test that generated config has all fields from AppConfig and validates."""
    config_data = generate_default_config()

    # Validate against AppConfig model
    config = AppConfig.model_validate(config_data)

    # Get all fields from AppConfig
    expected_fields = set(AppConfig.model_fields.keys())
    actual_fields = set(config_data.keys())

    # Verify all expected fields are present
    assert expected_fields == actual_fields, (
        f"Missing fields: {expected_fields - actual_fields}, "
        f"Extra fields: {actual_fields - expected_fields}"
    )

    # Verify nested structures are dicts/lists, not model instances
    for field_name, field_value in config_data.items():
        assert not hasattr(field_value, "model_dump"), (
            f"Field {field_name} should be dict/primitive, not Pydantic model"
        )

    # Verify config validates successfully
    assert config.environment == "production"
    assert config.embeddings.model.provider == "ollama"
    assert config.qa.model.provider == "ollama"
    assert config.research.model.provider == "ollama"
    assert config.reranking.model is None


def test_init_config_creates_valid_yaml(tmp_path):
    """Test that generated config can be written to YAML and loaded back."""
    config_file = tmp_path / "test-config.yaml"

    # Generate and write config
    config_data = generate_default_config()

    with open(config_file, "w") as f:
        f.write("# haiku.rag configuration file\n")
        f.write(
            "# See https://ggozad.github.io/haiku.rag/configuration/ for details\n\n"
        )
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    # Load it back
    with open(config_file) as f:
        loaded_data = yaml.safe_load(f)

    # Validate it
    config = AppConfig.model_validate(loaded_data)
    assert config.environment == "production"

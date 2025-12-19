import os
import tempfile
from pathlib import Path

# Prevent tests from loading user's local haiku.rag.yaml by setting env var
# to an empty config file BEFORE any haiku.rag imports.
# This ensures tests always use default config values.
_test_config_dir = tempfile.mkdtemp()
_test_config_path = Path(_test_config_dir) / "test-defaults.yaml"
_test_config_path.write_text("{}")  # Empty YAML = use all defaults
os.environ["HAIKU_RAG_CONFIG_PATH"] = str(_test_config_path)

import pytest  # noqa: E402
import yaml  # noqa: E402
from datasets import Dataset, load_dataset, load_from_disk  # noqa: E402


@pytest.fixture(scope="session")
def qa_corpus() -> Dataset:
    ds_path = Path(__file__).parent / "data" / "dataset"
    ds_path.mkdir(parents=True, exist_ok=True)
    try:
        ds: Dataset = load_from_disk(ds_path)  # type: ignore
        return ds
    except FileNotFoundError:
        ds: Dataset = load_dataset("ServiceNow/repliqa")["repliqa_3"]  # type: ignore
        corpus = ds.filter(lambda doc: doc["document_topic"] == "News Stories")
        corpus.save_to_disk(ds_path)
        return corpus


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing.

    Note: Tests that need a database should use HaikuRAG with create=True.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test.lancedb"


@pytest.fixture
def temp_yaml_config(tmp_path, monkeypatch):
    """Create a temporary YAML config file for testing.

    This fixture creates a config file in a temp directory and sets
    the environment variable so config.py will load it.
    """
    config_file = tmp_path / "test-config.yaml"
    config_data = {
        "environment": "production",
        "storage": {
            "data_dir": "",
            "monitor_directories": [],
            "vacuum_retention_seconds": 60,
        },
        "embeddings": {
            "model": {
                "provider": "ollama",
                "name": "qwen3-embedding:4b",
                "vector_dim": 2560,
            }
        },
        "qa": {"model": {"provider": "ollama", "name": "gpt-oss"}},
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Set env var so config loader will find it
    monkeypatch.setenv("HAIKU_RAG_CONFIG_PATH", str(config_file))

    yield config_file

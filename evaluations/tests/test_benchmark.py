from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from evaluations.benchmark import (
    _load_config,
    _resolve_dataset,
    build_experiment_metadata,
)
from haiku.rag.config.models import AppConfig, ModelConfig


class TestBuildExperimentMetadata:
    def test_basic_metadata(self) -> None:
        config = AppConfig()
        result = build_experiment_metadata(
            dataset_key="test",
            test_cases=42,
            config=config,
        )

        assert result["dataset"] == "test"
        assert result["test_cases"] == 42
        assert result["embedder_provider"] == config.embeddings.model.provider
        assert result["embedder_model"] == config.embeddings.model.name
        assert result["embedder_dim"] == config.embeddings.model.vector_dim
        assert result["chunk_size"] == config.processing.chunk_size
        assert result["search_limit"] == config.search.limit
        assert result["context_radius"] == config.search.context_radius
        assert result["qa_provider"] == config.qa.model.provider
        assert result["qa_model"] == config.qa.model.name
        assert "judge_provider" not in result

    def test_with_judge_config(self) -> None:
        config = AppConfig()
        judge = ModelConfig(
            provider="ollama", name="gpt-oss", enable_thinking=False, temperature=0.0
        )
        result = build_experiment_metadata(
            dataset_key="test",
            test_cases=10,
            config=config,
            judge_config=judge,
        )

        assert result["judge_provider"] == "ollama"
        assert result["judge_model"] == "gpt-oss"
        assert result["judge_temperature"] == 0.0
        assert result["judge_enable_thinking"] is False

    def test_no_reranker(self) -> None:
        config = AppConfig()
        result = build_experiment_metadata(
            dataset_key="test", test_cases=1, config=config
        )
        assert result["rerank_provider"] is None
        assert result["rerank_model"] is None

    def test_with_reranker(self) -> None:
        config = AppConfig()
        config.reranking.model = ModelConfig(
            provider="mxbai", name="mixedbread-ai/mxbai-rerank-base-v2"
        )
        result = build_experiment_metadata(
            dataset_key="test", test_cases=1, config=config
        )
        assert result["rerank_provider"] == "mxbai"
        assert result["rerank_model"] == "mixedbread-ai/mxbai-rerank-base-v2"


class TestResolveDataset:
    def test_valid_dataset(self) -> None:
        spec = _resolve_dataset("repliqa")
        assert spec.key == "repliqa"

    def test_case_insensitive(self) -> None:
        spec = _resolve_dataset("REPLIQA")
        assert spec.key == "repliqa"

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Unknown dataset 'nonexistent'"):
            _resolve_dataset("nonexistent")

    def test_error_lists_valid_datasets(self) -> None:
        with pytest.raises(typer.BadParameter, match="repliqa"):
            _resolve_dataset("nonexistent")


class TestLoadConfig:
    def test_explicit_path(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("search:\n  limit: 42\n")
        config = _load_config(config_file)
        assert config.search.limit == 42

    def test_explicit_path_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(typer.BadParameter, match="Config file not found"):
            _load_config(tmp_path / "nonexistent.yaml")

    def test_none_falls_back_to_find_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "haiku.rag.yaml"
        config_file.write_text("search:\n  limit: 99\n")
        with patch("evaluations.benchmark.find_config_file", return_value=config_file):
            config = _load_config(None)
        assert config.search.limit == 99

    def test_none_no_config_uses_defaults(self) -> None:
        with patch("evaluations.benchmark.find_config_file", return_value=None):
            config = _load_config(None)
        assert config == AppConfig()

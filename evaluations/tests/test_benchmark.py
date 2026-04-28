from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import typer

from evaluations.benchmark import (
    _load_config,
    _resolve_dataset,
    build_experiment_metadata,
    evaluate_dataset,
    run_qa_benchmark,
)
from evaluations.config import DatasetSpec
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


class TestRunQaBenchmarkJudgeModel:
    def _make_spec(self) -> DatasetSpec:
        return DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=lambda idx, doc: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )

    @pytest.mark.asyncio
    async def test_uses_custom_judge_model(self, tmp_path: Path) -> None:
        custom_judge = ModelConfig(provider="openai", name="gpt-4o")

        with (
            patch("evaluations.benchmark.get_model") as mock_get_model,
            patch("evaluations.benchmark.HaikuRAG"),
            patch("evaluations.benchmark.get_qa_agent"),
        ):
            mock_get_model.return_value = "fake-model"
            await run_qa_benchmark(
                self._make_spec(),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                judge_model=custom_judge,
            )

        mock_get_model.assert_called_once_with(custom_judge, AppConfig())

    @pytest.mark.asyncio
    async def test_defaults_to_judge_model_config(self, tmp_path: Path) -> None:
        with (
            patch("evaluations.benchmark.get_model") as mock_get_model,
            patch("evaluations.benchmark.HaikuRAG"),
            patch("evaluations.benchmark.get_qa_agent"),
        ):
            mock_get_model.return_value = "fake-model"
            await run_qa_benchmark(
                self._make_spec(),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
            )

        mock_get_model.assert_called_once_with(AppConfig().qa.model, AppConfig())


class TestEvaluateDatasetJudgeModel:
    @pytest.mark.asyncio
    async def test_threads_judge_model_to_qa_benchmark(self) -> None:
        custom_judge = ModelConfig(
            provider="anthropic", name="claude-sonnet-4-20250514"
        )

        with patch(
            "evaluations.benchmark.run_qa_benchmark", new_callable=AsyncMock
        ) as mock_qa:
            await evaluate_dataset(
                spec=DatasetSpec(
                    key="test",
                    db_filename="test.lancedb",
                    document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    document_mapper=lambda doc: None,
                    qa_loader=lambda: [],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    qa_case_builder=lambda idx, doc: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                ),
                config=AppConfig(),
                skip_db=True,
                skip_retrieval=True,
                skip_qa=False,
                limit=None,
                name=None,
                db_path=None,
                judge_model=custom_judge,
            )

        mock_qa.assert_called_once()
        assert mock_qa.call_args[1]["judge_model"] is custom_judge


class TestExperimentMetadataTargets:
    def test_default_target_is_qa(self) -> None:
        result = build_experiment_metadata(
            dataset_key="test", test_cases=1, config=AppConfig()
        )
        assert result["target"] == "qa"
        assert "skill_provider" not in result
        assert "skill_model" not in result

    def test_skill_target_includes_skill_config(self) -> None:
        skill = ModelConfig(provider="ollama", name="gpt-oss-large", temperature=0.2)
        result = build_experiment_metadata(
            dataset_key="test",
            test_cases=1,
            config=AppConfig(),
            target="rag-skill",
            skill_config=skill,
        )
        assert result["target"] == "rag-skill"
        assert result["skill_provider"] == "ollama"
        assert result["skill_model"] == "gpt-oss-large"
        assert result["skill_temperature"] == 0.2


class TestEvaluateDatasetTarget:
    def _spec(self) -> DatasetSpec:
        return DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=lambda idx, doc: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )

    @pytest.mark.asyncio
    async def test_threads_target_and_skill_model(self) -> None:
        skill = ModelConfig(provider="ollama", name="gpt-oss")
        with patch(
            "evaluations.benchmark.run_qa_benchmark", new_callable=AsyncMock
        ) as mock_qa:
            await evaluate_dataset(
                spec=self._spec(),
                config=AppConfig(),
                skip_db=True,
                skip_retrieval=True,
                skip_qa=False,
                limit=None,
                name=None,
                db_path=None,
                target="rag-skill",
                skill_model=skill,
            )

        mock_qa.assert_called_once()
        assert mock_qa.call_args[1]["target"] == "rag-skill"
        assert mock_qa.call_args[1]["skill_model"] is skill

    @pytest.mark.asyncio
    async def test_default_target_is_qa(self) -> None:
        with patch(
            "evaluations.benchmark.run_qa_benchmark", new_callable=AsyncMock
        ) as mock_qa:
            await evaluate_dataset(
                spec=self._spec(),
                config=AppConfig(),
                skip_db=True,
                skip_retrieval=True,
                skip_qa=False,
                limit=None,
                name=None,
                db_path=None,
            )
        assert mock_qa.call_args[1]["target"] == "qa"
        assert mock_qa.call_args[1]["skill_model"] is None


class TestRunQaBenchmarkSkillTarget:
    def _spec(self, tmp_path: Path) -> DatasetSpec:
        return DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=lambda idx, doc: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )

    @pytest.mark.asyncio
    async def test_rag_skill_target_uses_run_skill_question(
        self, tmp_path: Path
    ) -> None:
        from evaluations.skill_runner import SkillRunResult

        skill_run = AsyncMock(return_value=SkillRunResult(answer="from skill"))
        with (
            patch("evaluations.benchmark.get_model") as mock_get_model,
            patch(
                "evaluations.benchmark.run_skill_question", new=skill_run
            ) as mock_run_skill,
            patch("evaluations.benchmark.HaikuRAG") as mock_haiku,
        ):
            mock_get_model.return_value = "fake-model"
            await run_qa_benchmark(
                self._spec(tmp_path),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                target="rag-skill",
            )

        # When target is rag-skill, HaikuRAG context manager is NOT entered
        # (the skill manages its own client via lifespan).
        mock_haiku.assert_not_called()
        # skill model defaults to qa.model when not provided
        skill_call = mock_get_model.call_args_list[-1]
        assert skill_call[0][0] == AppConfig().qa.model
        assert mock_run_skill is skill_run

    @pytest.mark.asyncio
    async def test_analysis_skill_target_resolves_factory(self, tmp_path: Path) -> None:
        from evaluations.benchmark import _skill_factory_for_target
        from haiku.rag.skills.analysis import create_skill as analysis_factory
        from haiku.rag.skills.rag import create_skill as rag_factory

        assert _skill_factory_for_target("rag-skill") is rag_factory
        assert _skill_factory_for_target("analysis-skill") is analysis_factory
        with pytest.raises(ValueError, match="not a skill target"):
            _skill_factory_for_target("qa")  # type: ignore[arg-type]

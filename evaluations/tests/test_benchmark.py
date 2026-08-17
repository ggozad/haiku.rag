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


def _stub_spec(**overrides) -> DatasetSpec:
    """A DatasetSpec whose loaders/mappers are inert, for tests that only
    exercise the surrounding plumbing."""
    return DatasetSpec(
        key="test",
        db_filename="test.lancedb",
        document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        document_mapper=lambda doc: None,
        qa_loader=lambda: [],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        qa_case_builder=lambda idx, doc: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        **overrides,
    )


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

    def test_records_extra_body(self) -> None:
        config = AppConfig()
        config.qa.model.extra_body = {"top_k": 5}
        judge = ModelConfig(
            provider="openai",
            name="qwen",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        capability = ModelConfig(
            provider="openai", name="gemma", extra_body={"min_p": 0}
        )
        result = build_experiment_metadata(
            dataset_key="test",
            test_cases=1,
            config=config,
            judge_config=judge,
            capability_config=capability,
        )

        assert result["qa_extra_body"] == {"top_k": 5}
        assert result["judge_extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": True}
        }
        assert result["capability_extra_body"] == {"min_p": 0}

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
            provider="vllm", name="Qwen/Qwen3-Reranker-4B"
        )
        result = build_experiment_metadata(
            dataset_key="test", test_cases=1, config=config
        )
        assert result["rerank_provider"] == "vllm"
        assert result["rerank_model"] == "Qwen/Qwen3-Reranker-4B"


class TestResolveDataset:
    def test_valid_dataset(self) -> None:
        spec = _resolve_dataset("hotpotqa")
        assert spec.key == "hotpotqa"

    def test_case_insensitive(self) -> None:
        spec = _resolve_dataset("HOTPOTQA")
        assert spec.key == "hotpotqa"

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(typer.BadParameter, match="Unknown dataset 'nonexistent'"):
            _resolve_dataset("nonexistent")

    def test_error_lists_valid_datasets(self) -> None:
        with pytest.raises(typer.BadParameter, match="hotpotqa"):
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
            patch(
                "evaluations.benchmark.run_capability_question", new_callable=AsyncMock
            ),
        ):
            mock_get_model.return_value = "fake-model"
            await run_qa_benchmark(
                self._make_spec(),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                judge_model=custom_judge,
            )

        mock_get_model.assert_any_call(custom_judge, AppConfig())

    @pytest.mark.asyncio
    async def test_defaults_to_pinned_judge_model(self, tmp_path: Path) -> None:
        from evaluations.benchmark import DEFAULT_JUDGE_MODEL

        with (
            patch("evaluations.benchmark.get_model") as mock_get_model,
            patch(
                "evaluations.benchmark.run_capability_question", new_callable=AsyncMock
            ),
        ):
            mock_get_model.return_value = "fake-model"
            await run_qa_benchmark(
                self._make_spec(),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
            )

        mock_get_model.assert_any_call(DEFAULT_JUDGE_MODEL, AppConfig())

    def test_pinned_judge_avoids_greedy_decoding(self) -> None:
        from evaluations.benchmark import DEFAULT_JUDGE_MODEL

        assert DEFAULT_JUDGE_MODEL.temperature == 0.6
        assert DEFAULT_JUDGE_MODEL.max_tokens == 16384
        assert DEFAULT_JUDGE_MODEL.extra_body == {"top_p": 0.95}


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
    def test_default_target_is_rag_capability(self) -> None:
        result = build_experiment_metadata(
            dataset_key="test", test_cases=1, config=AppConfig()
        )
        assert result["target"] == "rag-capability"
        assert "capability_provider" not in result
        assert "capability_model" not in result

    def test_capability_target_includes_capability_config(self) -> None:
        capability = ModelConfig(
            provider="ollama", name="gpt-oss-large", temperature=0.2
        )
        result = build_experiment_metadata(
            dataset_key="test",
            test_cases=1,
            config=AppConfig(),
            target="rag-capability",
            capability_config=capability,
        )
        assert result["target"] == "rag-capability"
        assert result["capability_provider"] == "ollama"
        assert result["capability_model"] == "gpt-oss-large"
        assert result["capability_temperature"] == 0.2


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
    async def test_threads_target_and_capability_model(self) -> None:
        capability = ModelConfig(provider="ollama", name="gpt-oss")
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
                target="rag-capability",
                capability_model=capability,
            )

        mock_qa.assert_called_once()
        assert mock_qa.call_args[1]["target"] == "rag-capability"
        assert mock_qa.call_args[1]["capability_model"] is capability

    @pytest.mark.asyncio
    async def test_default_target_is_rag_capability(self) -> None:
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
        assert mock_qa.call_args[1]["target"] == "rag-capability"
        assert mock_qa.call_args[1]["capability_model"] is None


class TestRunQaBenchmarkCapabilityTarget:
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
    async def test_rag_capability_target_uses_run_capability_question(
        self, tmp_path: Path
    ) -> None:
        from evaluations.capability_runner import CapabilityRunResult

        capability_run = AsyncMock(
            return_value=CapabilityRunResult(answer="from capability")
        )
        with (
            patch("evaluations.benchmark.get_model") as mock_get_model,
            patch(
                "evaluations.benchmark.run_capability_question", new=capability_run
            ) as mock_run_capability,
            patch("evaluations.benchmark.HaikuRAG") as mock_haiku,
        ):
            mock_get_model.return_value = "fake-model"
            await run_qa_benchmark(
                self._spec(tmp_path),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                target="rag-capability",
            )

        # When target is rag-capability, HaikuRAG context manager is NOT entered
        # (the capability manages its own client via lifespan).
        mock_haiku.assert_not_called()
        # capability model defaults to qa.model when not provided
        capability_call = mock_get_model.call_args_list[-1]
        assert capability_call[0][0] == AppConfig().qa.model
        assert mock_run_capability is capability_run

    @pytest.mark.asyncio
    async def test_analysis_capability_target_resolves_factory(
        self, tmp_path: Path
    ) -> None:
        from evaluations.benchmark import _capability_factory_for_target
        from haiku.rag.capabilities.analysis import (
            create_capability as analysis_factory,
        )
        from haiku.rag.capabilities.rag import create_capability as rag_factory

        assert _capability_factory_for_target("rag-capability") is rag_factory
        assert _capability_factory_for_target("analysis-capability") is analysis_factory
        with pytest.raises(ValueError, match="not a capability target"):
            _capability_factory_for_target("unknown")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


class TestCitationEvaluatorWiring:
    def test_returns_map_twin_for_map_evaluator(self) -> None:
        from evaluations.benchmark import _citation_evaluator_for
        from evaluations.evaluators import CitationMAPEvaluator, MAPEvaluator

        result = _citation_evaluator_for(MAPEvaluator())
        assert isinstance(result, CitationMAPEvaluator)

    def test_returns_none_for_no_evaluator(self) -> None:
        from evaluations.benchmark import _citation_evaluator_for

        assert _citation_evaluator_for(None) is None


class TestAttachRelevantUris:
    def test_joins_by_question(self) -> None:
        from pydantic_evals import Case

        from evaluations.benchmark import _attach_relevant_uris
        from evaluations.config import RetrievalSample
        from evaluations.evaluators import MAPEvaluator

        cases: list[Case[str, str, dict]] = [
            Case(name="c1", inputs="What is X?", expected_output="X is a thing"),
            Case(
                name="c2",
                inputs="What is Y?",
                expected_output="Y is another",
                metadata={"existing": "value"},
            ),
            Case(
                name="c3",
                inputs="What is Z?",
                expected_output="not in retrieval set",
            ),
        ]

        spec = DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=lambda idx, doc: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            retrieval_loader=lambda: [  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                {"q": "What is X?", "uris": ("uri-x",)},
                {"q": "What is Y?", "uris": ("uri-y1", "uri-y2")},
            ],
            retrieval_mapper=lambda d: RetrievalSample(
                question=d["q"], expected_uris=d["uris"]
            ),
            retrieval_evaluator=MAPEvaluator(),
        )

        _attach_relevant_uris(cases, spec, limit=None)

        assert cases[0].metadata == {"relevant_uris": ["uri-x"]}
        assert cases[1].metadata == {
            "existing": "value",
            "relevant_uris": ["uri-y1", "uri-y2"],
        }
        # case c3 has no matching retrieval sample — metadata untouched
        assert cases[2].metadata is None

    def test_no_op_without_retrieval_loader(self) -> None:
        from pydantic_evals import Case

        from evaluations.benchmark import _attach_relevant_uris

        cases: list[Case[str, str, dict]] = [
            Case(name="c1", inputs="q", expected_output="a"),
        ]
        spec = DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=lambda idx, doc: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )
        _attach_relevant_uris(cases, spec, limit=None)
        assert cases[0].metadata is None


class TestFilterQaCorpus:
    def test_keeps_only_matching_ids(self) -> None:
        from datasets import Dataset

        from evaluations.benchmark import _filter_qa_corpus

        corpus = Dataset.from_list(
            [{"id": "a", "q": 1}, {"id": "b", "q": 2}, {"id": "c", "q": 3}]
        )
        out = _filter_qa_corpus(corpus, {"a", "c"})
        assert [r["id"] for r in out] == ["a", "c"]

    def test_none_returns_corpus_unchanged(self) -> None:
        from datasets import Dataset

        from evaluations.benchmark import _filter_qa_corpus

        corpus = Dataset.from_list([{"id": "a"}])
        assert _filter_qa_corpus(corpus, None) is corpus


class TestLoadCaseIds:
    def test_reads_strips_and_drops_blanks(self, tmp_path: Path) -> None:
        from evaluations.benchmark import _load_case_ids

        f = tmp_path / "ids.txt"
        f.write_text("finqa_dev_16\n  finqa_dev_66 \n\n\nfinqa_dev_113\n")
        assert _load_case_ids(f) == {"finqa_dev_16", "finqa_dev_66", "finqa_dev_113"}

    def test_none_path_returns_none(self) -> None:
        from evaluations.benchmark import _load_case_ids

        assert _load_case_ids(None) is None


class TestRetrievalTarget:
    def _spec(self) -> DatasetSpec:
        from evaluations.config import RetrievalSample
        from evaluations.evaluators import MAPEvaluator

        return _stub_spec(
            retrieval_loader=lambda: [{"q": "What is X?", "uris": ("uri-x",)}],
            retrieval_mapper=lambda d: RetrievalSample(
                question=d["q"], expected_uris=d["uris"]
            ),
            retrieval_evaluator=MAPEvaluator(),
        )

    @pytest.mark.asyncio
    async def test_scores_from_search_results_without_reading_documents(
        self, tmp_path: Path
    ) -> None:
        from haiku.rag.store.models.chunk import SearchResult

        from evaluations.benchmark import run_retrieval_benchmark

        searches: list[dict] = []

        class FakeRag:
            async def search(self, **kwargs) -> list[SearchResult]:
                searches.append(kwargs)
                return [
                    SearchResult(
                        content="x",
                        score=1.0,
                        document_id="doc-1",
                        document_uri="uri-x",
                    )
                ]

            async def get_document_by_id(self, document_id: str) -> None:
                raise AssertionError(
                    "retrieval scoring must not read whole document rows"
                )

        fake = FakeRag()
        with patch("evaluations.benchmark.HaikuRAG") as mock_haiku:
            mock_haiku.return_value.__aenter__.return_value = fake
            result = await run_retrieval_benchmark(
                self._spec(), AppConfig(), db_path=tmp_path / "test.lancedb"
            )

        assert result is not None
        assert result["map"] == 1.0
        assert searches[0]["include_images"] is False

    @pytest.mark.asyncio
    async def test_ranks_each_document_once(self, tmp_path: Path) -> None:
        from haiku.rag.store.models.chunk import SearchResult

        from evaluations.benchmark import run_retrieval_benchmark

        def _result(uri: str, score: float) -> SearchResult:
            return SearchResult(content="x", score=score, document_uri=uri)

        class FakeRag:
            async def search(self, **kwargs) -> list[SearchResult]:
                return [
                    _result("uri-other", 1.0),
                    _result("uri-x", 0.9),
                    _result("uri-other", 0.8),
                    _result("uri-x", 0.7),
                ]

        with patch("evaluations.benchmark.HaikuRAG") as mock_haiku:
            mock_haiku.return_value.__aenter__.return_value = FakeRag()
            result = await run_retrieval_benchmark(
                self._spec(), AppConfig(), db_path=tmp_path / "test.lancedb"
            )

        # uri-x is the only relevant document and ranks second of two
        assert result is not None
        assert result["map"] == 0.5


class TestResolveSearchFilter:
    def test_dataset_filter_used_when_no_override(self) -> None:
        from evaluations.benchmark import resolve_search_filter

        spec = _stub_spec(search_filter="uri LIKE '%arxiv%'")
        assert resolve_search_filter(spec, None) == "uri LIKE '%arxiv%'"

    def test_override_wins(self) -> None:
        from evaluations.benchmark import resolve_search_filter

        spec = _stub_spec(search_filter="uri LIKE '%arxiv%'")
        assert resolve_search_filter(spec, "uri LIKE '%.pdf'") == "uri LIKE '%.pdf'"

    def test_empty_override_clears_dataset_filter(self) -> None:
        """`--filter ""` runs a filtered dataset against the whole database."""
        from evaluations.benchmark import resolve_search_filter

        spec = _stub_spec(search_filter="uri LIKE '%arxiv%'")
        assert resolve_search_filter(spec, "") is None

    def test_none_when_neither_is_set(self) -> None:
        from evaluations.benchmark import resolve_search_filter

        assert resolve_search_filter(_stub_spec(), None) is None


class TestSearchFilterThreading:
    """The resolved filter must reach both benchmark phases, so retrieval and
    QA score the same subset of the database."""

    def test_metadata_records_filter(self) -> None:
        result = build_experiment_metadata(
            dataset_key="test",
            test_cases=1,
            config=AppConfig(),
            search_filter="uri LIKE '%arxiv%'",
        )
        assert result["search_filter"] == "uri LIKE '%arxiv%'"

    def test_metadata_filter_is_none_when_unset(self) -> None:
        result = build_experiment_metadata(
            dataset_key="test", test_cases=1, config=AppConfig()
        )
        assert result["search_filter"] is None

    @pytest.mark.asyncio
    async def test_retrieval_search_receives_filter(self, tmp_path: Path) -> None:
        from haiku.rag.store.models.chunk import SearchResult

        from evaluations.benchmark import run_retrieval_benchmark
        from evaluations.config import RetrievalSample
        from evaluations.evaluators import MAPEvaluator

        searches: list[dict] = []

        class FakeRag:
            async def search(self, **kwargs) -> list[SearchResult]:
                searches.append(kwargs)
                return [SearchResult(content="x", score=1.0, document_uri="uri-x")]

        spec = _stub_spec(
            retrieval_loader=lambda: [{"q": "What is X?", "uris": ("uri-x",)}],
            retrieval_mapper=lambda d: RetrievalSample(
                question=d["q"], expected_uris=d["uris"]
            ),
            retrieval_evaluator=MAPEvaluator(),
        )

        with patch("evaluations.benchmark.HaikuRAG") as mock_haiku:
            mock_haiku.return_value.__aenter__.return_value = FakeRag()
            await run_retrieval_benchmark(
                spec,
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                search_filter="uri LIKE '%arxiv%'",
            )

        assert searches[0]["filter"] == "uri LIKE '%arxiv%'"

    @pytest.mark.asyncio
    async def test_qa_capability_run_receives_filter(self, tmp_path: Path) -> None:
        from pydantic_evals import Case

        from evaluations.capability_runner import CapabilityRunResult
        from evaluations.evaluators import NumberMatchEvaluator

        # A deterministic evaluator, so no judge model is constructed.
        spec = DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [{"question": "What is X?", "answer": "42"}],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=lambda idx, doc: Case(
                name=f"case-{idx}",
                inputs=doc["question"],
                expected_output=doc["answer"],
            ),
            qa_evaluator=NumberMatchEvaluator(),
        )

        with patch(
            "evaluations.benchmark.run_capability_question",
            new_callable=AsyncMock,
            return_value=CapabilityRunResult(answer="ANSWER: 42"),
        ) as mock_run:
            await run_qa_benchmark(
                spec,
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                search_filter="uri LIKE '%arxiv%'",
            )

        mock_run.assert_awaited_once()
        assert mock_run.call_args[1]["document_filter"] == "uri LIKE '%arxiv%'"

    @pytest.mark.asyncio
    async def test_evaluate_dataset_resolves_once_for_both_phases(self) -> None:
        """The dataset's own filter reaches retrieval and QA without a flag."""
        spec = _stub_spec(search_filter="""metadata LIKE '%"corpus": "orb_text"%'""")

        with (
            patch(
                "evaluations.benchmark.run_retrieval_benchmark", new_callable=AsyncMock
            ) as mock_retrieval,
            patch(
                "evaluations.benchmark.run_qa_benchmark", new_callable=AsyncMock
            ) as mock_qa,
        ):
            await evaluate_dataset(
                spec=spec,
                config=AppConfig(),
                skip_db=True,
                skip_retrieval=False,
                skip_qa=False,
                limit=None,
                name=None,
                db_path=None,
            )

        expected = """metadata LIKE '%"corpus": "orb_text"%'"""
        assert mock_retrieval.call_args[1]["search_filter"] == expected
        assert mock_qa.call_args[1]["search_filter"] == expected

    @pytest.mark.asyncio
    async def test_evaluate_dataset_override_reaches_both_phases(self) -> None:
        spec = _stub_spec(search_filter="""metadata LIKE '%"corpus": "orb_text"%'""")

        with (
            patch(
                "evaluations.benchmark.run_retrieval_benchmark", new_callable=AsyncMock
            ) as mock_retrieval,
            patch(
                "evaluations.benchmark.run_qa_benchmark", new_callable=AsyncMock
            ) as mock_qa,
        ):
            await evaluate_dataset(
                spec=spec,
                config=AppConfig(),
                skip_db=True,
                skip_retrieval=False,
                skip_qa=False,
                limit=None,
                name=None,
                db_path=None,
                search_filter="title LIKE '%paper%'",
            )

        assert mock_retrieval.call_args[1]["search_filter"] == "title LIKE '%paper%'"
        assert mock_qa.call_args[1]["search_filter"] == "title LIKE '%paper%'"


class TestEvaluateDatasetCaseIds:
    def _spec(self) -> DatasetSpec:
        return _stub_spec()

    @pytest.mark.asyncio
    async def test_threads_case_ids_to_qa_benchmark(self) -> None:
        from evaluations.benchmark import evaluate_dataset

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
                case_ids={"finqa_dev_16", "finqa_dev_66"},
            )
        mock_qa.assert_called_once()
        assert mock_qa.call_args[1]["case_ids"] == {"finqa_dev_16", "finqa_dev_66"}

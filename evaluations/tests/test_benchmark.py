from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from evaluations.benchmark import (
    _load_config,
    _resolve_dataset,
    evaluate_dataset,
)
from evaluations.experiment import build_experiment_metadata
from evaluations.qa import run_qa_benchmark
from evaluations.config import DatasetSpec, DocumentPayload
from haiku.rag.config.models import AppConfig, ModelConfig


def _stub_spec(**overrides) -> DatasetSpec:
    """A DatasetSpec whose loaders/mappers are inert, for tests that only
    exercise the surrounding plumbing. Any field can be overridden."""
    fields: dict = {
        "key": "test",
        "db_filename": "test.lancedb",
        "document_loader": lambda: None,
        "document_mapper": lambda doc: None,
        "qa_loader": lambda: [],
        "qa_case_builder": lambda idx, doc: None,
    }
    return DatasetSpec(**{**fields, **overrides})


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


class TestConversationInputDispatch:
    @pytest.mark.asyncio
    async def test_prefix_rides_as_message_history(self, tmp_path: Path) -> None:
        """A ConversationInput case reaches the capability as final question
        plus the prefix converted to message history."""
        from dataclasses import dataclass

        from pydantic_evals import Case
        from pydantic_evals.evaluators import Evaluator, EvaluatorContext

        from evaluations.capability_runner import CapabilityRunResult
        from evaluations.config import ConversationInput, Turn

        @dataclass
        class AlwaysOne(Evaluator):
            def evaluate(self, ctx: EvaluatorContext) -> float:
                return 1.0

        def build_case(idx: int, doc) -> Case:
            return Case(
                name="c1",
                inputs=ConversationInput(
                    turns=[
                        Turn(speaker="user", text="q1"),
                        Turn(speaker="agent", text="a1"),
                        Turn(speaker="user", text="q2"),
                    ]
                ),
                expected_output="ref",
            )

        spec = DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [{"id": "t1"}],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=build_case,
            qa_evaluator=AlwaysOne(),
        )

        with (
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch(
                "evaluations.qa.run_capability_question",
                new_callable=AsyncMock,
                return_value=CapabilityRunResult(answer="answer"),
            ) as run_question,
        ):
            await run_qa_benchmark(spec, AppConfig(), db_path=tmp_path / "test.lancedb")

        assert run_question.await_args is not None
        kwargs = run_question.await_args.kwargs
        assert kwargs["question"] == "q2"
        history = kwargs["message_history"]
        assert len(history) == 2
        assert history[0].parts[0].content == "q1"
        assert history[1].parts[0].content == "a1"

    @pytest.mark.asyncio
    async def test_records_citation_status_attribute(self, tmp_path: Path) -> None:
        from dataclasses import dataclass

        from pydantic_evals import Case
        from pydantic_evals.evaluators import Evaluator, EvaluatorContext

        from evaluations.capability_runner import CapabilityRunResult

        @dataclass
        class AlwaysOne(Evaluator):
            def evaluate(self, ctx: EvaluatorContext) -> float:
                return 1.0

        def build_case(idx: int, doc) -> Case:
            return Case(name="c1", inputs="q1", expected_output="ref")

        spec = DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [{"id": "t1"}],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=build_case,
            qa_evaluator=AlwaysOne(),
        )

        recorded: dict[str, object] = {}
        with (
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch(
                "evaluations.qa.set_eval_attribute",
                side_effect=lambda key, value: recorded.__setitem__(key, value),
            ),
            patch(
                "evaluations.qa.run_capability_question",
                new_callable=AsyncMock,
                return_value=CapabilityRunResult(
                    answer="answer", citation_status="ungrounded"
                ),
            ),
        ):
            await run_qa_benchmark(spec, AppConfig(), db_path=tmp_path / "test.lancedb")

        assert recorded["citation_status"] == "ungrounded"


class TestRefusalMetrics:
    def _case(self, label: str | None, refused: bool | None) -> MagicMock:
        case = MagicMock()
        case.metadata = {"answerability": label} if label is not None else {}
        case.assertions = (
            {"refused": MagicMock(value=refused)} if refused is not None else {}
        )
        return case

    def test_precision_and_recall(self) -> None:
        from evaluations.qa import _refusal_metrics

        cases = [
            self._case("UNANSWERABLE", True),  # true refusal
            self._case("UNANSWERABLE", False),  # missed refusal
            self._case("ANSWERABLE", True),  # false refusal
            self._case("ANSWERABLE", False),  # answered correctly
            self._case("PARTIAL", None),  # skipped by the judge, no assertion
            self._case(None, None),  # no label
        ]

        metrics = _refusal_metrics(cases)

        assert metrics is not None
        precision, recall, unanswerable, refusals = metrics
        assert precision == 0.5  # 1 true refusal of 2 refusals
        assert recall == 0.5  # 1 of 2 unanswerable turns refused
        assert unanswerable == 2
        assert refusals == 2

    def test_none_when_no_judged_cases(self) -> None:
        from evaluations.qa import _refusal_metrics

        assert _refusal_metrics([self._case("PARTIAL", None)]) is None


class TestLiveSummary:
    def _case(self, scores: dict[str, float | int]) -> MagicMock:
        case = MagicMock()
        case.scores = {key: MagicMock(value=value) for key, value in scores.items()}
        return case

    def test_micro_and_macro_aggregation(self) -> None:
        from evaluations.qa import _live_summary

        # Conversation A: 1/4 turns pass; B: 2/2 pass. Micro weights turns
        # (3/6); macro averages conversations ((0.25 + 1.0) / 2).
        cases = [
            self._case(
                {
                    "turn_pass_rate": 0.25,
                    "turns_passed": 1,
                    "turns_judged": 4,
                    "turns_total": 4,
                    "cited_map": 0.5,
                    "cited_eligible": 3,
                    "true_refusals": 1,
                    "false_refusals": 1,
                    "unanswerable_turns": 2,
                }
            ),
            self._case(
                {
                    "turn_pass_rate": 1.0,
                    "turns_passed": 2,
                    "turns_judged": 2,
                    "turns_total": 2,
                    "cited_map": 1.0,
                    "cited_eligible": 1,
                    "true_refusals": 0,
                    "false_refusals": 0,
                    "unanswerable_turns": 0,
                }
            ),
        ]

        failure = MagicMock()
        failure.inputs = ["fq1", "fq2", "fq3"]
        summary = _live_summary(cases, [failure])

        assert summary is not None
        assert summary["conversations"] == 2
        assert summary["conversations_attempted"] == 3
        assert summary["turns_total"] == 6
        assert summary["turns_judged"] == 6
        assert summary["turns_attempted"] == 9
        assert summary["micro_pass_rate"] == pytest.approx(0.5)
        assert summary["macro_pass_rate"] == pytest.approx(0.625)
        assert summary["cited_eligible"] == 4
        assert summary["cited_map_micro"] == pytest.approx((0.5 * 3 + 1.0 * 1) / 4)
        assert summary["cited_map_macro"] == pytest.approx(0.75)
        assert summary["refusal_precision"] == pytest.approx(0.5)
        assert summary["refusal_recall"] == pytest.approx(0.5)

    def test_none_without_scored_cases(self) -> None:
        from evaluations.qa import _live_summary

        assert _live_summary([self._case({})], []) is None

    def test_micro_rate_uses_judged_turns(self) -> None:
        from evaluations.qa import _live_summary

        cases = [
            self._case(
                {
                    "turn_pass_rate": 1.0,
                    "turns_passed": 3,
                    "turns_judged": 3,
                    "turns_total": 4,  # one turn's judge errored
                    "cited_eligible": 0,
                    "true_refusals": 0,
                    "false_refusals": 0,
                    "unanswerable_turns": 0,
                }
            )
        ]

        summary = _live_summary(cases, [])

        assert summary is not None
        assert summary["micro_pass_rate"] == 1.0
        assert summary["turns_judged"] == 3
        assert summary["turns_total"] == 4

    def test_macro_rate_excludes_fully_unjudged_conversations(self) -> None:
        """A conversation whose every turn lost its judge reports
        turn_pass_rate 0.0; treating that as a failed conversation would
        contradict the exclusion policy. It must not enter the macro average."""
        from evaluations.qa import _live_summary

        cases = [
            self._case(
                {
                    "turn_pass_rate": 1.0,
                    "turns_passed": 2,
                    "turns_judged": 2,
                    "turns_total": 2,
                    "cited_eligible": 0,
                    "true_refusals": 0,
                    "false_refusals": 0,
                    "unanswerable_turns": 0,
                }
            ),
            self._case(
                {
                    "turn_pass_rate": 0.0,
                    "turns_passed": 0,
                    "turns_judged": 0,  # total judge outage for this conversation
                    "turns_total": 8,
                    "cited_eligible": 0,
                    "true_refusals": 0,
                    "false_refusals": 0,
                    "unanswerable_turns": 0,
                }
            ),
        ]

        summary = _live_summary(cases, [])

        assert summary is not None
        assert summary["macro_pass_rate"] == pytest.approx(1.0)
        assert summary["micro_pass_rate"] == pytest.approx(1.0)
        assert summary["turns_judged"] == 2
        assert summary["turns_total"] == 10

    def test_failed_conversations_do_not_affect_rates(self) -> None:
        from evaluations.qa import _live_summary

        cases = [
            self._case(
                {
                    "turn_pass_rate": 1.0,
                    "turns_passed": 2,
                    "turns_judged": 2,
                    "turns_total": 2,
                    "cited_eligible": 0,
                    "true_refusals": 0,
                    "false_refusals": 0,
                    "unanswerable_turns": 0,
                }
            )
        ]
        failure = MagicMock()
        failure.inputs = ["fq1", "fq2"]

        summary = _live_summary(cases, [failure])

        assert summary is not None
        assert summary["micro_pass_rate"] == 1.0
        assert summary["macro_pass_rate"] == 1.0
        assert summary["conversations_attempted"] == 2
        assert summary["turns_attempted"] == 4


class TestLiveConversationDispatch:
    @pytest.mark.asyncio
    async def test_live_spec_replays_conversation(self, tmp_path: Path) -> None:
        from pydantic_evals import Case

        from evaluations.benchmark import run_live_qa_benchmark
        from evaluations.capability_runner import CapabilityRunResult

        def build_case(idx: int, doc) -> Case:
            return Case(
                name="conv1",
                inputs=["q1", "q2"],
                metadata={
                    "conversation_id": "conv1",
                    "turns": [
                        {"reference": "r1", "answerability": "ANSWERABLE"},
                        {"reference": "r2", "answerability": "ANSWERABLE"},
                    ],
                },
            )

        spec = DatasetSpec(
            key="test_live",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [{"id": "conv1"}],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=build_case,
            live=True,
            compaction=True,
        )

        turn_results = [
            CapabilityRunResult(answer="a1", cited_uris=["u1"]),
            CapabilityRunResult(answer="a2", cited_uris=[]),
        ]
        with (
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch(
                "evaluations.qa.run_capability_conversation",
                new_callable=AsyncMock,
                return_value=turn_results,
            ) as run_conversation,
            patch(
                "evaluations.evaluators.conversation.judge_input_output_expected",
                new_callable=AsyncMock,
                return_value=MagicMock(score=None, pass_=True, reason=None),
            ),
            patch(
                "evaluations.evaluators.conversation.judge_output",
                new_callable=AsyncMock,
                return_value=MagicMock(score=None, pass_=False, reason=None),
            ),
        ):
            await run_live_qa_benchmark(
                spec,
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                document_filter="uri = 'manual.pdf'",
            )

        assert run_conversation.await_args is not None
        assert run_conversation.await_args.kwargs["questions"] == ["q1", "q2"]
        assert run_conversation.await_args.kwargs["compaction"] is True
        assert (
            run_conversation.await_args.kwargs["document_filter"]
            == "uri = 'manual.pdf'"
        )

    @pytest.mark.asyncio
    async def test_live_records_per_turn_traffic_arrays(self, tmp_path: Path) -> None:
        """Per-turn tool traffic is recorded as question-length arrays, in the
        same list-indexed-by-turn shape as turn_cited_uris."""
        from pydantic_evals import Case

        from evaluations.benchmark import run_live_qa_benchmark
        from evaluations.capability_runner import CapabilityRunResult

        def build_case(idx: int, doc) -> Case:
            return Case(
                name="conv1",
                inputs=["q1", "q2"],
                metadata={
                    "conversation_id": "conv1",
                    "turns": [
                        {"reference": "r1", "answerability": "ANSWERABLE"},
                        {"reference": "r2", "answerability": "ANSWERABLE"},
                    ],
                },
            )

        spec = DatasetSpec(
            key="test_live",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: None,
            qa_loader=lambda: [{"id": "conv1"}],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=build_case,
            live=True,
        )

        turn_results = [
            CapabilityRunResult(
                answer="a1",
                cited_uris=["u1"],
                n_search_calls=2,
                n_rejected_searches=1,
                n_failed_tools=1,
                n_requests=4,
                citation_status="grounded",
            ),
            CapabilityRunResult(answer="a2"),
        ]
        recorded: dict[str, object] = {}

        with (
            patch("evaluations.qa.get_model", return_value="fake-model"),
            patch(
                "evaluations.qa.set_eval_attribute",
                side_effect=lambda key, value: recorded.__setitem__(key, value),
            ),
            patch(
                "evaluations.qa.run_capability_conversation",
                new_callable=AsyncMock,
                return_value=turn_results,
            ),
            patch(
                "evaluations.evaluators.conversation.judge_input_output_expected",
                new_callable=AsyncMock,
                return_value=MagicMock(score=None, pass_=True, reason=None),
            ),
            patch(
                "evaluations.evaluators.conversation.judge_output",
                new_callable=AsyncMock,
                return_value=MagicMock(score=None, pass_=False, reason=None),
            ),
        ):
            await run_live_qa_benchmark(
                spec, AppConfig(), db_path=tmp_path / "test.lancedb"
            )

        assert recorded["turn_n_search_calls"] == [2, 0]
        assert recorded["turn_n_rejected_searches"] == [1, 0]
        assert recorded["turn_n_failed_tools"] == [1, 0]
        assert recorded["turn_n_requests"] == [4, 0]
        assert recorded["turn_citation_status"] == ["grounded", None]
        questions = 2
        for key, value in recorded.items():
            if key.startswith("turn_"):
                assert isinstance(value, list) and len(value) == questions, key


class TestResolveDatasets:
    def test_all_dedupes_shared_databases(self) -> None:
        """Specs sharing a db_filename (mtrag query variants) appear once, so
        `download all`/`upload all` do not process the same DB twice."""
        from evaluations.benchmark import _resolve_datasets

        specs = _resolve_datasets("all")
        filenames = [spec.db_filename for spec in specs]
        assert len(filenames) == len(set(filenames))
        assert "mtrag_clapnq.lancedb" in filenames

    def test_single_key_not_deduped(self) -> None:
        from evaluations.benchmark import _resolve_datasets

        specs = _resolve_datasets("mtrag_clapnq_rewrite")
        assert [spec.key for spec in specs] == ["mtrag_clapnq_rewrite"]


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
            patch("evaluations.qa.get_model") as mock_get_model,
            patch("evaluations.qa.run_capability_question", new_callable=AsyncMock),
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
        from evaluations.experiment import DEFAULT_JUDGE_MODEL

        with (
            patch("evaluations.qa.get_model") as mock_get_model,
            patch("evaluations.qa.run_capability_question", new_callable=AsyncMock),
        ):
            mock_get_model.return_value = "fake-model"
            await run_qa_benchmark(
                self._make_spec(),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
            )

        mock_get_model.assert_any_call(DEFAULT_JUDGE_MODEL, AppConfig())

    def test_pinned_judge_avoids_greedy_decoding(self) -> None:
        from evaluations.experiment import DEFAULT_JUDGE_MODEL

        assert DEFAULT_JUDGE_MODEL.temperature == 0.6
        assert DEFAULT_JUDGE_MODEL.name == "qwen3.8"
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
            patch("evaluations.qa.get_model") as mock_get_model,
            patch(
                "evaluations.qa.run_capability_question", new=capability_run
            ) as mock_run_capability,
        ):
            mock_get_model.return_value = "fake-model"
            await run_qa_benchmark(
                self._spec(tmp_path),
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                target="rag-capability",
            )

        # The capability manages its own client, so the QA runner never opens
        # one — it has no HaikuRAG reference to open.
        import evaluations.qa as qa_module

        assert not hasattr(qa_module, "HaikuRAG")
        # capability model defaults to qa.model when not provided
        assert any(
            call[0][0] == AppConfig().qa.model for call in mock_get_model.call_args_list
        )
        assert mock_run_capability is capability_run

    @pytest.mark.asyncio
    async def test_analysis_capability_target_resolves_factory(
        self, tmp_path: Path
    ) -> None:
        from evaluations.qa import _capability_factory_for_target
        from haiku.rag.capabilities.analysis import (
            create_capability as analysis_factory,
        )
        from haiku.rag.capabilities.rag import create_capability as rag_factory

        assert _capability_factory_for_target("rag-capability") is rag_factory
        assert _capability_factory_for_target("analysis-capability") is analysis_factory
        with pytest.raises(ValueError, match="not a capability target"):
            _capability_factory_for_target("unknown")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


class TestCitationEvaluatorWiring:
    def test_specs_with_retrieval_declare_citation_evaluator(self) -> None:
        """Citation scoring is declared per spec, not inferred: every dataset
        that scores retrieval also scores citations."""
        from evaluations.datasets import DATASETS
        from evaluations.evaluators import CitationMAPEvaluator

        for spec in DATASETS.values():
            if spec.retrieval_evaluators:
                assert isinstance(spec.citation_evaluator, CitationMAPEvaluator), (
                    spec.key
                )


class TestBatchedIngest:
    def _rag(
        self,
        complete_uris: list[str] | None = None,
        chunkless_uris: list[str] | None = None,
    ) -> MagicMock:
        complete_uris = complete_uris or []
        chunkless_uris = chunkless_uris or []

        def _table(rows: list[dict]) -> MagicMock:
            table = MagicMock()
            table.query.return_value.select.return_value.to_list = AsyncMock(
                return_value=rows
            )
            return table

        rag = MagicMock()
        rag.store.document_meta_table = _table(
            [{"id": f"id-{uri}", "uri": uri} for uri in complete_uris + chunkless_uris]
        )
        rag.store.chunks_table = _table(
            [{"document_id": f"id-{uri}"} for uri in complete_uris]
        )
        rag.convert = AsyncMock(side_effect=lambda content, **kw: f"docling:{content}")
        rag.chunk = AsyncMock(return_value=[])
        rag.import_documents = AsyncMock()
        rag.delete_document = AsyncMock()
        return rag

    def _spec(self) -> DatasetSpec:
        return DatasetSpec(
            key="test",
            db_filename="test.lancedb",
            document_loader=lambda: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            document_mapper=lambda doc: (
                None
                if doc["uri"] == "bad"
                else DocumentPayload(uri=doc["uri"], content=f"text {doc['uri']}")
            ),
            qa_loader=lambda: [],  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            qa_case_builder=lambda idx, doc: None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        )

    @pytest.mark.asyncio
    async def test_imports_in_bounded_batches(self) -> None:
        from evaluations.population import _ingest_batched

        rag = self._rag()
        corpus = [{"uri": f"u{i}"} for i in range(5)]

        await _ingest_batched(rag, self._spec(), corpus, batch_size=2)

        batch_uris = [
            [imp.uri for imp in call.args[0]]
            for call in rag.import_documents.call_args_list
        ]
        assert batch_uris == [["u0", "u1"], ["u2", "u3"], ["u4"]]

    @pytest.mark.asyncio
    async def test_resume_skips_complete_uris(self) -> None:
        from evaluations.population import _ingest_batched

        rag = self._rag(complete_uris=["u0", "u2"])
        corpus = [{"uri": f"u{i}"} for i in range(4)]

        await _ingest_batched(rag, self._spec(), corpus, batch_size=10)

        (batch,), _ = rag.import_documents.call_args
        assert [imp.uri for imp in batch] == ["u1", "u3"]
        assert rag.convert.await_count == 2
        rag.delete_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resume_reimports_chunkless_documents(self) -> None:
        """A crash between the document and chunk writes leaves a document
        without chunks; resume must delete and re-import it, not skip it."""
        from evaluations.population import _ingest_batched

        rag = self._rag(complete_uris=["u0"], chunkless_uris=["u1"])
        corpus = [{"uri": "u0"}, {"uri": "u1"}]

        await _ingest_batched(rag, self._spec(), corpus, batch_size=10)

        rag.delete_document.assert_awaited_once_with("id-u1")
        (batch,), _ = rag.import_documents.call_args
        assert [imp.uri for imp in batch] == ["u1"]

    @pytest.mark.asyncio
    async def test_unmapped_documents_skipped(self) -> None:
        from evaluations.population import _ingest_batched

        rag = self._rag()
        corpus = [{"uri": "u0"}, {"uri": "bad"}, {"uri": "u1"}]

        await _ingest_batched(rag, self._spec(), corpus, batch_size=10)

        (batch,), _ = rag.import_documents.call_args
        assert [imp.uri for imp in batch] == ["u0", "u1"]


class TestAttachRelevantUris:
    def test_joins_by_question(self) -> None:
        from pydantic_evals import Case

        from evaluations.qa import _attach_relevant_uris
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
            retrieval_evaluators=[MAPEvaluator()],
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

        from evaluations.qa import _attach_relevant_uris

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

        from evaluations.qa import _filter_qa_corpus

        corpus = Dataset.from_list(
            [{"id": "a", "q": 1}, {"id": "b", "q": 2}, {"id": "c", "q": 3}]
        )
        out = _filter_qa_corpus(corpus, {"a", "c"})
        assert [r["id"] for r in out] == ["a", "c"]

    def test_none_returns_corpus_unchanged(self) -> None:
        from datasets import Dataset

        from evaluations.qa import _filter_qa_corpus

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
            retrieval_evaluators=[MAPEvaluator()],
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
        with patch("evaluations.retrieval.HaikuRAG") as mock_haiku:
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

        with patch("evaluations.retrieval.HaikuRAG") as mock_haiku:
            mock_haiku.return_value.__aenter__.return_value = FakeRag()
            result = await run_retrieval_benchmark(
                self._spec(), AppConfig(), db_path=tmp_path / "test.lancedb"
            )

        # uri-x is the only relevant document and ranks second of two
        assert result is not None
        assert result["map"] == 0.5


class TestDocumentFilterThreading:
    """The filter must reach both benchmark phases, so retrieval and QA score
    the same subset of the database."""

    def test_metadata_records_filter(self) -> None:
        result = build_experiment_metadata(
            dataset_key="test",
            test_cases=1,
            config=AppConfig(),
            document_filter="uri LIKE '%arxiv%'",
        )
        assert result["document_filter"] == "uri LIKE '%arxiv%'"

    def test_metadata_filter_is_none_when_unset(self) -> None:
        result = build_experiment_metadata(
            dataset_key="test", test_cases=1, config=AppConfig()
        )
        assert result["document_filter"] is None

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
            retrieval_evaluators=[MAPEvaluator()],
        )

        with patch("evaluations.retrieval.HaikuRAG") as mock_haiku:
            mock_haiku.return_value.__aenter__.return_value = FakeRag()
            await run_retrieval_benchmark(
                spec,
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                document_filter="uri LIKE '%arxiv%'",
            )

        assert searches[0]["filter"] == "uri LIKE '%arxiv%'"

    @pytest.mark.asyncio
    async def test_qa_capability_run_receives_filter(self, tmp_path: Path) -> None:
        from pydantic_evals import Case

        from evaluations.capability_runner import CapabilityRunResult
        from evaluations.evaluators import NumberMatchEvaluator

        # A deterministic evaluator, so no judge model is constructed.
        spec = _stub_spec(
            qa_loader=lambda: [{"question": "What is X?", "answer": "42"}],
            qa_case_builder=lambda idx, doc: Case(
                name=f"case-{idx}",
                inputs=doc["question"],
                expected_output=doc["answer"],
            ),
            qa_evaluator=NumberMatchEvaluator(),
        )

        with patch(
            "evaluations.qa.run_capability_question",
            new_callable=AsyncMock,
            return_value=CapabilityRunResult(answer="ANSWER: 42"),
        ) as mock_run:
            await run_qa_benchmark(
                spec,
                AppConfig(),
                db_path=tmp_path / "test.lancedb",
                document_filter="uri LIKE '%arxiv%'",
            )

        mock_run.assert_awaited_once()
        assert mock_run.call_args[1]["document_filter"] == "uri LIKE '%arxiv%'"

    @pytest.mark.asyncio
    async def test_evaluate_dataset_passes_filter_to_both_phases(self) -> None:
        expected = """metadata LIKE '%"corpus": "orb_text"%'"""

        with (
            patch(
                "evaluations.benchmark.run_retrieval_benchmark", new_callable=AsyncMock
            ) as mock_retrieval,
            patch(
                "evaluations.benchmark.run_qa_benchmark", new_callable=AsyncMock
            ) as mock_qa,
        ):
            await evaluate_dataset(
                spec=_stub_spec(),
                config=AppConfig(),
                skip_db=True,
                skip_retrieval=False,
                skip_qa=False,
                limit=None,
                name=None,
                db_path=None,
                document_filter=expected,
            )

        assert mock_retrieval.call_args[1]["document_filter"] == expected
        assert mock_qa.call_args[1]["document_filter"] == expected


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


def test_a_case_filter_matching_nothing_raises():
    """An id set that matches no case raises."""
    import pytest
    from datasets import Dataset

    from evaluations.qa import _filter_qa_corpus

    corpus = Dataset.from_list([{"query_id": "a"}, {"query_id": "b"}])

    with pytest.raises(ValueError, match="matched none"):
        _filter_qa_corpus(corpus, {"a"})


def test_a_case_filter_that_matches_keeps_those_rows():
    from datasets import Dataset

    from evaluations.qa import _filter_qa_corpus

    corpus = Dataset.from_list([{"id": "a"}, {"id": "b"}, {"id": "c"}])

    assert _filter_qa_corpus(corpus, {"a", "c"})["id"] == ["a", "c"]


async def test_population_refuses_a_configured_set():
    """Population writes to one database and refuses a configured set."""
    import pytest
    from haiku.rag.config.models import AppConfig, LanceDBConfig

    from evaluations.benchmark import evaluate_dataset
    from evaluations.datasets import DATASETS

    spec = next(iter(DATASETS.values()))
    config = AppConfig(
        lancedb=LanceDBConfig(databases={"a": "/a.lancedb", "b": "/b.lancedb"})
    )

    with pytest.raises(ValueError, match="--skip-db"):
        await evaluate_dataset(
            spec,
            config,
            skip_db=False,
            skip_retrieval=True,
            skip_qa=True,
            limit=None,
            name=None,
            db_path=None,
        )

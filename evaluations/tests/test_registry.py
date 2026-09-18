import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from evaluations.arm import load_arm
from evaluations.experiment import DEFAULT_JUDGE_MODEL
from evaluations.registry import ArmRecord, Registry, launch_record
from haiku.rag.config.models import AppConfig, ModelConfig


def _record(**overrides: Any) -> ArmRecord:
    fields: dict[str, Any] = {
        "name": "frames-main",
        "dataset": "frames",
        "kind": "qa",
        "status": "launched",
        "started_at": "2026-01-01T00:00:00+00:00",
        "source": "harness",
        "git_sha": "a" * 40,
        "config_path": "/arms/frames.yaml",
        "config_hash": "c" * 64,
        "db_path": "/dbs/frames.lancedb",
        "db_documents": 2500,
        "db_chunks": 464150,
        "db_embedder": "vllm/nvidia/llama-nemotron-embed-vl-1b-v2 dim 2048",
        "db_version": "0.86.0",
        "capability_model": "Inferact/Muse-Glimmer-30B-NVFP4-W4A4",
        "capability_endpoint": "http://vllm:11450",
        "judge_model": "Inferact/Qwen3.8-27B-NVFP4",
        "reranker": "nvidia/llama-nemotron-rerank-vl-1b-v2",
        "embedder": "vllm/nvidia/llama-nemotron-embed-vl-1b-v2 dim 2048",
        "limit_cases": 150,
        "operator": "evaluations",
    }
    fields.update(overrides)
    return ArmRecord(**fields)


class TestRegistry:
    def test_a_launch_row_round_trips(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        record = _record()
        registry.register_launch(record)
        assert registry.get("frames-main") == record
        assert registry.get("missing") is None

    def test_names_are_unique(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        with pytest.raises(ValueError, match="frames-main"):
            registry.register_launch(_record())

    def test_completion_fills_the_result_fields(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        registry.complete(
            "frames-main",
            trace_id="0" * 32,
            cases=150,
            accuracy=0.78,
            cite_rate=0.9,
            cited_map=0.41,
            aborts=2,
            wall_seconds=23400.0,
            ended_at="2026-01-01T06:30:00+00:00",
        )
        record = registry.get("frames-main")
        assert record is not None
        assert record.status == "valid"
        assert record.trace_id == "0" * 32
        assert record.cases == 150
        assert record.accuracy == 0.78
        assert record.aborts == 2
        assert record.ended_at == "2026-01-01T06:30:00+00:00"

    def test_void_carries_its_reason(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        registry.mark_void("frames-main", "no telemetry")
        record = registry.get("frames-main")
        assert record is not None
        assert record.status == "void"
        assert record.void_reason == "no telemetry"

    def test_completing_or_voiding_an_unknown_arm_raises(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        with pytest.raises(ValueError, match="ghost"):
            registry.mark_void("ghost", "reason")
        with pytest.raises(ValueError, match="ghost"):
            registry.complete("ghost", trace_id=None, cases=0, accuracy=None)

    def test_list_filters_and_orders_by_start(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(
            _record(name="b-later", started_at="2026-01-02T00:00:00+00:00")
        )
        registry.register_launch(_record(name="a-earlier"))
        registry.register_launch(
            _record(name="orb", dataset="orb_text", db_path="/dbs/orb.lancedb")
        )
        registry.mark_void("orb", "no telemetry")

        assert [r.name for r in registry.list()] == ["a-earlier", "orb", "b-later"]
        assert [r.name for r in registry.list(dataset="frames")] == [
            "a-earlier",
            "b-later",
        ]
        assert [r.name for r in registry.list(db_path="/dbs/orb.lancedb")] == ["orb"]
        assert [r.name for r in registry.list(status="void")] == ["orb"]

    def test_export_is_diffable_and_imports_back(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(
            _record(name="b", started_at="2026-01-02T00:00:00+00:00")
        )
        registry.register_launch(_record(name="a"))
        registry.complete("a", trace_id="1" * 32, cases=10, accuracy=0.5)
        out = tmp_path / "arms.jsonl"

        assert registry.export_jsonl(out) == 2
        first = out.read_text()
        registry.export_jsonl(out)
        assert out.read_text() == first
        lines = first.splitlines()
        assert [json.loads(line)["name"] for line in lines] == ["a", "b"]
        assert list(json.loads(lines[0])) == sorted(json.loads(lines[0]))

        other = Registry(tmp_path / "other.sqlite")
        assert other.import_jsonl(out) == 2
        assert other.list() == registry.list()

    def test_import_upserts_by_name(self, tmp_path: Path) -> None:
        registry = Registry(tmp_path / "registry.sqlite")
        registry.register_launch(_record())
        out = tmp_path / "arms.jsonl"
        registry.export_jsonl(out)

        registry.mark_void("frames-main", "wrong target")
        registry.export_jsonl(out)
        other = Registry(tmp_path / "other.sqlite")
        other.register_launch(_record())
        other.import_jsonl(out)

        record = other.get("frames-main")
        assert record is not None
        assert record.status == "void"
        assert record.void_reason == "wrong target"
        assert len(other.list()) == 1


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def arm_file(tmp_path: Path) -> Path:
    repo = tmp_path / "wt"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "code.py").write_text("x = 1\n")
    _git(repo, "add", "code.py")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "init",
    )
    (repo / ".env").write_text("LOGFIRE_TOKEN=pylf_v1_eu_test\n")
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "frames.yaml").write_text(
        "qa:\n  model:\n    provider: openai\n    name: glimmer\n"
        "    base_url: http://vllm:11450\n"
        "reranking:\n  model:\n    provider: vllm\n    name: nemo-rerank\n"
        "    base_url: http://vllm:11434\n"
    )
    path = arms / "frames-main.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "frames-main",
                "dataset": "frames",
                "worktree": str(repo),
                "sha": _git(repo, "rev-parse", "HEAD")[:12],
                "config": "frames.yaml",
                "db": "frames.lancedb",
                "limit": 150,
                "flags": ["--skip-db", "--skip-retrieval"],
                "operator": "evaluations",
            }
        )
    )
    return path


def _fingerprint(db: Path | None) -> dict:
    return {
        "db_path": str(db),
        "db_documents": 2500,
        "db_chunks": 464150,
        "db_embedder_provider": "vllm",
        "db_embedder_model": "nemo-embed",
        "db_embedder_dim": 2048,
        "db_version": "0.86.0",
    }


class TestLaunchRecord:
    def test_fills_the_row_from_arm_config_and_corpus(self, arm_file: Path) -> None:
        arm = load_arm(arm_file)
        config = AppConfig.model_validate(yaml.safe_load(arm.config.read_text()))
        assert arm.db is not None

        record = launch_record(
            arm,
            config,
            _fingerprint(arm.db),
            started_at="2026-01-01T00:00:00+00:00",
        )

        assert record.name == "frames-main"
        assert record.dataset == "frames"
        assert record.kind == "qa"
        assert record.status == "launched"
        assert record.source == "harness"
        assert record.git_sha == arm.sha
        assert record.config_path == str(arm.config)
        assert len(record.config_hash or "") == 64
        assert record.db_path == str(arm.db)
        assert record.db_documents == 2500
        assert record.db_chunks == 464150
        assert record.db_embedder == "vllm/nemo-embed dim 2048"
        assert record.db_version == "0.86.0"
        assert record.capability_model == "glimmer"
        assert record.capability_endpoint == "http://vllm:11450"
        assert record.judge_model == DEFAULT_JUDGE_MODEL.name
        assert record.reranker == "nemo-rerank"
        assert record.embedder == (
            f"{config.embeddings.model.provider}/{config.embeddings.model.name} "
            f"dim {config.embeddings.model.vector_dim}"
        )
        assert record.limit_cases == 150
        assert record.operator == "evaluations"
        assert record.started_at == "2026-01-01T00:00:00+00:00"
        assert record.trace_id is None
        assert record.comparator is None
        assert record.differences is None

    def test_records_the_comparator_and_named_differences(self, arm_file: Path) -> None:
        import json

        base = arm_file.with_name("frames-base.yaml")
        base.write_text(
            arm_file.read_text().replace("name: frames-main", "name: frames-base")
        )
        treated = arm_file.with_name("frames-treated.yaml")
        treated.write_text(
            arm_file.read_text().replace("name: frames-main", "name: frames-treated")
            + "comparator: frames-base.yaml\ndifferences: [sha]\n"
            + "decision_rule: McNemar exact\n"
        )
        arm = load_arm(treated)
        config = AppConfig.model_validate(yaml.safe_load(arm.config.read_text()))
        record = launch_record(arm, config, _fingerprint(arm.db), started_at="t")
        assert record.comparator == "frames-base"
        assert json.loads(record.differences or "null") == ["sha"]
        assert record.decision_rule == "McNemar exact"

    def test_the_endpoint_is_the_one_the_run_opens(self, arm_file: Path) -> None:
        arm = load_arm(arm_file)
        config = AppConfig.model_validate(yaml.safe_load(arm.config.read_text()))
        config.qa.model = ModelConfig(provider="ollama", name="gpt-oss")
        config.providers.ollama.base_url = "http://box:11434"

        record = launch_record(arm, config, _fingerprint(arm.db), started_at="t")

        assert record.capability_model == "gpt-oss"
        assert record.capability_endpoint == "http://box:11434/v1"

    def test_a_run_without_qa_records_no_capability(self, arm_file: Path) -> None:
        arm = load_arm(arm_file).model_copy(
            update={"flags": ["--skip-db", "--skip-qa"]}
        )
        config = AppConfig.model_validate(yaml.safe_load(arm.config.read_text()))
        record = launch_record(arm, config, _fingerprint(arm.db), started_at="t")
        assert record.kind == "retrieval"
        assert record.capability_model is None
        assert record.capability_endpoint is None

    def test_judge_comes_from_the_config_when_set(self, arm_file: Path) -> None:
        arm = load_arm(arm_file)
        config = AppConfig.model_validate(yaml.safe_load(arm.config.read_text()))
        config.evaluations.judge = ModelConfig(provider="openai", name="q38")
        record = launch_record(arm, config, _fingerprint(arm.db), started_at="t")
        assert record.judge_model == "q38"

    def test_kind_follows_the_skipped_phases(self, arm_file: Path) -> None:
        arm = load_arm(arm_file)
        config = AppConfig.model_validate(yaml.safe_load(arm.config.read_text()))
        fingerprint = _fingerprint(arm.db)
        retrieval = arm.model_copy(update={"flags": ["--skip-db", "--skip-qa"]})
        build = arm.model_copy(update={"flags": ["--skip-qa", "--skip-retrieval"]})
        assert (
            launch_record(retrieval, config, fingerprint, started_at="t").kind
            == "retrieval"
        )
        assert launch_record(build, config, fingerprint, started_at="t").kind == "build"
        assert (
            launch_record(build, config, fingerprint, started_at="t").judge_model
            is None
        )

    def test_unjudged_datasets_record_no_judge(self, arm_file: Path) -> None:
        arm = load_arm(arm_file).model_copy(update={"dataset": "t2_finqa"})
        config = AppConfig.model_validate(yaml.safe_load(arm.config.read_text()))
        record = launch_record(arm, config, _fingerprint(arm.db), started_at="t")
        assert record.judge_model is None


class TestArmsCommands:
    def test_list_show_export_import(self, tmp_path: Path) -> None:
        from evaluations.benchmark import app

        registry_path = tmp_path / "registry.sqlite"
        registry = Registry(registry_path)
        registry.register_launch(_record())
        registry.register_launch(_record(name="orb-a", dataset="orb_text"))
        registry.mark_void("orb-a", "no telemetry")
        runner = CliRunner()

        listed = runner.invoke(app, ["arms", "list", "--registry", str(registry_path)])
        assert listed.exit_code == 0, listed.output
        assert "frames-main" in listed.output and "orb-a" in listed.output
        assert "void" in listed.output

        filtered = runner.invoke(
            app,
            ["arms", "list", "--registry", str(registry_path), "--dataset", "frames"],
        )
        assert "orb-a" not in filtered.output

        shown = runner.invoke(
            app, ["arms", "show", "frames-main", "--registry", str(registry_path)]
        )
        assert shown.exit_code == 0, shown.output
        assert "Inferact/Muse-Glimmer-30B-NVFP4-W4A4" in shown.output
        assert "/dbs/frames.lancedb" in shown.output

        missing = runner.invoke(
            app, ["arms", "show", "ghost", "--registry", str(registry_path)]
        )
        assert missing.exit_code == 1

        out = tmp_path / "arms.jsonl"
        exported = runner.invoke(
            app, ["arms", "export", str(out), "--registry", str(registry_path)]
        )
        assert exported.exit_code == 0, exported.output
        other = tmp_path / "other.sqlite"
        imported = runner.invoke(
            app, ["arms", "import", str(out), "--registry", str(other)]
        )
        assert imported.exit_code == 0, imported.output
        assert Registry(other).list() == registry.list()

    def test_void_marks_an_arm(self, tmp_path: Path) -> None:
        from evaluations.benchmark import app

        registry_path = tmp_path / "registry.sqlite"
        Registry(registry_path).register_launch(_record())
        result = CliRunner().invoke(
            app,
            [
                "arms",
                "void",
                "frames-main",
                "--reason",
                "wrong target",
                "--registry",
                str(registry_path),
            ],
        )
        assert result.exit_code == 0, result.output
        record = Registry(registry_path).get("frames-main")
        assert record is not None
        assert record.status == "void" and record.void_reason == "wrong target"


class TestPreflightRegisters:
    def test_a_passing_preflight_registers_the_launch(
        self, arm_file: Path, tmp_path: Path
    ) -> None:
        import asyncio

        from evaluations.benchmark import app
        from haiku.rag.client import HaikuRAG

        arm = load_arm(arm_file)
        config = AppConfig.model_validate(yaml.safe_load(arm.config.read_text()))

        assert arm.db is not None

        async def create() -> None:
            async with HaikuRAG(arm.db, config=config, create=True):
                pass

        asyncio.run(create())
        registry_path = tmp_path / "registry.sqlite"
        result = CliRunner().invoke(
            app,
            [
                "preflight",
                str(arm_file),
                "--register",
                "--registry",
                str(registry_path),
            ],
        )
        assert result.exit_code == 0, result.output
        record = Registry(registry_path).get("frames-main")
        assert record is not None
        assert record.status == "launched"
        assert record.db_documents == 0
        assert record.git_sha is not None and record.git_sha.startswith(arm.sha)
        assert "registered frames-main" in result.output

    def test_a_failing_preflight_registers_nothing(
        self, arm_file: Path, tmp_path: Path
    ) -> None:
        from evaluations.benchmark import app

        registry_path = tmp_path / "registry.sqlite"
        result = CliRunner().invoke(
            app,
            [
                "preflight",
                str(arm_file),
                "--register",
                "--registry",
                str(registry_path),
            ],
        )
        assert result.exit_code == 1
        assert Registry(registry_path).list() == []

from pathlib import Path

import pytest

from evaluations.datasets import DATASETS
from haiku.rag.config import load_yaml_config
from haiku.rag.config.models import AppConfig

CONFIG_DIR = Path(__file__).parent.parent / "configs"

PINNED_JUDGE_SAMPLING = {
    "temperature": 0.6,
    "max_tokens": 16384,
    "extra_body": {
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    },
}


def _config_paths() -> list[Path]:
    return sorted(CONFIG_DIR.glob("*.yaml"))


def _load(path: Path) -> AppConfig:
    return AppConfig.model_validate(load_yaml_config(path))


def test_configs_present() -> None:
    assert _config_paths(), f"no reference configs found in {CONFIG_DIR}"


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_config_validates(path: Path) -> None:
    _load(path)


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_filename_names_a_dataset(path: Path) -> None:
    assert path.stem in DATASETS


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: p.stem)
def test_judge_pinned_where_the_judge_runs(path: Path) -> None:
    """Wherever a judge runs, its sampling must be frozen so accuracy stays
    comparable across runs.

    A dataset without its own qa_evaluator is scored by the LLM judge and must
    carry the block. A dataset with a deterministic evaluator may still need one,
    because RefusalJudge runs on any case carrying an answerability label; where
    it declares no judge it is asserting that no case does.
    """
    judge = _load(path).evaluations.judge
    if DATASETS[path.stem].qa_evaluator is not None and judge is None:
        return

    assert judge is not None
    assert judge.temperature == PINNED_JUDGE_SAMPLING["temperature"]
    assert judge.max_tokens == PINNED_JUDGE_SAMPLING["max_tokens"]
    assert judge.extra_body == PINNED_JUDGE_SAMPLING["extra_body"]

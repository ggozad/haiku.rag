from evaluations.config import DatasetSpec

from .frames import FRAMES_SPEC
from .hotpotqa import HOTPOTQA_SPEC
from .open_rag_bench import (
    ORB_MULTIMODAL_NEMOTRON_SPEC,
    ORB_MULTIMODAL_SPEC,
    ORB_TEXT_SPEC,
)
from .t2_ragbench import T2_FINQA_SPEC, T2_TATDQA_SPEC
from .wix import WIX_SPEC

DATASETS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (
        WIX_SPEC,
        FRAMES_SPEC,
        HOTPOTQA_SPEC,
        ORB_TEXT_SPEC,
        ORB_MULTIMODAL_SPEC,
        ORB_MULTIMODAL_NEMOTRON_SPEC,
        T2_FINQA_SPEC,
        T2_TATDQA_SPEC,
    )
}

__all__ = ["DATASETS"]

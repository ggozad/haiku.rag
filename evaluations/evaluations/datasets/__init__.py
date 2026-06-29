from evaluations.config import DatasetSpec

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
        ORB_TEXT_SPEC,
        ORB_MULTIMODAL_SPEC,
        ORB_MULTIMODAL_NEMOTRON_SPEC,
        T2_FINQA_SPEC,
        T2_TATDQA_SPEC,
    )
}

__all__ = ["DATASETS"]

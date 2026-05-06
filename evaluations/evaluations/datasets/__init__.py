from evaluations.config import DatasetSpec

from .hotpotqa import HOTPOTQA_SPEC
from .open_rag_bench import ORB_MULTIMODAL_SPEC, ORB_TEXT_SPEC
from .repliqa import REPLIQA_SPEC
from .wix import WIX_SPEC

DATASETS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (
        REPLIQA_SPEC,
        WIX_SPEC,
        HOTPOTQA_SPEC,
        ORB_TEXT_SPEC,
        ORB_MULTIMODAL_SPEC,
    )
}

__all__ = ["DATASETS"]

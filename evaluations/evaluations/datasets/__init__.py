from evaluations.config import DatasetSpec

from .open_rag_bench import ORB_MULTIMODAL_SPEC, ORB_TEXT_SPEC
from .wix import WIX_SPEC

DATASETS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (
        WIX_SPEC,
        ORB_TEXT_SPEC,
        ORB_MULTIMODAL_SPEC,
    )
}

__all__ = ["DATASETS"]

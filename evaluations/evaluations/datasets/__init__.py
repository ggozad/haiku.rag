from evaluations.config import DatasetSpec

from .hotpotqa import HOTPOTQA_SPEC
from .repliqa import REPLIQ_SPEC
from .wix import WIX_SPEC

DATASETS: dict[str, DatasetSpec] = {
    spec.key: spec for spec in (REPLIQ_SPEC, WIX_SPEC, HOTPOTQA_SPEC)
}

__all__ = ["DATASETS"]

from evaluations.config import DatasetSpec

from .hotpotqa import HOTPOTQA_SPEC
from .open_rag_bench import OPEN_RAG_BENCH_SPEC
from .repliqa import REPLIQ_SPEC
from .wix import WIX_SPEC

DATASETS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (REPLIQ_SPEC, WIX_SPEC, HOTPOTQA_SPEC, OPEN_RAG_BENCH_SPEC)
}

__all__ = ["DATASETS"]

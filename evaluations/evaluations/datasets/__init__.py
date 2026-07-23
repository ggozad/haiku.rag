from evaluations.config import DatasetSpec

from .frames import FRAMES_SPEC
from .hotpotqa import HOTPOTQA_SPEC
from .mtrag import (
    MTRAG_CLAPNQ_LIVE_SPEC,
    MTRAG_CLAPNQ_LIVE_UNCOMPACTED_SPEC,
    MTRAG_CLAPNQ_REWRITE_SPEC,
    MTRAG_CLAPNQ_SPEC,
)
from .open_rag_bench import (
    ORB_MULTIMODAL_NEMOTRON_SPEC,
    ORB_MULTIMODAL_SPEC,
    ORB_TEXT_SPEC,
)
from .t2_ragbench import T2_FINQA_SPEC, T2_TATDQA_SPEC

DATASETS: dict[str, DatasetSpec] = {
    spec.key: spec
    for spec in (
        FRAMES_SPEC,
        HOTPOTQA_SPEC,
        MTRAG_CLAPNQ_SPEC,
        MTRAG_CLAPNQ_REWRITE_SPEC,
        MTRAG_CLAPNQ_LIVE_SPEC,
        MTRAG_CLAPNQ_LIVE_UNCOMPACTED_SPEC,
        ORB_TEXT_SPEC,
        ORB_MULTIMODAL_SPEC,
        ORB_MULTIMODAL_NEMOTRON_SPEC,
        T2_FINQA_SPEC,
        T2_TATDQA_SPEC,
    )
}

__all__ = ["DATASETS"]

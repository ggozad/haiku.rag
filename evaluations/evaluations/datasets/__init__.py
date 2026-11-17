from importlib.metadata import EntryPoint, entry_points

from evaluations.config import DatasetSpec

from .enterprise_rag import ENTERPRISE_RAG_SPEC
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
        ENTERPRISE_RAG_SPEC,
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

ENTRY_POINT_GROUP = "haiku.rag.evaluations.datasets"


def registered_datasets() -> dict[str, EntryPoint]:
    """The entry points registered under ENTRY_POINT_GROUP by lowercased name,
    unloaded, so a dataset with a broken import fails only when it is run."""
    return {ep.name.lower(): ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


__all__ = ["DATASETS", "ENTRY_POINT_GROUP", "registered_datasets"]

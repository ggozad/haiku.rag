from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np

from haiku.rag.client.scope import DatabaseRef
from haiku.rag.config.models import DuplicateDetectionConfig
from haiku.rag.doctor import DuplicateFamily, VectorScan, duplicate_families


class ComparedDocument(TypedDict):
    id: str
    title: str | None
    uri: str | None


@dataclass
class Comparison:
    """The documents of one database compared by centroid.

    `documents` holds each compared document's `id`, `title` and `uri`, in
    `ids` order, which is the order of `unit`'s rows.
    """

    documents: list[ComparedDocument]
    families: list[DuplicateFamily]
    left_off: int
    ids: list[str]
    unit: np.ndarray
    # The chunks and document_meta table versions read when the scan began.
    versions: dict[str, int | None] = field(default_factory=dict)


def compare(
    scan: VectorScan,
    meta: dict[str, dict[str, Any]],
    cfg: DuplicateDetectionConfig,
) -> Comparison:
    """Compare every document with an embedded chunk by its centroid.

    `meta` maps a document id to its `title` and `uri`. A document without a
    finite, non-zero centroid, one with no embedded chunk included, is left out
    and counted, as is a document in `meta` missing from the scan.
    """
    centroids = np.asarray(scan.centroids, dtype=np.float64).reshape(
        len(scan.document_ids), scan.dimension
    )
    finite_parts = np.isfinite(centroids)
    finite = finite_parts.all(axis=1)
    norms = np.linalg.norm(np.where(finite_parts, centroids, 0.0), axis=1)
    kept = np.nonzero(finite & (norms > 0))[0]
    ids = [scan.document_ids[i] for i in kept]
    # Cosine over unit rows: the raw centroids of extreme vectors overflow.
    unit = (centroids[kept] / norms[kept][:, None]).astype(np.float32)
    return Comparison(
        documents=[
            ComparedDocument(
                id=doc_id,
                title=meta.get(doc_id, {}).get("title"),
                uri=meta.get(doc_id, {}).get("uri"),
            )
            for doc_id in ids
        ],
        families=duplicate_families(ids, unit, scan.counts[kept], cfg),
        left_off=len(set(scan.document_ids) | set(meta)) - len(ids),
        ids=ids,
        unit=unit,
    )


def similarity(unit: np.ndarray, ids: list[str], doc_id: str) -> list[float]:
    """Centroid cosine of every document to `doc_id`, in `ids` order."""
    return np.round(unit @ unit[ids.index(doc_id)], 4).tolist()


@dataclass(frozen=True)
class Centroids:
    """One comparison's unit centroids, in `ids` order, and its generation."""

    generation: str
    ids: list[str]
    unit: np.ndarray


class Retained:
    """Unit centroids kept for similarity lookups, within one memory budget
    across databases. A newer comparison evicts the oldest others, and one that
    does not fit alone is not kept."""

    def __init__(self, budget: int = 256 * 1024**2) -> None:
        self.budget = budget
        self._kept: dict[DatabaseRef, Centroids] = {}

    def keep(
        self, database: DatabaseRef, generation: str, ids: list[str], unit: np.ndarray
    ) -> bool:
        self._kept.pop(database, None)
        if unit.nbytes > self.budget:
            return False
        while self._kept and self._used() + unit.nbytes > self.budget:
            del self._kept[next(iter(self._kept))]
        self._kept[database] = Centroids(generation, ids, unit)
        return True

    def get(self, database: DatabaseRef) -> Centroids | None:
        return self._kept.get(database)

    def _used(self) -> int:
        return sum(kept.unit.nbytes for kept in self._kept.values())

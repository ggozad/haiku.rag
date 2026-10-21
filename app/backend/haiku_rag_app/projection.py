import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.manifold import TSNE

from haiku.rag.config.models import DuplicateDetectionConfig
from haiku.rag.doctor import DuplicateFamily, VectorScan, duplicate_families


@dataclass
class Projection:
    """The documents of one database placed in two dimensions.

    `points` and `families` are what the page draws. `unit` keeps the unit
    centroids, in `ids` order, for similarity lookups.
    """

    points: list[dict[str, Any]]
    families: list[DuplicateFamily]
    left_off: int
    ids: list[str]
    unit: np.ndarray
    # The chunks and document_meta table versions read when the scan began.
    versions: dict[str, int | None] = field(default_factory=dict)


def place(unit: np.ndarray) -> np.ndarray:
    """t-SNE coordinates of unit rows by cosine. Identical rows share a point.

    Fewer than three distinct rows are spaced along the x axis.
    """
    distinct, owner = np.unique(unit, axis=0, return_inverse=True)
    if len(distinct) < 3:
        points = np.column_stack([np.arange(len(distinct)), np.zeros(len(distinct))])
    else:
        points = TSNE(
            n_components=2,
            metric="cosine",
            # Neighbourhoods span 3 x perplexity points, which must leave some out.
            perplexity=min(30, (len(distinct) - 1) / 3),
            init="pca",
            random_state=0,
        ).fit_transform(distinct)
    return points[owner.reshape(-1)]


def project(
    scan: VectorScan,
    meta: dict[str, dict[str, Any]],
    cfg: DuplicateDetectionConfig,
) -> Projection:
    """Place every document with an embedded chunk by its centroid.

    `meta` maps a document id to its `title`, `uri` and `metadata`. A document
    without a finite, non-zero centroid, one with no embedded chunk included, is
    left off the map and counted, as is a document in `meta` the scan never saw.
    """
    centroids = np.asarray(scan.centroids, dtype=np.float64).reshape(
        len(scan.document_ids), scan.dimension
    )
    finite_parts = np.isfinite(centroids)
    finite = finite_parts.all(axis=1)
    norms = np.linalg.norm(np.where(finite_parts, centroids, 0.0), axis=1)
    kept = np.nonzero(finite & (norms > 0))[0]
    ids = [scan.document_ids[i] for i in kept]
    unit = centroids[kept] / norms[kept][:, None]
    coordinates = place(unit)

    unit32 = unit.astype(np.float32)
    # Cosine over unit rows: the raw centroids of extreme vectors overflow.
    families = duplicate_families(ids, unit32, scan.counts[kept], cfg)
    family_of = {
        member: index
        for index, family in enumerate(families)
        for member in family.members
    }
    points = []
    for row, doc_id in enumerate(ids):
        info = meta.get(doc_id, {})
        metadata = info.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata or "{}")
        points.append(
            {
                "id": doc_id,
                "x": float(coordinates[row, 0]),
                "y": float(coordinates[row, 1]),
                "title": info.get("title"),
                "uri": info.get("uri"),
                "source_id": metadata.get("source_id"),
                "content_type": metadata.get("content_type"),
                "chunks": int(scan.counts[kept[row]]),
                "family": family_of.get(doc_id),
            }
        )
    return Projection(
        points=points,
        families=families,
        left_off=len(set(scan.document_ids) | set(meta)) - len(ids),
        ids=ids,
        unit=unit32,
    )


def similarity(unit: np.ndarray, ids: list[str], doc_id: str) -> list[float]:
    """Centroid cosine of every document to `doc_id`, in `ids` order."""
    return np.round(unit @ unit[ids.index(doc_id)], 4).tolist()


@dataclass(frozen=True)
class Centroids:
    """One map's unit centroids, in `ids` order, and the map they belong to."""

    generation: str
    ids: list[str]
    unit: np.ndarray


class Retained:
    """Unit centroids kept for similarity lookups, within one memory budget
    across databases. A newer projection evicts the oldest others; one that
    does not fit alone is not kept."""

    def __init__(self, budget: int = 256 * 1024**2) -> None:
        self.budget = budget
        self._kept: dict[str, Centroids] = {}

    def keep(
        self, source: str, generation: str, ids: list[str], unit: np.ndarray
    ) -> bool:
        self._kept.pop(source, None)
        if unit.nbytes > self.budget:
            return False
        while self._kept and self._used() + unit.nbytes > self.budget:
            del self._kept[next(iter(self._kept))]
        self._kept[source] = Centroids(generation, ids, unit)
        return True

    def get(self, source: str) -> Centroids | None:
        return self._kept.get(source)

    def _used(self) -> int:
        return sum(kept.unit.nbytes for kept in self._kept.values())

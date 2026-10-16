import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from haiku.rag.config.models import DuplicateDetectionConfig
from haiku.rag.doctor import DuplicateFamily, VectorScan, duplicate_families


@dataclass
class Projection:
    """The documents of one database placed in two dimensions.

    `points` and `families` are what the page draws. `unit` keeps the unit
    centroids, in `ids` order, for nearest-neighbour lookups.
    """

    points: list[dict[str, Any]]
    families: list[DuplicateFamily]
    left_off: int
    explained_variance: float
    ids: list[str]
    unit: np.ndarray
    # The chunks and document_meta table versions read when the scan began.
    versions: dict[str, int | None] = field(default_factory=dict)


def principal_plane(unit: np.ndarray) -> tuple[np.ndarray, float]:
    """Coordinates on the two leading principal components, and the share of
    the variance they explain.

    Exact: the covariance is dimension × dimension, so `eigh` is cheap whatever
    the number of documents. A single point, or points without variance, all
    land on the origin with no variance explained.
    """
    n = unit.shape[0]
    if n < 2:
        return np.zeros((n, 2)), 0.0
    centred = unit - unit.mean(axis=0)
    covariance = centred.T @ centred / n
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    total = float(eigenvalues.sum())
    if total <= 0:
        return np.zeros((n, 2)), 0.0
    leading = eigenvectors[:, -2:][:, ::-1]
    # Rounding can put a rank-two share a hair above one.
    explained = min(1.0, float(eigenvalues[-2:].sum()) / total)
    coordinates = centred @ leading
    # A one-dimensional embedding has a single component: the second is zero.
    return np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1]))), explained


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
    coordinates, explained = principal_plane(unit)

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
        explained_variance=explained,
        ids=ids,
        unit=unit32,
    )


def nearest(unit: np.ndarray, ids: list[str], doc_id: str, k: int) -> list[dict]:
    """The `k` documents closest to `doc_id` by centroid cosine."""
    index = ids.index(doc_id)
    similarity = unit @ unit[index]
    order = [i for i in np.argsort(-similarity) if i != index][:k]
    return [{"id": ids[i], "similarity": float(similarity[i])} for i in order]

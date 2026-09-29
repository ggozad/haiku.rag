import numpy as np
from pydantic import BaseModel

from haiku.rag.config.models import DuplicateDetectionConfig


def document_centroids(
    document_id_column, vectors: "np.ndarray", embedded: "np.ndarray", dim: int
) -> tuple[list[str], "np.ndarray", "np.ndarray"]:
    """Reduce each document's chunk vectors to one summed centroid.

    Dictionary-encode the document ids into integer codes, then sum each
    document's embedded rows in a single pass per document — no second full copy
    of the vector matrix. Returns (document ids, summed centroids, chunk counts);
    the caller normalizes.
    """
    encoded = document_id_column.combine_chunks().dictionary_encode()
    ids = encoded.dictionary.to_pylist()
    codes = encoded.indices.to_numpy(zero_copy_only=False)
    centroids = np.zeros((len(ids), dim), dtype=np.float32)
    counts = np.zeros(len(ids), dtype=np.int64)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes, np.arange(len(ids) + 1), sorter=order)
    for d in range(len(ids)):
        rows = order[bounds[d] : bounds[d + 1]]
        rows = rows[embedded[rows]]
        counts[d] = rows.size
        if rows.size:
            centroids[d] = vectors[rows].sum(axis=0)
    return ids, centroids, counts


class DuplicateFamily(BaseModel):
    members: list[str]
    keep: str
    similarity: dict[str, float]
    sizes: dict[str, int]


def duplicate_families(
    doc_ids: list[str],
    centroids: np.ndarray,
    counts: np.ndarray,
    cfg: DuplicateDetectionConfig,
) -> list[DuplicateFamily]:
    """Cluster documents whose embedding centroids are nearly identical.

    ``centroids`` holds one summed (unnormalized) centroid per document and
    ``counts`` its embedded-chunk count. Documents below the small-document
    floor are dropped; the rest are normalized and clustered by union-find over
    pairwise cosine above ``similarity_threshold``. One family per component,
    each carrying every member's highest cosine to another member.
    """
    centroids = np.asarray(centroids, dtype=np.float32)
    counts = np.asarray(counts)
    norms = np.linalg.norm(centroids, axis=1)
    eligible = np.nonzero((counts >= cfg.min_chunks) & (norms > 0))[0]
    if eligible.size < 2:
        return []
    unit = centroids[eligible] / norms[eligible][:, None]
    ids = [doc_ids[i] for i in eligible]
    sizes = {doc_ids[i]: int(counts[i]) for i in eligible}
    n = len(ids)

    # Pairwise cosine, block-wise to avoid a full D×D matrix at once. Each row
    # only compares against higher-indexed documents (upper triangle). Cluster
    # with union-find and keep only each document's best similarity to a twin —
    # a self-similar corpus forms one clique, so storing every pair would be
    # O(D²) objects.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    best = np.zeros(n, dtype=np.float32)
    linked = False
    block = 512
    for start in range(0, n, block):
        sims = unit[start : start + block] @ unit.T
        for row in range(sims.shape[0]):
            gi = start + row
            cols = (
                gi + 1 + np.nonzero(sims[row, gi + 1 :] >= cfg.similarity_threshold)[0]
            )
            if cols.size == 0:
                continue
            linked = True
            row_best = sims[row, cols]
            best[gi] = max(best[gi], float(row_best.max()))
            best[cols] = np.maximum(best[cols], row_best)
            ri = find(gi)
            for gj in cols.tolist():
                parent[find(gj)] = ri
    if not linked:
        return []

    components: dict[int, list[int]] = {}
    for idx in range(n):
        components.setdefault(find(idx), []).append(idx)

    families: list[DuplicateFamily] = []
    for indices in components.values():
        if len(indices) < 2:
            continue
        members = sorted(ids[i] for i in indices)
        # Largest document (most chunks) is the one to keep; smallest id on a tie.
        keep = min(members, key=lambda d: (-sizes[d], d))
        families.append(
            DuplicateFamily(
                members=members,
                keep=keep,
                similarity={ids[i]: round(float(best[i]), 3) for i in indices},
                sizes={d: sizes[d] for d in members},
            )
        )
    return sorted(families, key=lambda f: f.members)

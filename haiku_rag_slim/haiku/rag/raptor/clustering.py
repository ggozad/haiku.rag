import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

RANDOM_SEED = 42

_UMAP = None
_GaussianMixture = None
_import_error: ImportError | None = None

try:
    from sklearn.mixture import GaussianMixture as _GaussianMixture

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImportWarning)
        from umap import UMAP as _UMAP
except ImportError as e:
    _import_error = e


def _check_dependencies() -> None:
    """Check if RAPTOR dependencies are installed."""
    if _import_error is not None:
        raise ImportError(
            "RAPTOR requires additional dependencies. "
            "Install them with: pip install 'haiku.rag-slim[raptor]'"
        ) from _import_error


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int | None = None,
    min_dist: float = 0.0,
    metric: str = "cosine",
) -> np.ndarray:
    """Reduce embedding dimensionality using UMAP.

    Args:
        embeddings: High-dimensional embeddings (n_samples, n_features)
        n_components: Target dimensionality
        n_neighbors: UMAP neighborhood size. If None, uses sqrt(n_samples)
        min_dist: UMAP minimum distance parameter
        metric: Distance metric for UMAP

    Returns:
        Reduced embeddings (n_samples, n_components)
    """
    _check_dependencies()
    assert _UMAP is not None

    if n_neighbors is None:
        n_neighbors = max(2, int((len(embeddings) - 1) ** 0.5))

    # Ensure n_neighbors doesn't exceed dataset size
    n_neighbors = min(n_neighbors, len(embeddings) - 1)

    reducer = _UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=RANDOM_SEED,
    )
    return reducer.fit_transform(embeddings)  # type: ignore[return-value]


def get_optimal_cluster_count(
    embeddings: np.ndarray,
    max_clusters: int = 50,
) -> int:
    """Find optimal number of clusters using BIC.

    Args:
        embeddings: Input embeddings
        max_clusters: Maximum clusters to consider

    Returns:
        Optimal number of clusters
    """
    _check_dependencies()
    assert _GaussianMixture is not None

    max_clusters = min(max_clusters, len(embeddings))
    if max_clusters <= 1:
        return 1

    bics = []
    for n in range(1, max_clusters):
        gm = _GaussianMixture(n_components=n, random_state=RANDOM_SEED)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))

    return int(np.argmin(bics) + 1)


def gmm_cluster(
    embeddings: np.ndarray,
    threshold: float = 0.1,
) -> tuple[list[np.ndarray], int]:
    """Perform soft clustering using GMM.

    Args:
        embeddings: Input embeddings
        threshold: Probability threshold for cluster membership

    Returns:
        Tuple of (cluster_labels_per_item, n_clusters)
        Each item in cluster_labels is an array of cluster IDs it belongs to
    """
    _check_dependencies()
    assert _GaussianMixture is not None

    n_clusters = get_optimal_cluster_count(embeddings)
    gm = _GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def cluster_embeddings(
    embeddings: np.ndarray,
    reduction_dim: int = 10,
    threshold: float = 0.1,
    n_neighbors: int = 10,
    min_dist: float = 0.0,
) -> list[np.ndarray]:
    """Cluster embeddings using UMAP dimensionality reduction + GMM.

    This implements the global + local clustering approach from RAPTOR:
    1. Reduce dimensions globally
    2. Perform global GMM clustering
    3. For each global cluster, perform local clustering

    Args:
        embeddings: High-dimensional embeddings
        reduction_dim: Target dimension for UMAP
        threshold: GMM probability threshold for soft clustering
        n_neighbors: UMAP neighborhood size
        min_dist: UMAP minimum distance

    Returns:
        List of cluster assignments for each embedding (soft clustering)
    """
    n_samples = len(embeddings)

    # Handle edge cases
    if n_samples <= 2:
        return [np.array([0]) for _ in range(n_samples)]

    # Adjust reduction_dim if needed
    effective_dim = min(reduction_dim, n_samples - 2)

    # Global dimensionality reduction
    reduced = reduce_embeddings(
        embeddings,
        n_components=effective_dim,
        n_neighbors=min(n_neighbors, n_samples - 1),
        min_dist=min_dist,
    )

    # Global clustering
    global_labels, n_global = gmm_cluster(reduced, threshold)

    # Initialize result - each item starts with empty assignments
    all_clusters: list[np.ndarray] = [np.array([], dtype=int) for _ in range(n_samples)]
    total_clusters = 0

    # Local clustering within each global cluster
    for global_cluster_id in range(n_global):
        # Get indices of items in this global cluster
        in_cluster = [
            i for i, labels in enumerate(global_labels) if global_cluster_id in labels
        ]

        if len(in_cluster) == 0:
            continue

        cluster_embeddings_subset = embeddings[in_cluster]

        # If cluster is small, don't subdivide
        if len(in_cluster) <= effective_dim + 1:
            for idx in in_cluster:
                all_clusters[idx] = np.append(all_clusters[idx], total_clusters)
            total_clusters += 1
            continue

        # Local dimensionality reduction and clustering
        local_reduced = reduce_embeddings(
            cluster_embeddings_subset,
            n_components=effective_dim,
            n_neighbors=min(n_neighbors, len(in_cluster) - 1),
            min_dist=min_dist,
        )
        local_labels, n_local = gmm_cluster(local_reduced, threshold)

        # Map local cluster assignments back to global indices
        for local_cluster_id in range(n_local):
            for local_idx, global_idx in enumerate(in_cluster):
                if local_cluster_id in local_labels[local_idx]:
                    all_clusters[global_idx] = np.append(
                        all_clusters[global_idx], total_clusters + local_cluster_id
                    )

        total_clusters += n_local

    # Ensure every item has at least one cluster
    for i, clusters in enumerate(all_clusters):
        if len(clusters) == 0:
            all_clusters[i] = np.array([0])

    return all_clusters


def group_into_clusters[T](
    items: list[T],
    cluster_assignments: list[np.ndarray],
) -> list[list[T]]:
    """Group items by their cluster assignments.

    Args:
        items: List of items to group
        cluster_assignments: Cluster labels for each item (from cluster_embeddings)

    Returns:
        List of clusters, where each cluster is a list of items
    """
    if not items:
        return []

    # Find all unique cluster IDs
    all_cluster_ids = set()
    for assignments in cluster_assignments:
        all_cluster_ids.update(assignments.tolist())

    # Group items by cluster
    clusters: list[list[T]] = []
    for cluster_id in sorted(all_cluster_ids):
        cluster_items = [
            item
            for item, assignments in zip(items, cluster_assignments)
            if cluster_id in assignments
        ]
        if cluster_items:
            clusters.append(cluster_items)

    return clusters

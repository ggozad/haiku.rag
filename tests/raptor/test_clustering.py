import numpy as np


class TestReduceEmbeddings:
    def test_reduces_dimensions(self):
        from haiku.rag.raptor.clustering import reduce_embeddings

        embeddings = np.random.rand(20, 100)
        reduced = reduce_embeddings(embeddings, n_components=10)
        assert reduced.shape == (20, 10)

    def test_handles_small_dataset(self):
        from haiku.rag.raptor.clustering import reduce_embeddings

        embeddings = np.random.rand(5, 50)
        reduced = reduce_embeddings(embeddings, n_components=3)
        assert reduced.shape == (5, 3)


class TestGetOptimalClusterCount:
    def test_returns_positive_integer(self):
        from haiku.rag.raptor.clustering import get_optimal_cluster_count

        embeddings = np.random.rand(30, 10)
        count = get_optimal_cluster_count(embeddings, max_clusters=10)
        assert isinstance(count, int)
        assert count >= 1

    def test_respects_max_clusters(self):
        from haiku.rag.raptor.clustering import get_optimal_cluster_count

        embeddings = np.random.rand(50, 10)
        count = get_optimal_cluster_count(embeddings, max_clusters=5)
        assert count <= 5


class TestGMMCluster:
    def test_returns_labels_for_each_embedding(self):
        from haiku.rag.raptor.clustering import gmm_cluster

        embeddings = np.random.rand(20, 10)
        labels, n_clusters = gmm_cluster(embeddings, threshold=0.1)
        assert len(labels) == 20
        assert n_clusters >= 1

    def test_soft_clustering_allows_multiple_labels(self):
        from haiku.rag.raptor.clustering import gmm_cluster

        # Create embeddings that should overlap between clusters
        np.random.seed(42)
        embeddings = np.random.rand(30, 10)
        labels, _ = gmm_cluster(embeddings, threshold=0.1)
        # Verify structure is correct - each label is an array
        assert all(isinstance(label, np.ndarray) for label in labels)


class TestClusterEmbeddings:
    def test_returns_cluster_assignments(self):
        from haiku.rag.raptor.clustering import cluster_embeddings

        np.random.seed(42)
        embeddings = np.random.rand(30, 100)
        clusters = cluster_embeddings(
            embeddings, reduction_dim=10, threshold=0.1, n_neighbors=10
        )
        assert len(clusters) == 30
        # Each item should have at least one cluster assignment
        assert all(len(c) >= 1 for c in clusters)

    def test_handles_minimum_viable_input(self):
        from haiku.rag.raptor.clustering import cluster_embeddings

        # Minimum for UMAP is around 4-5 points
        embeddings = np.random.rand(5, 50)
        clusters = cluster_embeddings(
            embeddings, reduction_dim=2, threshold=0.1, n_neighbors=3
        )
        assert len(clusters) == 5


class TestGroupIntoClusters:
    def test_groups_by_cluster_labels(self):
        from haiku.rag.raptor.clustering import group_into_clusters

        # Simulate cluster assignments: item 0 in cluster 0, item 1 in clusters 0 and 1, etc.
        cluster_assignments = [
            np.array([0]),
            np.array([0, 1]),
            np.array([1]),
            np.array([2]),
        ]
        items = ["a", "b", "c", "d"]

        groups = group_into_clusters(items, cluster_assignments)

        # Should have 3 clusters (0, 1, 2)
        assert len(groups) == 3
        # Cluster 0 should have items a, b
        assert set(groups[0]) == {"a", "b"}
        # Cluster 1 should have items b, c
        assert set(groups[1]) == {"b", "c"}
        # Cluster 2 should have item d
        assert set(groups[2]) == {"d"}

    def test_handles_empty_input(self):
        from haiku.rag.raptor.clustering import group_into_clusters

        groups = group_into_clusters([], [])
        assert groups == []

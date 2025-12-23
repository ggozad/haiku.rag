from haiku.rag.config.models import AppConfig, RaptorConfig


class TestRaptorConfig:
    def test_default_values(self):
        config = RaptorConfig()
        assert config.enabled is False
        assert config.max_depth == 5
        assert config.min_cluster_size == 3
        assert config.max_cluster_size == 15
        assert config.umap_n_neighbors == 10
        assert config.umap_min_dist == 0.0
        assert config.model is None
        assert config.max_search_results == 3

    def test_custom_values(self):
        config = RaptorConfig(
            enabled=True,
            max_depth=3,
            min_cluster_size=5,
            max_cluster_size=20,
            umap_n_neighbors=15,
            umap_min_dist=0.1,
        )
        assert config.enabled is True
        assert config.max_depth == 3
        assert config.min_cluster_size == 5
        assert config.max_cluster_size == 20
        assert config.umap_n_neighbors == 15
        assert config.umap_min_dist == 0.1

    def test_app_config_has_raptor(self):
        app_config = AppConfig()
        assert hasattr(app_config, "raptor")
        assert isinstance(app_config.raptor, RaptorConfig)
        assert app_config.raptor.enabled is False

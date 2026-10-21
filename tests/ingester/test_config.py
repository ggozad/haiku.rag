from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from haiku.rag.config import (
    APIConfig,
    AppConfig,
    FSSourceConfig,
    HTTPSourceConfig,
    IngestedDatabase,
    IngesterConfig,
    LanceDBConfig,
    QueueConfig,
    RetryPolicyConfig,
    S3SourceConfig,
    WorkerConfig,
)


def test_default_ingester_config_has_sane_values():
    cfg = IngesterConfig()
    assert cfg.databases == {}
    assert cfg.workers.worker_count == 4
    assert cfg.workers.retry.max_attempts == 5
    assert cfg.workers.lease_ttl_s == 120
    assert cfg.workers.heartbeat_interval_s == 30
    assert cfg.api.enabled is True
    assert cfg.api.port == 8765
    assert cfg.api.root_path == ""


def test_queue_namespace_defaults_to_empty():
    assert QueueConfig().namespace == ""


def test_worker_config_rejects_heartbeat_too_close_to_lease():
    with pytest.raises(ValidationError, match="heartbeat_interval_s"):
        WorkerConfig(lease_ttl_s=60, heartbeat_interval_s=30)


def test_worker_config_accepts_heartbeat_within_a_third_of_lease():
    cfg = WorkerConfig(lease_ttl_s=60, heartbeat_interval_s=20)
    assert cfg.heartbeat_interval_s == 20


def test_worker_config_rejects_non_positive_timings():
    with pytest.raises(ValidationError):
        WorkerConfig(lease_ttl_s=0)
    with pytest.raises(ValidationError):
        WorkerConfig(heartbeat_interval_s=0)


def test_worker_config_rejects_unknown_field():
    """The former claim_timeout_s (and any typo) must fail loudly rather than
    being silently ignored."""
    with pytest.raises(ValidationError, match="claim_timeout_s"):
        WorkerConfig(claim_timeout_s=1800)  # ty: ignore[unknown-argument]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", ""),
        ("/", ""),
        ("ingester", "/ingester"),
        ("/ingester", "/ingester"),
        ("/ingester/", "/ingester"),
        ("  /ingester/  ", "/ingester"),
    ],
)
def test_api_root_path_normalized(raw, expected):
    cfg = APIConfig(root_path=raw)

    assert cfg.root_path == expected


def test_api_root_path_normalized_on_assignment():
    """validate_assignment ensures CLI overrides (--root-path) get the same
    normalization as values parsed from the config file."""
    cfg = APIConfig()

    cfg.root_path = "ingester/"

    assert cfg.root_path == "/ingester"


def test_discriminator_picks_fs_source():
    cfg = IngestedDatabase.model_validate(
        {"sources": [{"type": "fs", "root": "/tmp/docs"}]}
    )
    assert isinstance(cfg.sources[0], FSSourceConfig)
    assert cfg.sources[0].root == Path("/tmp/docs")
    assert cfg.sources[0].delete_orphans is True
    assert cfg.sources[0].metadata_provider is None


def test_source_carries_metadata_provider_name():
    cfg = IngestedDatabase.model_validate(
        {
            "sources": [
                {"type": "fs", "root": "/tmp/docs", "metadata_provider": "example"}
            ]
        }
    )
    assert cfg.sources[0].metadata_provider == "example"


def test_discriminator_picks_http_source():
    cfg = IngestedDatabase.model_validate(
        {"sources": [{"type": "http", "id": "arxiv", "urls": ["https://x"]}]}
    )
    assert isinstance(cfg.sources[0], HTTPSourceConfig)
    assert cfg.sources[0].id == "arxiv"


def test_discriminator_picks_s3_source():
    cfg = IngestedDatabase.model_validate(
        {"sources": [{"type": "s3", "uri": "s3://bucket/prefix/"}]}
    )
    assert isinstance(cfg.sources[0], S3SourceConfig)
    assert cfg.sources[0].uri == "s3://bucket/prefix/"


def test_discriminator_picks_webdav_source():
    cfg = IngestedDatabase.model_validate(
        {
            "sources": [
                {
                    "type": "webdav",
                    "id": "nc",
                    "base_url": "https://nc.example.com/dav/",
                    "username": "alice",
                    "password": "hunter2",
                }
            ]
        }
    )
    from haiku.rag.config import WebDAVSourceConfig

    assert isinstance(cfg.sources[0], WebDAVSourceConfig)
    assert cfg.sources[0].base_url == "https://nc.example.com/dav/"
    assert cfg.sources[0].username == "alice"


def test_discriminator_picks_plugin_source():
    cfg = IngestedDatabase.model_validate(
        {
            "sources": [
                {
                    "type": "plugin",
                    "id": "git-docs",
                    "plugin": "git",
                    "options": {"owner": "acme", "repo": "api"},
                }
            ]
        }
    )
    from haiku.rag.config import PluginSourceConfig

    assert isinstance(cfg.sources[0], PluginSourceConfig)
    assert cfg.sources[0].plugin == "git"
    assert cfg.sources[0].options == {"owner": "acme", "repo": "api"}


def test_discriminator_rejects_unknown_type():
    with pytest.raises(ValidationError):
        IngestedDatabase.model_validate({"sources": [{"type": "ftp", "uri": "x"}]})


def test_per_source_retry_overrides_default():
    cfg = IngestedDatabase.model_validate(
        {
            "sources": [
                {
                    "type": "fs",
                    "root": "/tmp",
                    "retry": {"max_attempts": 10},
                }
            ]
        }
    )
    src = cfg.sources[0]
    assert src.retry is not None
    assert src.retry.max_attempts == 10
    # other retry fields fall back to the RetryPolicyConfig defaults
    assert src.retry.base_delay_s == RetryPolicyConfig().base_delay_s


def test_yaml_round_trip():
    yaml_text = """
ingester:
  databases:
    haiku.rag:
      sources:
        - type: fs
          root: /data/docs
          ignore_patterns: ["**/.git/**"]
          delete_orphans: true
        - type: s3
          uri: s3://my-bucket/incoming/
          poll_interval_s: 300
          storage_options:
            endpoint: http://seaweed:8333
        - type: http
          id: arxiv
          urls: [https://arxiv.org/pdf/2301.12345.pdf]
          headers:
            Authorization: Bearer abc
          poll_interval_s: 86400
  workers:
    worker_count: 8
  api:
    enabled: false
"""
    data = yaml.safe_load(yaml_text)
    app = AppConfig.model_validate(data)
    fs, s3, http = app.ingester.databases["haiku.rag"].sources
    assert isinstance(fs, FSSourceConfig)
    assert isinstance(s3, S3SourceConfig)
    assert isinstance(http, HTTPSourceConfig)
    assert fs.ignore_patterns == ["**/.git/**"]
    assert s3.storage_options["endpoint"] == "http://seaweed:8333"
    assert http.headers["Authorization"] == "Bearer abc"
    assert app.ingester.workers.worker_count == 8
    assert app.ingester.api.enabled is False


_FS = {"type": "fs", "root": "/tmp/docs"}


def test_ingester_sources_is_refused_with_the_replacement_named():
    with pytest.raises(ValidationError, match="ingester.databases"):
        AppConfig.model_validate({"ingester": {"sources": [_FS]}})


def test_an_ingested_database_is_one_lancedb_names():
    config = AppConfig.model_validate(
        {
            "lancedb": {"databases": {"papers": "/srv/papers.lancedb"}},
            "ingester": {"databases": {"papers": {"sources": [_FS]}}},
        }
    )

    assert list(config.ingester.databases) == ["papers"]


def test_an_ingested_database_unknown_to_lancedb_is_refused():
    with pytest.raises(ValidationError, match=r"ingester\.databases\[wiki\]"):
        AppConfig.model_validate(
            {
                "lancedb": {"databases": {"papers": "/srv/papers.lancedb"}},
                "ingester": {"databases": {"wiki": {"sources": [_FS]}}},
            }
        )


def test_an_ingested_database_at_a_tag_is_refused():
    with pytest.raises(ValidationError, match=r"ingester\.databases\[papers\].*tag"):
        AppConfig.model_validate(
            {
                "lancedb": {
                    "databases": {
                        "papers": {"location": "/srv/papers.lancedb", "tag": "v1"}
                    }
                },
                "ingester": {"databases": {"papers": {"sources": [_FS]}}},
            }
        )


def test_a_tagged_entry_the_ingester_does_not_name_is_left_to_readers():
    config = AppConfig(
        lancedb=LanceDBConfig(
            databases={
                "papers": "/srv/papers.lancedb",
                "papers-release": {"location": "/srv/papers.lancedb", "tag": "v1"},
            }
        ),
        ingester=IngesterConfig(
            databases={"papers": IngestedDatabase.model_validate({"sources": [_FS]})}
        ),
    )

    assert list(config.ingester.databases) == ["papers"]


def test_without_lancedb_databases_the_default_database_is_ingested():
    config = AppConfig.model_validate(
        {"ingester": {"databases": {"haiku.rag": {"sources": [_FS]}}}}
    )

    assert list(config.ingester.databases) == ["haiku.rag"]


def test_without_lancedb_databases_another_name_is_refused():
    with pytest.raises(ValidationError, match=r"ingester\.databases\[papers\]"):
        AppConfig.model_validate(
            {"ingester": {"databases": {"papers": {"sources": [_FS]}}}}
        )


def test_an_ingested_database_without_sources_is_refused():
    with pytest.raises(ValidationError) as raised:
        AppConfig.model_validate({"ingester": {"databases": {"haiku.rag": {}}}})

    [error] = raised.value.errors()
    assert error["loc"] == ("ingester", "databases", "haiku.rag", "sources")

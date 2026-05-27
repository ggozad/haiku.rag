from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from haiku.rag.config import (
    AppConfig,
    FSSourceConfig,
    HTTPSourceConfig,
    IngesterConfig,
    RetryPolicyConfig,
    S3SourceConfig,
)


def test_default_ingester_config_has_sane_values():
    cfg = IngesterConfig()
    assert cfg.sources == []
    assert cfg.workers.worker_count == 4
    assert cfg.workers.retry.max_attempts == 5
    assert cfg.api.enabled is True
    assert cfg.api.port == 8765


def test_discriminator_picks_fs_source():
    cfg = IngesterConfig.model_validate(
        {"sources": [{"type": "fs", "root": "/tmp/docs"}]}
    )
    assert isinstance(cfg.sources[0], FSSourceConfig)
    assert cfg.sources[0].root == Path("/tmp/docs")
    assert cfg.sources[0].delete_orphans is True


def test_discriminator_picks_http_source():
    cfg = IngesterConfig.model_validate(
        {"sources": [{"type": "http", "id": "arxiv", "urls": ["https://x"]}]}
    )
    assert isinstance(cfg.sources[0], HTTPSourceConfig)
    assert cfg.sources[0].id == "arxiv"


def test_discriminator_picks_s3_source():
    cfg = IngesterConfig.model_validate(
        {"sources": [{"type": "s3", "uri": "s3://bucket/prefix/"}]}
    )
    assert isinstance(cfg.sources[0], S3SourceConfig)
    assert cfg.sources[0].uri == "s3://bucket/prefix/"


def test_discriminator_picks_webdav_source():
    cfg = IngesterConfig.model_validate(
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


def test_discriminator_rejects_unknown_type():
    with pytest.raises(ValidationError):
        IngesterConfig.model_validate({"sources": [{"type": "ftp", "uri": "x"}]})


def test_per_source_retry_overrides_default():
    cfg = IngesterConfig.model_validate(
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
    assert len(app.ingester.sources) == 3
    fs, s3, http = app.ingester.sources
    assert isinstance(fs, FSSourceConfig)
    assert isinstance(s3, S3SourceConfig)
    assert isinstance(http, HTTPSourceConfig)
    assert fs.ignore_patterns == ["**/.git/**"]
    assert s3.storage_options["endpoint"] == "http://seaweed:8333"
    assert http.headers["Authorization"] == "Bearer abc"
    assert app.ingester.workers.worker_count == 8
    assert app.ingester.api.enabled is False

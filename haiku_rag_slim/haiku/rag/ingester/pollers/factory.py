from haiku.rag.config import (
    FSSourceConfig,
    HTTPSourceConfig,
    S3SourceConfig,
    SourceConfig,
    WebDAVSourceConfig,
)
from haiku.rag.ingester.sources import (
    FSSource,
    HTTPSource,
    S3Source,
    Source,
    WebDAVSource,
)


def build_source(
    cfg: SourceConfig,
    *,
    supported_extensions: list[str] | None = None,
) -> Source:
    """Instantiate the right adapter for a SourceConfig.

    Source IDs auto-derive from the target when the config didn't supply one,
    matching the conventions in the adapters themselves (fs:<root>,
    s3:<bucket>/<prefix>, http:<id>, webdav:<id>).
    """
    if isinstance(cfg, FSSourceConfig):
        return FSSource(
            root=cfg.root,
            ignore_patterns=cfg.ignore_patterns or None,
            include_patterns=cfg.include_patterns or None,
            supported_extensions=supported_extensions,
            source_id=cfg.id,
        )
    if isinstance(cfg, HTTPSourceConfig):
        if cfg.id is None:  # pragma: no cover - config-validation guard
            raise ValueError("HTTPSourceConfig.id is required")
        return HTTPSource(source_id=cfg.id, urls=cfg.urls, headers=cfg.headers)
    if isinstance(cfg, S3SourceConfig):
        return S3Source(
            uri=cfg.uri,
            storage_options=cfg.storage_options,
            ignore_patterns=cfg.ignore_patterns or None,
            include_patterns=cfg.include_patterns or None,
            supported_extensions=supported_extensions,
            source_id=cfg.id,
        )
    if isinstance(cfg, WebDAVSourceConfig):
        if cfg.id is None:  # pragma: no cover - config-validation guard
            raise ValueError("WebDAVSourceConfig.id is required")
        return WebDAVSource(
            source_id=cfg.id,
            base_url=cfg.base_url,
            username=cfg.username,
            password=cfg.password,
            headers=cfg.headers,
            ignore_patterns=cfg.ignore_patterns or None,
            include_patterns=cfg.include_patterns or None,
            supported_extensions=supported_extensions,
        )
    raise TypeError(  # pragma: no cover - discriminator union exhausts all cases
        f"Unsupported source config: {type(cfg).__name__}"
    )

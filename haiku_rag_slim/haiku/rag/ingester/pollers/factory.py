from haiku.rag.config import (
    FSSourceConfig,
    HTTPSourceConfig,
    PluginSourceConfig,
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
from haiku.rag.ingester.sources.plugins import (
    ENTRY_POINT_GROUP,
    load_source_factories,
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
            max_file_size=cfg.max_file_size,
        )
    if isinstance(cfg, HTTPSourceConfig):
        return HTTPSource(
            source_id=cfg.id,
            urls=cfg.urls,
            headers=cfg.headers,
            max_file_size=cfg.max_file_size,
        )
    if isinstance(cfg, S3SourceConfig):
        return S3Source(
            uri=cfg.uri,
            storage_options=cfg.storage_options,
            ignore_patterns=cfg.ignore_patterns or None,
            include_patterns=cfg.include_patterns or None,
            supported_extensions=supported_extensions,
            source_id=cfg.id,
            max_file_size=cfg.max_file_size,
        )
    if isinstance(cfg, WebDAVSourceConfig):
        return WebDAVSource(
            source_id=cfg.id,
            base_url=cfg.base_url,
            username=cfg.username,
            password=cfg.password,
            headers=cfg.headers,
            ignore_patterns=cfg.ignore_patterns or None,
            include_patterns=cfg.include_patterns or None,
            supported_extensions=supported_extensions,
            max_file_size=cfg.max_file_size,
        )
    if isinstance(cfg, PluginSourceConfig):
        factories = load_source_factories()
        try:
            entry_point = factories[cfg.plugin]
        except KeyError:
            raise ValueError(
                f"Source {cfg.id!r} references unknown source plugin "
                f"{cfg.plugin!r}; no entry point registered under "
                f"{ENTRY_POINT_GROUP!r}."
            ) from None
        source = entry_point.load()(
            source_id=cfg.id,
            options=cfg.options,
            supported_extensions=supported_extensions,
            max_file_size=cfg.max_file_size,
        )
        if not isinstance(source, Source):
            raise TypeError(
                f"Source plugin {cfg.plugin!r} returned "
                f"{type(source).__name__}, which does not satisfy the "
                f"Source protocol."
            )
        return source
    raise TypeError(  # pragma: no cover - discriminator union exhausts all cases
        f"Unsupported source config: {type(cfg).__name__}"
    )

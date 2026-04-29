from typing import Any


def make_s3_store(bucket: str, storage_options: dict[str, str] | None) -> Any:
    """Build an obstore `S3Store` from LanceDB-style storage_options.

    Accepts the same dict shape as `LanceDBConfig.storage_options` (the same
    Rust `object_store` crate is used by both LanceDB and obstore, so the keys
    line up). Recognized keys include aws_access_key_id, aws_secret_access_key,
    aws_session_token, region (or aws_region), endpoint (or aws_endpoint),
    allow_http. Empty/missing keys fall back to the AWS default credential
    chain (environment variables, IAM role, AWS profile).
    """
    try:
        from obstore.store import (
            S3Store,  # type: ignore[import-not-found]
        )
    except ImportError as e:
        raise ImportError(
            "obstore is required for s3:// sources. "
            "Install with: pip install haiku.rag-slim[s3]"
        ) from e

    options = dict(storage_options or {})
    allow_http = str(options.pop("allow_http", "")).lower() == "true"

    # Custom endpoints (SeaweedFS, MinIO, etc.) need path-style requests; AWS
    # accepts virtual-hosted-style by default.
    has_custom_endpoint = bool(options.get("endpoint") or options.get("aws_endpoint"))

    kwargs: dict[str, Any] = {}
    if options:
        kwargs["config"] = options
    if allow_http:
        kwargs["client_options"] = {"allow_http": True}
    if has_custom_endpoint:
        kwargs["virtual_hosted_style_request"] = False

    return S3Store(bucket, **kwargs)

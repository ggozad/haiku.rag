from typing import Any


def make_s3_session(
    storage_options: dict[str, str] | None,
) -> tuple[Any, dict[str, Any]]:
    """Build an aioboto3 Session and S3 client kwargs from LanceDB-style options.

    Accepts the same dict shape as `LanceDBConfig.storage_options` so users can
    copy-paste their LanceDB credentials. Recognized keys: aws_access_key_id,
    aws_secret_access_key, aws_session_token, region (or region_name), endpoint
    (or endpoint_url), allow_http. Empty/missing keys fall back to the boto3
    default credential chain (environment variables, IAM role, AWS profile).
    """
    try:
        import aioboto3  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "aioboto3 is required for s3:// sources. "
            "Install with: pip install haiku.rag-slim[s3]"
        ) from e

    storage_options = storage_options or {}
    session_kwargs: dict[str, Any] = {}
    client_kwargs: dict[str, Any] = {}

    for key in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token"):
        if key in storage_options:
            session_kwargs[key] = storage_options[key]

    region = storage_options.get("region") or storage_options.get("region_name")
    if region:
        session_kwargs["region_name"] = region

    endpoint = storage_options.get("endpoint") or storage_options.get("endpoint_url")
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint

    if str(storage_options.get("allow_http", "")).lower() == "true":
        client_kwargs["use_ssl"] = False

    return aioboto3.Session(**session_kwargs), client_kwargs

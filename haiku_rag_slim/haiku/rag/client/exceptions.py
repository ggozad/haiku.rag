class UnsupportedSourceError(ValueError):
    """A source URI or file cannot be ingested and never will be on a retry —
    unsupported extension, unsupported content type, missing file, malformed
    S3 URI, etc. ``ValueError`` for backward compatibility with callers that
    used to catch a plain ValueError; the ingester pipeline catches the
    specific type to classify as ``PermanentError`` without string matching.
    """

class UnsupportedSourceError(ValueError):
    """A source URI or file cannot be ingested and never will be on a retry —
    unsupported extension, unsupported content type, missing file, malformed
    S3 URI, etc. ``ValueError`` for backward compatibility with callers that
    used to catch a plain ValueError; the ingester pipeline catches the
    specific type to classify as ``PermanentError`` without string matching.
    """


class AttachmentDeletionError(ValueError):
    """A PDF attachment cannot be deleted on its own while its parent is stored."""


class DirectoryIngestError(Exception):
    """Some files of a directory failed to ingest while the rest went in.

    `documents` holds what was ingested, `failures` maps each failed file to
    its reason.
    """

    def __init__(self, documents: list, failures: dict[str, str]):
        super().__init__(
            f"{len(failures)} of {len(documents) + len(failures)} files failed "
            "to ingest"
        )
        self.documents = documents
        self.failures = failures

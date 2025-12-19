class ReadOnlyError(Exception):
    """Raised when a write operation is attempted on a read-only store."""

    pass

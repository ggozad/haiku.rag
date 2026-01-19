class ReadOnlyError(Exception):
    """Raised when a write operation is attempted on a read-only store."""

    pass


class MigrationRequiredError(Exception):
    """Database requires migration. Run 'haiku-rag migrate' to upgrade."""

    pass

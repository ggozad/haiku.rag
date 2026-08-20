class ReadOnlyError(Exception):
    """Raised when a write operation is attempted on a read-only store."""

    pass


class MigrationRequiredError(Exception):
    """Database requires migration. Run 'haiku-rag migrate' to upgrade."""

    pass


class AmbiguousDatabaseError(Exception):
    """A command that works on one database was run against a configured set."""


class SourceUnavailableError(Exception):
    """A configured database could not be opened.

    Carries the configured name and never the location: a path or URI in an
    error message travels into logs and into whatever a consumer renders, and
    the point of naming databases is that locations stay in the configuration.
    """

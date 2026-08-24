class ReadOnlyError(Exception):
    """Raised when a write operation is attempted on a read-only store."""

    pass


class ConfigMismatchError(Exception):
    """Raised when stored config doesn't match current config."""

    pass


class MigrationRequiredError(Exception):
    """Database requires migration. Run 'haiku-rag migrate' to upgrade."""

    pass


class AmbiguousDatabaseError(Exception):
    """An operation that works on one database was asked of a configured set.

    Raised by the CLI for a command that cannot tell which database to use, and
    by the client for a method that has no meaning across several.
    """


class SourceUnavailableError(Exception):
    """A configured database could not be opened.

    Carries the configured name and never the location: a path or URI in an
    error message travels into logs and into whatever a consumer renders, and
    the point of naming databases is that locations stay in the configuration.
    """

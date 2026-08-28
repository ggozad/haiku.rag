from dataclasses import dataclass
from pathlib import Path

from haiku.rag.config import AppConfig
from haiku.rag.store.exceptions import (
    AmbiguousDatabaseError,
    UnknownDatabaseError,
)
from haiku.rag.utils import locate_database


@dataclass(frozen=True)
class DatabaseRef:
    """A resolved database location, and the configured name it answers to.

    Exactly one of ``uri`` and ``db_path`` is set. ``name`` is the key from
    ``lancedb.databases``, and the only identity that leaves the configuration:
    it travels in results, citations and errors, where a location must not.
    None where nothing names the database.
    """

    name: str | None
    uri: str
    db_path: Path | None

    def __post_init__(self) -> None:
        if bool(self.uri) == (self.db_path is not None):
            raise ValueError(
                "a database is either a URI or a local path: "
                f"got uri={self.uri!r} and db_path={self.db_path!r}"
            )

    @classmethod
    def at(cls, path: Path | str, *, name: str | None = None) -> "DatabaseRef":
        """A database at a path the caller named, taken as given."""
        return cls(name=name, uri="", db_path=Path(path))

    @classmethod
    def configured(cls, name: str | None, location: str) -> "DatabaseRef":
        """A database the configuration placed, by ``lancedb.uri`` or an entry in
        ``lancedb.databases``. A location carrying a scheme is a URI, anything
        else a local path."""
        uri, db_path = locate_database(location)
        return cls(name=name, uri=uri, db_path=db_path)

    def connection(self, config: AppConfig) -> tuple[AppConfig, Path | None]:
        """The configuration and path to open this one database with.

        A copy: the caller's configuration still names whatever set it named.
        """
        one = config.model_copy(deep=True)
        one.lancedb.databases = {}
        one.lancedb.uri = self.uri
        return one, self.db_path


@dataclass(frozen=True)
class DatabaseScope:
    """The databases an operation covers.

    Resolved once, from configuration plus at most one selector, then passed
    down. Never empty.

    Nothing here reads the environment: ``HAIKU_RAG_DB`` is the capability entry
    point's to honour.
    """

    databases: tuple[DatabaseRef, ...]

    def __post_init__(self) -> None:
        if not self.databases:
            raise ValueError("a scope covers at least one database")

    @classmethod
    def resolve(
        cls,
        config: AppConfig,
        *,
        database_name: str | None = None,
        database_path: Path | str | None = None,
    ) -> "DatabaseScope":
        """The databases named by `config` and at most one selector.

        A path names one database that nothing calls anything; a name selects one
        of the configured set and keeps its name. With no selector the configured
        set is covered in configuration order, a set of one included.
        """
        if database_name is not None and database_path is not None:
            raise AmbiguousDatabaseError(
                "a database name and a database path both name one database; "
                "pass one of them"
            )

        declared = config.lancedb.databases

        if database_path is not None:
            return cls((DatabaseRef.at(database_path),))

        if database_name is not None:
            if database_name not in declared:
                raise UnknownDatabaseError(
                    f"unknown database {database_name!r}; lancedb.databases names "
                    f"{', '.join(sorted(declared)) or 'nothing'}"
                )
            return cls(
                (DatabaseRef.configured(database_name, declared[database_name]),)
            )

        if declared:
            return cls(
                tuple(
                    DatabaseRef.configured(name, location)
                    for name, location in declared.items()
                )
            )

        if config.lancedb.uri:
            return cls((DatabaseRef.configured(None, config.lancedb.uri),))

        return cls((DatabaseRef.at(config.storage.data_dir / "haiku.rag.lancedb"),))

    def select(self, names: list[str]) -> "DatabaseScope":
        """The databases in this scope named by `names`, in the order given.

        Repeats collapse: a name selects its database once.
        """
        if not names:
            raise ValueError(
                "sources=[] selects no database; pass None for all of them"
            )
        by_name = {ref.name: ref for ref in self.databases if ref.name is not None}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise UnknownDatabaseError(
                f"unknown database(s) {', '.join(sorted(missing))}; "
                f"configured: {', '.join(sorted(by_name))}"
            )
        return DatabaseScope(tuple(by_name[name] for name in dict.fromkeys(names)))

    @property
    def covers_multiple(self) -> bool:
        """Whether this scope covers more than one database."""
        return len(self.databases) > 1

    @property
    def names(self) -> tuple[str, ...]:
        """The configured names covered, in order. Empty where none is named."""
        return tuple(ref.name for ref in self.databases if ref.name is not None)

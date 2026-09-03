from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from haiku.rag.config import AppConfig
from haiku.rag.store.exceptions import (
    AmbiguousDatabaseError,
    UnknownDatabaseError,
)
from haiku.rag.utils import locate_database

DEFAULT_DATABASE_FILENAME = "haiku.rag.lancedb"


def database_name(path: Path) -> str:
    """The name a database at `path` answers to: the path's stem."""
    if not path.stem:
        raise ValueError(f"a database at {path} has no name: the path has no stem")
    return path.stem


@dataclass(frozen=True)
class DatabaseRef:
    """A resolved database: the name it answers to, and where it is.

    ``name`` is the key from ``lancedb.databases``, or the stem of a path the
    caller gave. It is the only identity that leaves the configuration: it
    travels in results, citations and errors, where a location must not.
    ``location`` is a local path, or a URI. ``given`` marks a path the caller
    gave, whose errors may name it: the caller already knows where it is.
    """

    name: str
    location: Path | str
    given: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(f"a database at {self.location} has no name")
        if isinstance(self.location, str):
            if not self.location.strip():
                raise ValueError(f"database {self.name!r} has no location")
            object.__setattr__(self, "location", locate_database(self.location))
        if self.given and not isinstance(self.location, Path):
            raise ValueError(
                f"database {self.name!r} is given as a path, and {self.location} "
                "is a URI"
            )

    @classmethod
    def at(cls, path: Path | str) -> "DatabaseRef":
        """A database at a path the caller named, taken as given."""
        path = Path(path)
        return cls(name=database_name(path), location=path, given=True)

    @classmethod
    def configured(cls, name: str, location: str | Path) -> "DatabaseRef":
        """A database the configuration placed. A location carrying a scheme is
        a URI, anything else a local path."""
        return cls(name=name, location=location)

    @property
    def db_path(self) -> Path | None:
        """The local path, or None for a database behind a URI."""
        return self.location if isinstance(self.location, Path) else None


@dataclass(frozen=True)
class DatabaseScope:
    """The databases an operation covers.

    Resolved once, from configuration plus at most one selector, then passed
    down. Never empty. Nothing here reads the environment.
    """

    databases: tuple[DatabaseRef, ...]

    def __post_init__(self) -> None:
        if not self.databases:
            raise ValueError("a scope covers at least one database")

    @classmethod
    def at(cls, path: Path | str) -> "DatabaseScope":
        """One database at a path the caller named, whatever is configured.

        The CLI's ``--db``: a human typing a path means that database.
        """
        return cls((DatabaseRef.at(path),))

    @classmethod
    def resolve(
        cls,
        config: AppConfig,
        *,
        database_name: str | None = None,
        database_path: Path | str | None = None,
    ) -> "DatabaseScope":
        """The databases named by `config` and at most one selector.

        The configuration places databases: ``lancedb.databases``, or where it
        names none, the default database under ``storage.data_dir`` as the entry
        ``haiku.rag``. A name selects one of them. A path places a database
        where the configuration places none, and is refused beside one it does.
        """
        if database_name is not None and database_path is not None:
            raise AmbiguousDatabaseError(
                "a database name and a database path both name one database; "
                "pass one of them"
            )

        configured = config.lancedb.databases

        if database_path is not None:
            if configured:
                raise AmbiguousDatabaseError(
                    "a database path and lancedb.databases both place the "
                    f"database: db_path={Path(database_path)} and databases "
                    f"name {', '.join(sorted(configured))}; pass one of them"
                )
            return cls.at(database_path)

        declared: Mapping[str, str | Path] = configured or {
            "haiku.rag": config.storage.data_dir / DEFAULT_DATABASE_FILENAME
        }

        if database_name is not None:
            if database_name not in declared:
                raise UnknownDatabaseError(
                    f"unknown database {database_name!r}; lancedb.databases names "
                    f"{', '.join(sorted(declared))}"
                )
            return cls(
                (DatabaseRef.configured(database_name, declared[database_name]),)
            )

        return cls(
            tuple(
                DatabaseRef.configured(name, location)
                for name, location in declared.items()
            )
        )

    def select(self, names: list[str]) -> "DatabaseScope":
        """The databases in this scope named by `names`, in the order given.

        Repeats collapse: a name selects its database once.
        """
        if not names:
            raise ValueError(
                "sources=[] selects no database; pass None for all of them"
            )
        by_name = {ref.name: ref for ref in self.databases}
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
        """The names of the databases covered, in order."""
        return tuple(ref.name for ref in self.databases)

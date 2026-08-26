from dataclasses import dataclass
from pathlib import Path

from haiku.rag.config import AppConfig
from haiku.rag.store.exceptions import AmbiguousDatabaseError
from haiku.rag.utils import locate_database


@dataclass(frozen=True)
class DatabaseRef:
    """One database, and the name it answers to.

    ``name`` is the key from ``lancedb.databases``, and the only identity that
    leaves the configuration: it travels in results, citations and the errors an
    operator or a model sees, where a location must not. The invariant below is
    the exception, and deliberately so — a malformed ref is a programming error
    raised in the caller's own process, where naming what it was given is what
    makes it fixable. None where nothing names the database — a path given
    directly, or the legacy single ``uri``.

    Location is resolved once, on construction, into exactly one of ``uri`` and
    ``db_path``. Keeping the configured string and re-reading it later would let a
    path the caller gave be reinterpreted as a URI because it happens to carry a
    scheme.
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
        """A database the configuration placed, by ``lancedb.uri`` or by an entry
        in ``lancedb.databases``. A value carrying a scheme is a URI and anything
        else is a local path, so the two settings place a database alike."""
        uri, db_path = locate_database(location)
        return cls(name=name, uri=uri, db_path=db_path)

    def connection(self, config: AppConfig) -> tuple[AppConfig, Path | None]:
        """The configuration and path to open this one database with.

        A copy. The scope is resolved once from the caller's configuration, and
        rewriting that configuration in place is what left downstream code unable
        to tell that a set had been named.
        """
        one = config.model_copy(deep=True)
        one.lancedb.databases = {}
        one.lancedb.uri = self.uri
        return one, self.db_path


@dataclass(frozen=True)
class DatabaseScope:
    """The databases an operation covers.

    Resolved once, from configuration plus at most one selector, and passed down
    rather than re-derived. Never empty: every resolution reaches a database, and
    the sessions built from a scope have no meaning without one.

    Nothing here reads the environment. ``HAIKU_RAG_DB`` is honoured by the
    capability entry point alone, which passes it as ``database_path``, so
    resolving a scope cannot quietly change what any other caller opens.
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
                raise AmbiguousDatabaseError(
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

        Repeats collapse: a database named twice would be searched twice and
        fused as two rank lists, which counts it double.
        """
        if not names:
            raise ValueError(
                "sources=[] selects no database; pass None for all of them"
            )
        by_name = {ref.name: ref for ref in self.databases if ref.name is not None}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise KeyError(
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

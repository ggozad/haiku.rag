from pathlib import Path

import pytest

from haiku.rag.client.scope import DatabaseRef, DatabaseScope
from haiku.rag.config.models import AppConfig, LanceDBConfig, StorageConfig
from haiku.rag.store.exceptions import (
    AmbiguousDatabaseError,
    UnknownDatabaseError,
)


def _config(**kwargs) -> AppConfig:
    return AppConfig(lancedb=LanceDBConfig(**kwargs))


class TestResolution:
    """One selector at most, and the same answer wherever it is asked."""

    def test_a_name_and_a_path_together_are_refused(self):
        config = _config(databases={"alpha": "/data/alpha.lancedb"})

        with pytest.raises(AmbiguousDatabaseError, match="pass one of them"):
            DatabaseScope.resolve(
                config, database_name="alpha", database_path=Path("/data/other.lancedb")
            )

    def test_a_path_names_one_unnamed_database(self):
        """A path says which database, not what it is called, even where the
        configuration names one."""
        config = _config(databases={"alpha": "/data/alpha.lancedb"})

        scope = DatabaseScope.resolve(config, database_path=Path("/data/other.lancedb"))

        assert scope.databases == (DatabaseRef.at("/data/other.lancedb"),)
        assert scope.names == ()
        assert not scope.covers_multiple

    def test_a_named_database_keeps_its_name(self):
        config = _config(databases={"alpha": "/data/alpha.lancedb", "beta": "b://b"})

        scope = DatabaseScope.resolve(config, database_name="beta")

        assert scope.databases == (DatabaseRef("beta", "b://b", None),)
        assert scope.names == ("beta",)

    def test_an_unknown_name_is_refused(self):
        config = _config(databases={"alpha": "/data/alpha.lancedb"})

        with pytest.raises(UnknownDatabaseError, match="unknown database 'nope'"):
            DatabaseScope.resolve(config, database_name="nope")

    def test_no_selector_covers_the_configured_set_in_order(self):
        config = _config(
            databases={"beta": "/data/b.lancedb", "alpha": "/data/a.lancedb"}
        )

        scope = DatabaseScope.resolve(config)

        assert scope.names == ("beta", "alpha")
        assert scope.covers_multiple

    def test_a_configured_set_of_one_is_still_a_named_database(self):
        """Its name is what results and citations carry, so it survives."""
        config = _config(databases={"alpha": "/data/alpha.lancedb"})

        scope = DatabaseScope.resolve(config)

        assert scope.databases == (
            DatabaseRef.configured("alpha", "/data/alpha.lancedb"),
        )
        assert not scope.covers_multiple

    def test_a_bare_uri_is_one_unnamed_database(self):
        scope = DatabaseScope.resolve(_config(uri="s3://bucket/one.lancedb"))

        assert scope.databases == (DatabaseRef(None, "s3://bucket/one.lancedb", None),)

    def test_a_bare_uri_without_a_scheme_is_a_local_path(self):
        """`lancedb.uri` places one database the same way an entry in
        `lancedb.databases` does, so a schemeless value is a path and gets the
        existence check a local database gets."""
        scope = DatabaseScope.resolve(_config(uri="/data/notes.lancedb"))

        [ref] = scope.databases
        assert ref.name is None
        assert ref.db_path == Path("/data/notes.lancedb")
        assert ref.uri == ""

    def test_a_path_selects_the_database_over_a_configured_uri(self):
        """`--db` exists to override what is configured."""
        config = _config(uri="s3://bucket/one.lancedb")

        scope = DatabaseScope.resolve(config, database_path=Path("/data/local"))

        [ref] = scope.databases
        assert ref.location == Path("/data/local")

    def test_nothing_configured_falls_back_to_the_data_directory(self, tmp_path):
        config = AppConfig(storage=StorageConfig(data_dir=tmp_path))

        scope = DatabaseScope.resolve(config)

        assert scope.databases == (DatabaseRef.at(tmp_path / "haiku.rag.lancedb"),)

    def test_the_environment_is_not_consulted(self, monkeypatch, tmp_path):
        """HAIKU_RAG_DB is honoured by the capability entry point alone;
        resolution never reads the environment."""
        monkeypatch.setenv("HAIKU_RAG_DB", "/data/from-the-environment.lancedb")
        config = _config(databases={"alpha": "/data/alpha.lancedb"})

        scope = DatabaseScope.resolve(config)

        assert scope.names == ("alpha",)

    def test_a_path_is_never_reinterpreted_as_a_uri(self):
        """A caller naming a path means that path, whatever scheme it carries."""
        scope = DatabaseScope.resolve(
            _config(), database_path="s3://bucket/looks-like-a-uri.lancedb"
        )

        [ref] = scope.databases
        assert ref.db_path == Path("s3://bucket/looks-like-a-uri.lancedb")
        assert ref.uri == ""

    def test_a_configured_location_with_a_scheme_is_a_uri(self):
        """A configured value is a URI or a path depending on its scheme, which is
        what makes it different from a path the caller gave."""
        config = _config(databases={"alpha": "s3://bucket/alpha.lancedb"})

        [ref] = DatabaseScope.resolve(config).databases

        assert ref.uri == "s3://bucket/alpha.lancedb"
        assert ref.db_path is None

    def test_a_database_is_a_uri_or_a_path(self):
        """A ref holding both, or neither, is refused at construction.

        The message names what it was given: this is a programming error raised
        in the caller's own process, not one an operator or a model ever sees.
        """
        with pytest.raises(ValueError, match="either a URI or a local path") as both:
            DatabaseRef(None, "s3://bucket/a.lancedb", Path("/data/a.lancedb"))
        assert "s3://bucket/a.lancedb" in str(both.value)

        with pytest.raises(ValueError, match="either a URI or a local path") as neither:
            DatabaseRef(None, "", None)
        assert "db_path=None" in str(neither.value)

    def test_a_scope_covers_at_least_one_database(self):
        """Every resolution reaches a database, and the sessions built from a
        scope have no meaning without one."""
        with pytest.raises(ValueError, match="at least one database"):
            DatabaseScope(())


class TestLocation:
    """One value says where a database is: a path for a local one, a URI string
    for a remote one. Storage connects to it as given."""

    def test_a_local_location_is_a_path(self):
        config = _config(databases={"alpha": "/data/alpha.lancedb"})
        [ref] = DatabaseScope.resolve(config).databases

        assert ref.location == Path("/data/alpha.lancedb")

    def test_a_uri_location_is_the_uri(self):
        config = _config(databases={"alpha": "s3://bucket/alpha.lancedb"})
        [ref] = DatabaseScope.resolve(config).databases

        assert ref.location == "s3://bucket/alpha.lancedb"

    def test_a_path_the_caller_gave_is_its_location(self):
        assert DatabaseRef.at("/data/other.lancedb").location == Path(
            "/data/other.lancedb"
        )

from pathlib import Path

import pytest

from haiku.rag.client.scope import DatabaseRef, DatabaseScope, database_name
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

    def test_a_path_places_the_database_where_the_configuration_places_none(self):
        scope = DatabaseScope.resolve(_config(), database_path="/data/other.lancedb")

        assert scope.databases == (DatabaseRef.at("/data/other.lancedb"),)
        assert scope.names == ("other",)
        assert not scope.covers_multiple

    def test_a_path_beside_a_configured_placement_is_refused(self):
        """The configuration places databases; a path beside it is a second
        placement, and the refusal names both."""
        config = _config(databases={"alpha": "/data/alpha.lancedb", "beta": "b://b"})

        with pytest.raises(AmbiguousDatabaseError) as raised:
            DatabaseScope.resolve(config, database_path=Path("/data/other.lancedb"))

        message = str(raised.value)
        assert "/data/other.lancedb" in message
        assert "alpha" in message and "beta" in message
        assert "lancedb.databases" in message

    def test_a_named_database_keeps_its_name(self):
        config = _config(databases={"alpha": "/data/alpha.lancedb", "beta": "b://b"})

        scope = DatabaseScope.resolve(config, database_name="beta")

        assert scope.databases == (DatabaseRef("beta", "b://b"),)
        assert scope.names == ("beta",)

    def test_an_unknown_name_is_refused(self):
        """The message lists the databases there are, configured or default."""
        config = _config(databases={"alpha": "/data/alpha.lancedb"})

        with pytest.raises(
            UnknownDatabaseError,
            match="unknown database 'nope'.*the databases are alpha",
        ):
            DatabaseScope.resolve(config, database_name="nope")

        with pytest.raises(
            UnknownDatabaseError,
            match="unknown database 'nope'.*the databases are haiku.rag",
        ):
            DatabaseScope.resolve(_config(), database_name="nope")

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

        assert scope.databases == (DatabaseRef("alpha", Path("/data/alpha.lancedb")),)
        assert not scope.covers_multiple

    def test_nothing_configured_is_the_default_entry(self, tmp_path):
        """No `databases` reads as one entry, `haiku.rag`, under the data
        directory: an ordinary configured database in every respect."""
        config = AppConfig(storage=StorageConfig(data_dir=tmp_path))

        scope = DatabaseScope.resolve(config)

        assert scope.databases == (
            DatabaseRef("haiku.rag", tmp_path / "haiku.rag.lancedb"),
        )
        assert scope.names == ("haiku.rag",)

    def test_the_default_entry_is_selectable_by_name(self, tmp_path):
        config = AppConfig(storage=StorageConfig(data_dir=tmp_path))

        by_name = DatabaseScope.resolve(config, database_name="haiku.rag")
        selected = DatabaseScope.resolve(config).select(["haiku.rag"])

        assert by_name == selected == DatabaseScope.resolve(config)

    def test_the_default_entry_does_not_hide_a_configured_set(self, tmp_path):
        """The default stands in only where nothing is configured."""
        config = AppConfig(
            storage=StorageConfig(data_dir=tmp_path),
            lancedb=LanceDBConfig(databases={"alpha": "/data/alpha.lancedb"}),
        )

        with pytest.raises(UnknownDatabaseError, match="haiku.rag"):
            DatabaseScope.resolve(config, database_name="haiku.rag")

    def test_the_environment_is_not_consulted(self, monkeypatch):
        """Resolution reads the configuration alone."""
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
        assert ref.location == Path("s3://bucket/looks-like-a-uri.lancedb")

    def test_a_configured_location_with_a_scheme_is_a_uri(self):
        """A configured value is a URI or a path depending on its scheme, which is
        what makes it different from a path the caller gave."""
        config = _config(databases={"alpha": "s3://bucket/alpha.lancedb"})

        [ref] = DatabaseScope.resolve(config).databases

        assert ref.location == "s3://bucket/alpha.lancedb"
        assert ref.db_path is None

    def test_a_configured_location_without_a_scheme_is_a_path(self):
        config = _config(databases={"alpha": "/data/alpha.lancedb"})

        [ref] = DatabaseScope.resolve(config).databases

        assert ref.location == Path("/data/alpha.lancedb")
        assert ref.db_path == Path("/data/alpha.lancedb")

    def test_a_scope_covers_at_least_one_database(self):
        """Every resolution reaches a database, and the sessions built from a
        scope have no meaning without one."""
        with pytest.raises(ValueError, match="at least one database"):
            DatabaseScope(())


class TestTheReference:
    """Constructed directly, a reference still holds what it advertises."""

    def test_a_schemeless_string_location_is_a_path(self):
        ref = DatabaseRef("x", "local.lancedb")

        assert ref.location == Path("local.lancedb")
        assert ref.db_path == Path("local.lancedb")

    def test_a_location_with_a_scheme_stays_a_uri(self):
        ref = DatabaseRef("x", "s3://bucket/x.lancedb")

        assert ref.location == "s3://bucket/x.lancedb"
        assert ref.db_path is None

    def test_a_blank_name_is_refused(self):
        with pytest.raises(ValueError, match="no name"):
            DatabaseRef("", "/data/x.lancedb")
        with pytest.raises(ValueError, match="no name"):
            DatabaseRef("  ", "s3://bucket/x.lancedb")

    def test_a_blank_location_is_refused(self):
        """A blank string would resolve to the working directory."""
        with pytest.raises(ValueError, match="no location"):
            DatabaseRef("x", "")
        with pytest.raises(ValueError, match="no location"):
            DatabaseRef("x", "   ")

    def test_a_given_database_is_a_local_path(self):
        """Only a path can be given: a given database's errors name its
        location, and a URI must never travel that way."""
        assert DatabaseRef("x", "local.lancedb", given=True).location == Path(
            "local.lancedb"
        )
        with pytest.raises(ValueError, match="is a URI"):
            DatabaseRef("x", "s3://bucket/x.lancedb", given=True)


class TestNamingAPath:
    """A path the caller gave is named by its stem, the one rule for the
    default database and for `--db`."""

    def test_the_stem_names_the_database(self):
        assert database_name(Path("/data/foo.lancedb")) == "foo"
        assert database_name(Path("relative.lancedb")) == "relative"
        assert database_name(Path("/data/haiku.rag.lancedb")) == "haiku.rag"

    def test_a_path_with_no_stem_is_refused(self):
        with pytest.raises(ValueError, match="no name"):
            database_name(Path("/"))

    def test_at_names_the_path_it_is_given(self):
        """A given path is marked as such: its errors may name it, since the
        caller already knows where it is."""
        assert DatabaseRef.at("/data/other.lancedb") == DatabaseRef(
            "other", Path("/data/other.lancedb"), given=True
        )
        assert not DatabaseRef.configured("other", "/data/other.lancedb").given

    def test_a_scope_at_a_path_ignores_the_configuration(self):
        """The CLI's `--db` is a human's explicit override: it constructs the
        scope directly and consults no configuration."""
        scope = DatabaseScope.at(Path("/data/other.lancedb"))

        assert scope.databases == (DatabaseRef.at("/data/other.lancedb"),)
        assert scope.names == ("other",)

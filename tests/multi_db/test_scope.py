"""Resolving which databases an operation covers."""

import pytest
from pydantic import ValidationError

from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseScope
from haiku.rag.config.models import AppConfig, LanceDBConfig
from haiku.rag.utils import locate_database
from tests.multi_db.helpers import (
    _config,
    _seed,
)


class TestConfig:
    def test_databases_and_uri_are_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="databases"):
            LanceDBConfig(
                uri="s3://b/one.lancedb", databases={"one": "s3://b/one.lancedb"}
            )

    def test_databases_alone_is_fine(self):
        config = LanceDBConfig(databases={"one": "s3://b/one.lancedb"})
        assert config.databases == {"one": "s3://b/one.lancedb"}

    def test_uri_alone_is_fine(self):
        assert LanceDBConfig(uri="s3://b/one.lancedb").databases == {}


class TestNamingIsRequired:
    def test_a_blank_name_is_rejected(self):
        """An unnamed database is unreachable: every source check reads the
        empty name as no name at all."""
        with pytest.raises(ValidationError, match="entry with no name"):
            LanceDBConfig(databases={"": "/tmp/a.lancedb"})
        with pytest.raises(ValidationError, match="entry with no name"):
            LanceDBConfig(databases={"   ": "/tmp/a.lancedb"})

    def test_a_blank_location_is_rejected(self):
        """A blank location resolves to the working directory."""
        with pytest.raises(
            ValidationError, match=r"databases\[alpha\] has no location"
        ):
            LanceDBConfig(databases={"alpha": ""})


class TestNamingADatabaseDirectly:
    @pytest.mark.asyncio
    async def test_an_explicit_db_path_wins_over_the_configured_set(
        self, tmp_path, temp_db_path
    ):
        """A caller that names a path means that database, not the configured
        set: the CLI resolves `--db` to one and must not fan out instead."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(temp_db_path, config=config, create=True) as rag:
            assert not rag.covers_multiple
            assert rag.source is None
            assert rag.store.db_path == temp_db_path

    @pytest.mark.asyncio
    async def test_one_configured_database_is_opened_by_name(self, tmp_path):
        """A set of one is not federated, and the client resolves it."""
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert not rag.covers_multiple
            assert rag.source == "alpha"
            results = await rag.search("cats", search_type="fts", limit=10)

        assert [r.source for r in results] == ["alpha"]


class TestOneConfiguredLocation:
    """`lancedb.uri` places one unnamed database, at a URI or at a local path."""

    def _config(self, location) -> AppConfig:
        return AppConfig(lancedb=LanceDBConfig(uri=str(location)))

    @pytest.mark.asyncio
    async def test_a_local_uri_opens_the_configured_database(self, tmp_path):
        located = tmp_path / "notes.lancedb"
        config = self._config(located)

        async with HaikuRAG(config=config, create=True) as rag:
            assert rag.store.db_path == located
            # It places a database without naming one: only `lancedb.databases`
            # assigns the name results and citations carry.
            assert rag.source is None
        assert located.exists()

    @pytest.mark.asyncio
    async def test_an_explicit_path_overrides_a_local_uri(self, tmp_path):
        """`--db` overrides the configured location for one invocation."""
        config = self._config(tmp_path / "configured.lancedb")
        chosen = tmp_path / "chosen.lancedb"

        async with HaikuRAG(chosen, config=config, create=True) as rag:
            assert rag.store.db_path == chosen
        assert chosen.exists()
        assert not (tmp_path / "configured.lancedb").exists()

    @pytest.mark.asyncio
    async def test_a_local_uri_that_does_not_exist_is_refused(self, tmp_path):
        """A mistyped path fails instead of quietly becoming an empty database,
        which is what a value carrying a scheme would do."""
        config = self._config(tmp_path / "typo.lancedb")

        with pytest.raises(FileNotFoundError):
            async with HaikuRAG(config=config):
                pass
        assert not (tmp_path / "typo.lancedb").exists()

    def test_a_uri_with_a_scheme_stays_a_uri(self, tmp_path):
        """Object storage has no local path to check, and a location that does
        not exist yet is normal there."""
        from haiku.rag.store.engine import ConnectionMode

        config = self._config("s3://bucket/one.lancedb")

        [ref] = DatabaseScope.resolve(config).databases
        one, db_path = ref.connection(config)

        assert db_path is None
        assert ConnectionMode.from_config(one) == ConnectionMode.OBJECT_STORAGE


class TestLocate:
    def test_a_scheme_is_a_uri(self):
        assert locate_database("s3://bucket/one.lancedb") == (
            "s3://bucket/one.lancedb",
            None,
        )

    def test_anything_else_is_a_local_path(self):
        uri, db_path = locate_database("/data/one.lancedb")
        assert uri == ""
        assert db_path is not None and str(db_path) == "/data/one.lancedb"


class TestSelection:
    @pytest.mark.asyncio
    async def test_unknown_source_at_construction_is_rejected(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])

        with pytest.raises(KeyError, match="nope"):
            async with HaikuRAG(config=config, sources=["nope"]):
                pass

    @pytest.mark.asyncio
    async def test_unknown_source_across_several_databases_is_rejected(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(KeyError, match="nope"):
                await rag.search("cats", search_type="fts", sources=["nope"])

    @pytest.mark.asyncio
    async def test_no_matches_anywhere_returns_nothing(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            assert await rag.search("aardvarks", search_type="fts") == []


class TestPlacingADatabase:
    """What a client says about the databases it covers, so nothing outside has
    to read its private state to find out."""

    @pytest.mark.asyncio
    async def test_a_set_names_every_database_it_covers(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config, read_only=True) as rag:
            assert rag.covers_multiple
            assert rag.source_names == ("alpha", "beta")
            assert rag.source is None

    @pytest.mark.asyncio
    async def test_one_named_database_names_itself(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])

        async with HaikuRAG(config=config, read_only=True, sources=["alpha"]) as rag:
            assert not rag.covers_multiple
            assert rag.source_names == ("alpha",)
            assert rag.source == "alpha"

    @pytest.mark.asyncio
    async def test_a_named_database_keeps_its_name_on_re_entry(self, tmp_path):
        """Entering derives a single-database configuration from what was
        configured. Deriving it from the last derivation loses the name."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        rag = HaikuRAG(config=config, read_only=True, sources=["alpha"])
        async with rag:
            assert rag.source == "alpha"
        async with rag:
            assert rag.source == "alpha"
            assert rag.source_names == ("alpha",)
            results = await rag.search("cats", search_type="fts")

        assert {r.source for r in results} == {"alpha"}

    @pytest.mark.asyncio
    async def test_an_unnamed_database_names_nothing(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            assert rag.source_names == ()
            assert rag.source is None

    @pytest.mark.asyncio
    async def test_the_reader_for_a_database_is_the_client_holding_it(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config, read_only=True) as rag:
            reader = await rag.reader_for("beta")

            assert reader is not None
            assert reader.source == "beta"
            # Asked twice, the same wrapper comes back.
            assert await rag.reader_for("beta") is reader

    @pytest.mark.asyncio
    async def test_a_client_reading_one_database_is_its_own_reader(self, temp_db_path):
        async with HaikuRAG(temp_db_path, create=True) as rag:
            assert await rag.reader_for(None) is rag

    @pytest.mark.asyncio
    async def test_one_database_refuses_a_name_it_does_not_cover(self, tmp_path):
        """Answering with itself would hand back the wrong database's reader for
        a citation that named another."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config, sources=["alpha"]) as alpha:
            assert await alpha.reader_for("alpha") is alpha
            with pytest.raises(KeyError, match="beta"):
                await alpha.reader_for("beta")

    @pytest.mark.asyncio
    async def test_an_unnamed_database_refuses_any_name(self, temp_db_path):
        """Nothing names it, so no name can be the one it covers."""
        async with HaikuRAG(temp_db_path, create=True) as rag:
            with pytest.raises(KeyError, match="single unnamed database"):
                await rag.reader_for("anything")

    @pytest.mark.asyncio
    async def test_a_set_cannot_place_evidence_that_names_no_database(self, tmp_path):
        """Evidence recorded before databases could be named carries no source."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])

        async with HaikuRAG(config=config, read_only=True) as rag:
            assert await rag.reader_for(None) is None


class TestNamingOneOfTheSetOnTheCommandLine:
    """`--db-name NAME` reaches the application layer as a name, and every
    client it opens has to honour it — one that ignores it covers the set and
    quietly answers from the wrong database."""

    @pytest.mark.asyncio
    async def test_a_named_database_is_the_one_read(self, tmp_path, capsys):
        from haiku.rag.app import HaikuRAGApp

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        scope = DatabaseScope.resolve(config).select(["beta"])
        app = HaikuRAGApp(scope=scope, config=config, read_only=True)
        await app.list_documents()

        # Rich wraps long lines, so match the unwrapped part of the URI.
        printed = capsys.readouterr().out
        assert "test://beta/" in printed
        assert "test://alpha/" not in printed

    @pytest.mark.asyncio
    async def test_naming_none_of_them_covers_the_set(self, tmp_path, capsys):
        from haiku.rag.app import HaikuRAGApp

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        app = HaikuRAGApp(
            scope=DatabaseScope.resolve(config), config=config, read_only=True
        )
        await app.list_documents()

        printed = capsys.readouterr().out
        assert "test://alpha/" in printed
        assert "test://beta/" in printed

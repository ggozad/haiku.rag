import shutil

import pytest

from haiku.rag.client import HaikuRAG
from haiku.rag.client.scope import DatabaseRef, DatabaseScope
from haiku.rag.sandbox import AnalysisContext, Sandbox
from tests.multi_db.helpers import _config, _seed


async def _mounted(rag, sources=None):
    """The sandbox's view of the corpus, and the sandbox itself."""
    sandbox = Sandbox(
        db_path=None,
        config=rag._config,
        context=AnalysisContext(sources=sources),
        rag=rag,
    )
    docs, owners = await sandbox._documents()
    return sandbox, docs, owners


class TestStandaloneAcrossDatabases:
    """Without a lent client the sandbox opens its own. The owners it hands out
    are stored for later file reads, so that connection has to outlive the call
    that produced them."""

    @pytest.mark.asyncio
    async def test_owners_stay_open_for_later_reads(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        sandbox = Sandbox(db_path=None, config=config, context=AnalysisContext())
        try:
            _, owners = await sandbox._documents()

            assert len(owners) == 2
            assert all(owner.store.db.is_open() for owner in owners.values())
        finally:
            await sandbox.close()

        assert not any(owner.store.db.is_open() for owner in owners.values())


class TestDocumentsAcrossDatabases:
    @pytest.mark.asyncio
    async def test_the_corpus_covers_every_configured_database(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            _, docs, owners = await _mounted(rag)

        assert {d.uri for d in docs} == {
            "test://alpha/alpha document about cats",
            "test://beta/beta document about cats",
        }
        assert {owner.source for owner in owners.values()} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_selected_databases_bound_the_corpus(self, tmp_path):
        """A question scoped to one database must not mount another's documents."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            _, docs, owners = await _mounted(rag, sources=["alpha"])

        assert [d.uri for d in docs] == ["test://alpha/alpha document about cats"]
        assert {owner.source for owner in owners.values()} == {"alpha"}

    @pytest.mark.asyncio
    async def test_one_database_needs_no_owners(self, tmp_path, temp_db_path):
        """A single connection serves every read, so nothing has to be routed."""
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            _, docs, owners = await _mounted(rag)

        assert len(docs) == 1
        assert owners == {}

    @pytest.mark.asyncio
    async def test_a_document_is_read_from_the_database_holding_it(self, tmp_path):
        """Reads addressed to one document go through its owner, which is the
        only client that can answer them."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            sandbox, docs, owners = await _mounted(rag)
            sandbox._owners = owners
            for doc in docs:
                assert doc.id is not None
                async with sandbox._connection(owners[doc.id]) as owner:
                    content = await owner.document_repository.get_content(doc.id)
                assert content is not None
                assert owners[doc.id].source is not None
                assert owners[doc.id].source in content


class TestTheSandboxConstructors:
    """`Sandbox` is public and takes a path; `_covering` is for callers that
    already resolved a scope, as `HaikuRAG._covering` is."""

    @pytest.mark.asyncio
    async def test_the_public_constructor_resolves_the_path_it_is_given(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        sandbox = Sandbox(
            db_path=tmp_path / "alpha.lancedb",
            config=config,
            context=AnalysisContext(),
        )

        assert sandbox._scope.databases == (DatabaseRef.at(tmp_path / "alpha.lancedb"),)

    @pytest.mark.asyncio
    async def test_no_path_covers_what_the_configuration_places(self, tmp_path):
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        sandbox = Sandbox(db_path=None, config=config, context=AnalysisContext())

        assert sandbox._scope.names == ("alpha", "beta")

    @pytest.mark.asyncio
    async def test_covering_resolves_nothing_of_its_own(self, tmp_path, monkeypatch):
        """Handed a scope, it must not reach resolution again: resolving twice
        is what let a capability's databases and its sandbox's disagree."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        scope = DatabaseScope.resolve(config, database_name="alpha")

        def _refuse(*args, **kwargs):
            raise AssertionError("resolved a scope it was already given")

        monkeypatch.setattr(DatabaseScope, "resolve", _refuse)
        sandbox = Sandbox._covering(scope, config, AnalysisContext())

        assert sandbox._scope is scope
        assert sandbox._config is config


class TestTheSandboxCoversWhatTheCapabilityCovers:
    @pytest.mark.asyncio
    async def test_the_capability_hands_over_the_scope_it_resolved(self, tmp_path):
        """The capability resolved its databases once. Letting the sandbox
        resolve them again from the same configuration reaches a different
        answer wherever a path or the environment named one of a set."""
        from haiku.rag.capabilities.analysis import AnalysisState, create_capability

        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])

        capability = create_capability(
            db_path=tmp_path / "alpha.lancedb", config=config, defer_loading=False
        )
        capability.state = AnalysisState()

        sandbox = await capability._ensure_sandbox()
        try:
            assert sandbox._scope is capability.scope
            assert capability.scope.names == ()
        finally:
            await capability._close()


class TestExecutingAcrossDatabases:
    @pytest.mark.asyncio
    async def test_code_reads_documents_from_every_database(self, tmp_path):
        """The virtual filesystem is one flat namespace over the whole selected
        set, so code reads a document without knowing which database holds it."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            sandbox = Sandbox(
                db_path=None,
                config=rag._config,
                context=AnalysisContext(),
                rag=rag,
            )
            try:
                result = await sandbox.execute(
                    "docs = await list_documents()\n"
                    "for d in sorted(docs, key=lambda d: d['uri']):\n"
                    "    with open('/documents/' + d['id'] + '/content.txt') as f:\n"
                    "        print(f.read())"
                )
            finally:
                await sandbox.close()

        assert result.success, result.stderr
        assert "alpha document about cats" in result.stdout
        assert "beta document about cats" in result.stdout

    @pytest.mark.asyncio
    async def test_code_cannot_read_an_unselected_database(self, tmp_path):
        """Scoping the question scopes the filesystem."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha document about cats"])
        await _seed(config, "beta", ["beta document about cats"])

        async with HaikuRAG(config=config) as rag:
            beta = (await rag.clients_for(["beta"]))[0]
            [outside] = await beta.document_repository.list_all(limit=1)
            sandbox = Sandbox(
                db_path=None,
                config=rag._config,
                context=AnalysisContext(sources=["alpha"]),
                rag=rag,
            )
            try:
                result = await sandbox.execute(
                    "docs = await list_documents()\n"
                    "print(len(docs))\n"
                    f"print(open('/documents/{outside.id}/content.txt').read())"
                )
            finally:
                await sandbox.close()

        assert not result.success
        assert "beta document" not in result.stdout


class TestSelectionOnOneDatabase:
    """A client covering a single named database answers a selection the same way
    a search does, or the sandbox would mount what a search would refuse."""

    @pytest.mark.asyncio
    async def test_selecting_no_database_mounts_nothing(self, tmp_path):
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            _, docs, owners = await _mounted(rag, sources=[])

        assert docs == []
        assert owners == {}

    @pytest.mark.asyncio
    async def test_selecting_another_database_is_refused(self, tmp_path):
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(KeyError, match="beta"):
                await _mounted(rag, sources=["beta"])

    @pytest.mark.asyncio
    async def test_selecting_it_by_name_mounts_it(self, tmp_path):
        config = _config(tmp_path, ["alpha"])
        await _seed(config, "alpha", ["alpha document about cats"])

        async with HaikuRAG(config=config) as rag:
            _, docs, _ = await _mounted(rag, sources=["alpha"])

        assert len(docs) == 1


class TestCopiedDatabases:
    @pytest.mark.asyncio
    async def test_a_document_in_two_databases_is_refused(self, tmp_path):
        """Ids are unique per database, not across a copy of one: two documents
        would claim one path and the last would answer for both."""
        config = _config(tmp_path, ["alpha", "clone"])
        await _seed(config, "alpha", ["alpha document about cats"])
        shutil.rmtree(tmp_path / "clone.lancedb", ignore_errors=True)
        shutil.copytree(tmp_path / "alpha.lancedb", tmp_path / "clone.lancedb")

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(ValueError, match="one document per id"):
                await _mounted(rag)

    @pytest.mark.asyncio
    async def test_the_refusal_names_the_databases(self, tmp_path):
        config = _config(tmp_path, ["alpha", "clone"])
        await _seed(config, "alpha", ["alpha document about cats"])
        shutil.rmtree(tmp_path / "clone.lancedb", ignore_errors=True)
        shutil.copytree(tmp_path / "alpha.lancedb", tmp_path / "clone.lancedb")

        async with HaikuRAG(config=config) as rag:
            with pytest.raises(ValueError) as raised:
                await _mounted(rag)

        assert "alpha" in str(raised.value)
        assert "clone" in str(raised.value)
        assert str(tmp_path) not in str(raised.value)


class TestListingOrder:
    @pytest.mark.asyncio
    async def test_the_listing_interleaves_the_databases(self, tmp_path):
        """Code reads the listing through a truncated output, so concatenating
        shows one database's documents until the truncation and hides the rest."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", [f"alpha {i}" for i in range(5)])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            sandbox = Sandbox(
                db_path=None,
                config=config,
                context=AnalysisContext(),
                rag=rag,
            )
            docs, _ = await sandbox._documents()

        assert len(docs) == 6
        # The head has to reveal both databases.
        assert {(d.uri or "").split("/")[2] for d in docs[:2]} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_in_code_list_documents_names_the_database(self, tmp_path):
        """`source` is what lets code group the corpus by database."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config) as rag:
            sandbox = Sandbox(
                db_path=None,
                config=config,
                context=AnalysisContext(),
                rag=rag,
            )
            rows = await sandbox._build_external_functions()["list_documents"]()

        assert "source" in rows[0]
        assert {r["source"] for r in rows} == {"alpha", "beta"}

    @pytest.mark.asyncio
    async def test_in_code_list_documents_names_one_database_too(self, tmp_path):
        """A document knows which database it came from whether or not the
        analysis spans several."""
        config = _config(tmp_path, ["alpha", "beta"])
        await _seed(config, "alpha", ["alpha one"])
        await _seed(config, "beta", ["beta one"])

        async with HaikuRAG(config=config, sources=["alpha"]) as rag:
            sandbox = Sandbox(
                db_path=None,
                config=config,
                context=AnalysisContext(),
                rag=rag,
            )
            rows = await sandbox._build_external_functions()["list_documents"]()

        assert [r["source"] for r in rows] == ["alpha"]

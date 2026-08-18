import pytest

from haiku.rag.capabilities.rag import create_capability
from haiku.rag.client import HaikuRAG


@pytest.mark.asyncio
async def test_a_borrowed_client_is_reused_not_reopened(temp_db_path, monkeypatch):
    """A capability handed a client must not open a second connection to the
    same database."""
    from haiku.rag.store.engine import Store

    async with HaikuRAG(temp_db_path, create=True) as client:
        opens = 0
        initialize = Store._initialize

        async def counted(self):
            nonlocal opens
            opens += 1
            return await initialize(self)

        monkeypatch.setattr(Store, "_initialize", counted)

        capability = create_capability(
            db_path=client.store.db_path, config=client._config, rag=client
        )

        assert await capability._ensure_rag() is client
        assert opens == 0


@pytest.mark.asyncio
async def test_closing_never_closes_a_borrowed_client(temp_db_path):
    """`_close` owns only what it opened. Closing the caller's client would be a
    use-after-close for the caller."""
    async with HaikuRAG(temp_db_path, create=True) as client:
        capability = create_capability(
            db_path=client.store.db_path, config=client._config, rag=client
        )
        await capability._ensure_rag()

        await capability._close()

        # Still usable by its owner.
        assert await client.list_documents() == []


@pytest.mark.asyncio
async def test_a_borrowed_client_survives_for_run(temp_db_path):
    """for_run clears the owned connection per run; a borrowed one is the
    caller's and carries into the run copy."""
    from tests.capabilities.test_capabilities import Deps, make_context

    async with HaikuRAG(temp_db_path, create=True) as client:
        capability = create_capability(
            db_path=client.store.db_path, config=client._config, rag=client
        )

        run_capability = await capability.for_run(make_context(Deps()))

        assert run_capability is not capability
        assert run_capability.rag is None
        assert run_capability.borrowed_rag is client
        assert await run_capability._ensure_rag() is client


@pytest.mark.asyncio
async def test_ask_hands_its_client_to_the_capability(temp_db_path, monkeypatch):
    """`ask` built the capability from a db_path alone, so the capability opened
    its own connection to a database the client already had open."""
    from haiku.rag.capabilities import rag as rag_capability
    from haiku.rag.store.engine import Store

    real = rag_capability.create_capability
    built = {}

    def spy(**kwargs):
        built["capability"] = real(**kwargs)
        raise RuntimeError("stop before running the agent")

    async with HaikuRAG(temp_db_path, create=True) as client:
        monkeypatch.setattr(rag_capability, "create_capability", spy)

        with pytest.raises(RuntimeError, match="stop before running the agent"):
            await client.ask("anything")

        capability = built["capability"]
        opens = 0
        initialize = Store._initialize

        async def counted(self):
            nonlocal opens
            opens += 1
            return await initialize(self)

        monkeypatch.setattr(Store, "_initialize", counted)

        assert await capability._ensure_rag() is client
        assert opens == 0


@pytest.mark.asyncio
async def test_analyze_hands_its_client_to_the_capability(temp_db_path, monkeypatch):
    from haiku.rag.capabilities import analysis as analysis_capability
    from haiku.rag.store.engine import Store

    real = analysis_capability.create_capability
    built = {}

    def spy(**kwargs):
        built["capability"] = real(**kwargs)
        raise RuntimeError("stop before running the agent")

    async with HaikuRAG(temp_db_path, create=True) as client:
        monkeypatch.setattr(analysis_capability, "create_capability", spy)

        with pytest.raises(RuntimeError, match="stop before running the agent"):
            await client.analyze("anything")

        capability = built["capability"]
        opens = 0
        initialize = Store._initialize

        async def counted(self):
            nonlocal opens
            opens += 1
            return await initialize(self)

        monkeypatch.setattr(Store, "_initialize", counted)

        assert await capability._ensure_rag() is client
        assert opens == 0

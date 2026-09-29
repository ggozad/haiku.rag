from haiku.rag.config import AppConfig
from haiku.rag.config.models import CurateStoreConfig
from haiku.rag.curate.store.migrations import open_store
from haiku.rag.curate.store.models import DatabaseSweep
from haiku.rag.curate.store.repository import CurateRepository
from haiku.rag.curate.sweep import sweep


async def run_sweep(config: AppConfig) -> list[DatabaseSweep]:
    """Open the store, sweep every configured database once, close the store."""
    engine = await open_store(config.curate.store)
    try:
        return await sweep(config, CurateRepository(engine))
    finally:
        await engine.dispose()


async def ensure_store(store: CurateStoreConfig) -> None:
    """Create the store if missing and bring its schema up to date."""
    engine = await open_store(store)
    await engine.dispose()

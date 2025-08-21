from pathlib import Path

from haiku.rag.store.engine import Store
from haiku.rag.store.repositories.chunk import ChunkRepository
from haiku.rag.store.repositories.document import DocumentRepository
from haiku.rag.store.repositories.settings import SettingsRepository


def create_store(
    db_path: Path,
    skip_validation: bool = False,
) -> Store:
    """Create a Store instance."""
    return Store(db_path, skip_validation)


def create_repositories(store: Store):
    """Create repository instances for the store."""
    return {
        "chunk": ChunkRepository(store),
        "document": DocumentRepository(store),
        "settings": SettingsRepository(store),
    }

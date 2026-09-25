import json

from haiku.rag.store.engine import Store, recorded_settings
from haiku.rag.store.upgrades import Upgrade


async def _apply_record_embedder_only(store: Store) -> None:
    """Cut the settings row to the version and embedder identity."""
    current = await store._read_stored_settings()
    reduced = recorded_settings(current)
    if reduced != current:
        await store.settings_table.update(
            {"settings": json.dumps(reduced)}, where="id = 'settings'"
        )


upgrade_record_embedder_only = Upgrade(
    version="0.89.0",
    apply=_apply_record_embedder_only,
    description="Store only the version and embedder identity in settings",
)

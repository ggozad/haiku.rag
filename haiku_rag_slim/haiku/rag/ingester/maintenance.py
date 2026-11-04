from dataclasses import dataclass
from typing import TYPE_CHECKING

from haiku.rag.store.engine import ConnectionMode

if TYPE_CHECKING:
    from haiku.rag.client import HaikuRAG
    from haiku.rag.store.engine import Store


class MaintenanceError(Exception):
    """A maintenance pass did not complete."""


@dataclass(frozen=True)
class MaintenanceResult:
    vacuumed: bool
    index_built: bool


async def has_vector_index(store: "Store") -> bool:
    """Whether the chunks table has an index on `vector`."""
    indices = await store.chunks_table.list_indices()
    return any("vector" in index.columns for index in indices)


async def maintain(client: "HaikuRAG") -> MaintenanceResult:
    """Vacuum, which keeps every existing index current, then build the vector
    index if there is none. Raises `MaintenanceError` when the vacuum does not
    complete. Does nothing on LanceDB Cloud."""
    store = client.store
    if ConnectionMode.of(store.location) == ConnectionMode.CLOUD:
        return MaintenanceResult(vacuumed=False, index_built=False)
    if not await store.vacuum():
        raise MaintenanceError(f"vacuum of {client.source!r} did not complete")
    built = not await has_vector_index(store) and await client.create_vector_index()
    return MaintenanceResult(vacuumed=True, index_built=built)

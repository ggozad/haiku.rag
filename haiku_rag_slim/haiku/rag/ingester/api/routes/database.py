from fastapi import APIRouter, Depends, HTTPException, status

from haiku.rag.ingester.api.server import APIState, get_state
from haiku.rag.store.info import DatabaseInfo, gather_database_info

router = APIRouter(tags=["database"])


@router.get("/databases/{name:path}", response_model=DatabaseInfo)
async def database(name: str, state: APIState = Depends(get_state)) -> DatabaseInfo:
    """Read-only snapshot of an ingested database: stored version, embeddings,
    per-table counts/sizes, vector index status, pending migrations and
    package versions. The same data the `haiku-rag info` command prints.

    Opens a fresh read-only connection per call; not cached and not part of
    the dashboard's polling loop."""
    ref = state.databases.get(name)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="database not ingested"
        )
    return await gather_database_info(ref.location, state.config)

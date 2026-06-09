from fastapi import APIRouter, Depends, HTTPException, status

from haiku.rag.ingester.api.server import APIState, get_state
from haiku.rag.store.engine import DatabaseInfo, gather_database_info

router = APIRouter(tags=["database"])


@router.get("/database", response_model=DatabaseInfo)
async def database(state: APIState = Depends(get_state)) -> DatabaseInfo:
    """Read-only snapshot of the LanceDB target — stored version, embeddings,
    per-table counts/sizes, vector index status, pending migrations and
    package versions. The same data the `haiku-rag info` command prints.

    Opens a fresh read-only connection per call; not cached and not part of
    the dashboard's polling loop."""
    if state.db_path is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database path not configured",
        )
    return await gather_database_info(state.config, state.db_path)

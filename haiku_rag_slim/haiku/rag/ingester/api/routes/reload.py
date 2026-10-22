from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from haiku.rag.ingester.api.server import APIState, get_state
from haiku.rag.ingester.app import ReloadReport
from haiku.rag.ingester.exceptions import (
    InvalidConfigError,
    ReloadCancelledError,
    ReloadRefusedError,
)

router = APIRouter(tags=["reload"])


@router.post(
    "/reload",
    response_model=ReloadReport,
    responses={500: {"model": ReloadReport}},
)
async def reload(state: APIState = Depends(get_state)) -> ReloadReport | JSONResponse:
    """Read the configuration file again and apply its changes to
    `lancedb.databases` and `ingester.databases`. Returns once they are
    applied, with what was added, removed or failed: 200 when everything
    applied, 500 with the same report when a database failed."""
    if state.reload is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="the ingester is not running",
        )
    try:
        report = await state.reload()
    except ReloadRefusedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except InvalidConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        )
    except ReloadCancelledError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    if report.outcome != "applied":
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=report.model_dump(mode="json"),
        )
    return report

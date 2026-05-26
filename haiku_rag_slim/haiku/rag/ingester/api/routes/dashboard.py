from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["dashboard"])

_INDEX_HTML_PATH = Path(__file__).resolve().parent.parent / "static" / "index.html"


@router.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    """Serve the operator dashboard. Static page that polls /health, /sources,
    /stats and /jobs from the browser. Auth happens via the bearer header the
    JS attaches to its fetches — the dashboard route itself is unauthenticated
    so an operator can open it in a browser and paste the token on demand."""
    return FileResponse(_INDEX_HTML_PATH, media_type="text/html")

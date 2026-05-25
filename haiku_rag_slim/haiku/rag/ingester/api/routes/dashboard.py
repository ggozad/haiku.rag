from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_INDEX_HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the operator dashboard. Static page that polls /health, /sources,
    /stats and /jobs from the browser. Auth happens via the bearer header the
    JS attaches to its fetches — the dashboard route itself is unauthenticated
    so an operator can open it in a browser and paste the token on demand."""
    return HTMLResponse(content=_INDEX_HTML)

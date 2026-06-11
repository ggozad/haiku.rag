from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["dashboard"])

_INDEX_HTML_PATH = Path(__file__).resolve().parent.parent / "static" / "index.html"
_INDEX_HTML = _INDEX_HTML_PATH.read_text(encoding="utf-8")


@router.get("/", include_in_schema=False)
async def dashboard(request: Request) -> HTMLResponse:
    """Serve the operator dashboard. Static page that polls /health, /sources,
    /stats and /jobs from the browser. Auth happens via the bearer header the
    JS attaches to its fetches — the dashboard route itself is unauthenticated
    so an operator can open it in a browser and paste the token on demand.

    A ``<base href>`` matching the server's ``root_path`` is injected so the
    page's relative fetches resolve correctly whether the control plane is
    served at the root or reverse-proxied under a sub-path (e.g. ``/ingester/``).
    """
    root_path = request.scope.get("root_path", "")
    base_href = f"{root_path}/" if root_path else "/"
    html = _INDEX_HTML.replace("<head>", f'<head>\n    <base href="{base_href}" />', 1)
    return HTMLResponse(html)

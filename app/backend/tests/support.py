import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image
from starlette.applications import Starlette

from haiku.rag.config.models import AppConfig
from haiku_rag_app.main import create_app


def image_bytes(fmt: str, width: int = 8, height: int = 8) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format=fmt)
    return buffer.getvalue()


def png(width: int = 8, height: int = 8) -> str:
    """A base64 PNG, as the page sends an image."""
    return base64.b64encode(image_bytes("PNG", width, height)).decode()


@asynccontextmanager
async def served(
    config: AppConfig, sessions_path: Path
) -> AsyncIterator[tuple[Starlette, httpx.AsyncClient]]:
    """An app over `config` and a client of it, closed on exit."""
    app = create_app(config, sessions_path=sessions_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as http:
            yield app, http
    finally:
        await app.state.databases.aclose()
        await app.state.sessions.aclose()


async def settled(client: httpx.AsyncClient, source: str, path: str = "") -> dict:
    """The doctor run, or with `path="/projection"` the projection, once it ends."""
    for _ in range(600):
        body = (await client.get(f"/api/doctor/{source}{path}")).json()
        run = body.get("run") or body.get("projection")
        if run and run["status"] in ("done", "failed"):
            return run
        await asyncio.sleep(0.02)
    raise AssertionError("the job did not settle")

import asyncio
import base64
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import cache
from io import BytesIO
from pathlib import Path

import httpx
from docling_core.types.doc.base import BoundingBox, Size
from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image
from starlette.applications import Starlette

from haiku.rag.config.models import AppConfig
from haiku_rag_app.main import create_app


def image_bytes(fmt: str, width: int = 8, height: int = 8) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format=fmt)
    return buffer.getvalue()


@cache
def letter_pdf(pages: int) -> bytes:
    """A PDF of blank letter-size pages, the original of a seeded document.

    Cached: pdfium writes a new file ID on every save, and tests compare MD5s.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(612, 792)
    buffer = BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def png(width: int = 8, height: int = 8) -> str:
    """A base64 PNG, as the page sends an image."""
    return base64.b64encode(image_bytes("PNG", width, height)).decode()


@asynccontextmanager
async def served(
    config: AppConfig, sessions_path: Path, config_path: Path | None = None
) -> AsyncGenerator[tuple[Starlette, httpx.AsyncClient], None]:
    """An app over `config` and a client of it, closed on exit."""
    app = create_app(config, sessions_path=sessions_path, config_path=config_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as http:
            yield app, http
    finally:
        await app.state.configuration.aclose()
        await app.state.sessions.aclose()


async def settled(client: httpx.AsyncClient, source: str, path: str = "") -> dict:
    """The doctor run, or with `path="/similarity"` the similarity job, once it ends."""
    for _ in range(600):
        body = (await client.get(f"/api/doctor/{source}{path}")).json()
        run = body.get("run") or body.get("similarity")
        if run and run["status"] in ("done", "failed"):
            return run
        await asyncio.sleep(0.02)
    raise AssertionError("the job did not settle")


def paged_document(text: str) -> DoclingDocument:
    """A one-page document whose only item has a bounding box to draw."""
    doc = DoclingDocument(name=text)
    doc.add_page(page_no=1, size=Size(width=612.0, height=792.0))
    doc.add_text(
        label=DocItemLabel.PARAGRAPH,
        text=text,
        prov=ProvenanceItem(
            page_no=1,
            bbox=BoundingBox(l=50, t=700, r=550, b=650),
            charspan=(0, len(text)),
        ),
    )
    return doc

import httpx
from httpx import ASGITransport

from haiku.rag.curate.api.server import APIState, build_app
from tests.curate.test_sweep import curate  # noqa: F401


async def _get(curate, root_path: str = "") -> httpx.Response:  # noqa: F811
    _, config, repository = curate
    app = build_app(
        APIState(config, repository), auth_token="secret", root_path=root_path
    )
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app, root_path=root_path),
        base_url="http://testserver",
    ) as client:
        return await client.get(f"{root_path}/")


async def test_dashboard_is_served_without_a_token(curate):  # noqa: F811
    response = await _get(curate)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>haiku-curate</title>" in response.text
    assert '<base href="/" />' in response.text


async def test_dashboard_base_href_follows_the_root_path(curate):  # noqa: F811
    response = await _get(curate, root_path="/curate")
    assert response.status_code == 200
    assert '<base href="/curate/" />' in response.text

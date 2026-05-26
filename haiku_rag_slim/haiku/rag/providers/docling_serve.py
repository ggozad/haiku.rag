"""Shared client for docling-serve async API."""

import asyncio
import itertools
from typing import Any

import httpx

# Process-global registry of round-robin rotators over docling-serve
# instances, keyed by the sorted tuple of base URLs. Clients are constructed
# per-job by `get_converter` / `get_chunker`, so the rotation cursor has to
# live OUTSIDE the instance to actually advance across jobs. Two clients
# pointing at the same instance set share one rotator; clients with
# different sets get independent rotators.
_instance_rotators: dict[tuple[str, ...], "itertools.cycle[str]"] = {}


class DoclingServeClient:
    """Client for docling-serve async workflow.

    Handles the submit → poll → fetch pattern used by both conversion and
    chunking. Accepts a list of base URLs and round-robins jobs across
    them via a process-wide rotator keyed by the URL set. Each job's
    submit/poll/result trio stays on the same instance — task IDs are
    instance-local, so picking a different URL mid-job would 404.
    """

    def __init__(
        self,
        base_urls: str | list[str],
        api_key: str | None = None,
        timeout: float = 300,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        urls = [base_urls] if isinstance(base_urls, str) else list(base_urls)
        if not urls:
            raise ValueError("DoclingServeClient requires at least one base_url")
        self.base_urls: list[str] = [u.rstrip("/") for u in urls]
        self.api_key = api_key
        self.timeout = timeout
        # transport is for testing — production callers leave it None.
        self._transport = transport
        # setdefault is atomic under the GIL — concurrent constructors with
        # the same instance set will end up sharing one rotator.
        key = tuple(sorted(self.base_urls))
        self._instance_rotator = _instance_rotators.setdefault(
            key, itertools.cycle(self.base_urls)
        )

    def _httpx_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    @property
    def base_url(self) -> str:
        """First-URL view, mostly for log messages. Don't use for dispatch —
        callers should let `_pick_url` round-robin per request."""
        return self.base_urls[0]

    def _pick_url(self) -> str:
        return next(self._instance_rotator)

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    async def _submit_and_wait(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        endpoint: str,
        files: dict[str, Any],
        data: dict[str, Any],
        headers: dict[str, str],
        name: str,
    ) -> str:
        """Submit a task and poll until success. Returns the task_id.

        Shared by submit_and_poll (JSON results) and submit_and_poll_zip
        (binary zip results) — only the result-fetching step differs.
        """
        submit_url = f"{base_url}{endpoint}"
        response = await client.post(
            submit_url,
            files=files,
            data=data,
            headers=headers,
        )
        response.raise_for_status()
        submit_result = response.json()
        task_id = submit_result.get("task_id")

        if not task_id:
            raise ValueError("docling-serve did not return a task_id")

        poll_url = f"{base_url}/v1/status/poll/{task_id}"
        while True:
            poll_response = await client.get(poll_url, headers=headers)
            poll_response.raise_for_status()
            poll_result = poll_response.json()
            status = poll_result.get("task_status")

            if status == "success":
                return task_id
            elif status in ("failure", "error"):
                raise ValueError(f"docling-serve task failed for {name}: {poll_result}")

            await asyncio.sleep(1)

    async def submit_and_poll(
        self,
        endpoint: str,
        files: dict[str, Any],
        data: dict[str, Any],
        name: str = "document",
    ) -> dict[str, Any]:
        """Submit a task and poll until completion; fetch result as JSON.

        httpx exceptions (ConnectError, HTTPStatusError, TimeoutException,
        etc.) propagate so the ingester's pipeline classifier can route
        4xx → PermanentError and 5xx/network → TransientError. ValueError
        is raised by `_submit_and_wait` when docling-serve reports a task
        failure or returns no task_id.
        """
        headers = self._get_headers()
        base_url = self._pick_url()
        async with self._httpx_client() as client:
            task_id = await self._submit_and_wait(
                client, base_url, endpoint, files, data, headers, name
            )
            result_url = f"{base_url}/v1/result/{task_id}"
            result_response = await client.get(result_url, headers=headers)
            result_response.raise_for_status()
            return result_response.json()

    async def submit_and_poll_zip(
        self,
        endpoint: str,
        files: dict[str, Any],
        data: dict[str, Any],
        name: str = "document",
    ) -> bytes:
        """Submit a task and poll until completion; fetch result as raw bytes.

        Used when the caller requested ``target_type=zip`` (e.g. to retrieve
        picture image bytes that docling-serve only emits as referenced files
        bundled into a zip archive). The submit/poll flow is identical to
        ``submit_and_poll``; only the result-fetching step differs.
        """
        headers = self._get_headers()
        base_url = self._pick_url()
        async with self._httpx_client() as client:
            task_id = await self._submit_and_wait(
                client, base_url, endpoint, files, data, headers, name
            )
            result_url = f"{base_url}/v1/result/{task_id}"
            result_response = await client.get(result_url, headers=headers)
            result_response.raise_for_status()
            return result_response.content

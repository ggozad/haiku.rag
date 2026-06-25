"""Shared client for docling-serve async API."""

import asyncio
import itertools
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Statuses worth retrying on another instance: 408/429 are transient overload,
# 5xx covers a crashed or restarting docling-serve worker (the OOM-leak failure
# mode). Other 4xx are the caller's fault and won't succeed elsewhere.
_RETRYABLE_STATUS = frozenset({408, 429})


def _is_retryable(exc: BaseException) -> bool:
    """Whether a failed docling-serve request should be retried on another
    instance. True for transport-level failures (connection reset / timeout —
    an instance that crashed or went unresponsive) and server-side 5xx/overload;
    False for other 4xx and task-level ``ValueError``\\ s, which won't succeed on
    a retry (a deterministically bad document is handled upstream, not here)."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in _RETRYABLE_STATUS or status >= 500
    return isinstance(exc, httpx.TransportError)


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
        max_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
    ):
        urls = [base_urls] if isinstance(base_urls, str) else list(base_urls)
        if not urls:
            raise ValueError("DoclingServeClient requires at least one base_url")
        self.base_urls: list[str] = [u.rstrip("/") for u in urls]
        self.api_key = api_key
        self.timeout = timeout
        # transport is for testing — production callers leave it None.
        self._transport = transport
        # Bounded retry with failover: docling-serve instances crash (memory
        # leaks), so a request that hits a dying instance is retried, preferring
        # an instance that hasn't already failed this request.
        self._max_attempts = max(1, max_attempts)
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
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

    def _pick_url(self, exclude: frozenset[str] = frozenset()) -> str:
        """Next instance in the round-robin, preferring one not in ``exclude``
        (instances that already failed this request). Falls back to a
        possibly-excluded instance when every instance is excluded — a
        single-instance fleet, or one where all instances failed, still gets
        retried after backoff in case the instance has since restarted."""
        url = next(self._instance_rotator)
        if url not in exclude:
            return url
        for _ in range(len(self.base_urls) - 1):
            url = next(self._instance_rotator)
            if url not in exclude:
                return url
        return url

    def _retry_delay(self, attempt_no: int) -> float:
        """Capped exponential backoff between retry attempts."""
        return min(self._retry_base_delay * (2**attempt_no), self._retry_max_delay)

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    async def _run_with_retry(
        self,
        attempt: Callable[[httpx.AsyncClient, str], Awaitable[_T]],
        name: str,
    ) -> _T:
        """Run ``attempt(client, base_url)`` with bounded retry + failover.

        Each attempt runs the full submit → poll → fetch trio against one
        instance (task IDs are instance-local, so a trio can't be split across
        instances). On a retryable failure — a transport error (crashed /
        unresponsive instance) or 5xx/overload — the next attempt prefers an
        instance that hasn't already failed this request. Non-retryable errors
        (4xx, task-level ``ValueError``) propagate immediately.
        """
        tried: set[str] = set()
        last_exc: BaseException | None = None
        for attempt_no in range(self._max_attempts):
            base_url = self._pick_url(exclude=frozenset(tried))
            try:
                async with self._httpx_client() as client:
                    return await attempt(client, base_url)
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                last_exc = exc
                tried.add(base_url)
                remaining = self._max_attempts - attempt_no - 1
                if remaining <= 0:
                    raise
                logger.warning(
                    "docling-serve request for %s failed on %s (%s); retrying "
                    "on another instance (%d attempt(s) left)",
                    name,
                    base_url,
                    exc,
                    remaining,
                )
                await asyncio.sleep(self._retry_delay(attempt_no))
        # Unreachable: max_attempts >= 1, and the final iteration either returns
        # on success or re-raises on failure (remaining <= 0).
        raise last_exc or RuntimeError(  # pragma: no cover
            "retry loop exited without a result"
        )

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

        Retries on a different instance when an instance crashes or returns
        5xx/overload (see ``_run_with_retry``). Non-retryable httpx exceptions
        (4xx other than 408/429) and the ValueError raised by
        ``_submit_and_wait`` (task failure / missing task_id) propagate so the
        ingester's pipeline classifier can route them (4xx → PermanentError,
        ValueError → TransientError for a whole-document retry).
        """
        headers = self._get_headers()

        async def _attempt(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
            task_id = await self._submit_and_wait(
                client, base_url, endpoint, files, data, headers, name
            )
            result_url = f"{base_url}/v1/result/{task_id}"
            result_response = await client.get(result_url, headers=headers)
            result_response.raise_for_status()
            return result_response.json()

        return await self._run_with_retry(_attempt, name)

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
        ``submit_and_poll`` (including the retry + failover); only the
        result-fetching step differs.
        """
        headers = self._get_headers()

        async def _attempt(client: httpx.AsyncClient, base_url: str) -> bytes:
            task_id = await self._submit_and_wait(
                client, base_url, endpoint, files, data, headers, name
            )
            result_url = f"{base_url}/v1/result/{task_id}"
            result_response = await client.get(result_url, headers=headers)
            result_response.raise_for_status()
            return result_response.content

        return await self._run_with_retry(_attempt, name)

"""Shared client for docling-serve async API."""

import asyncio
import itertools
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

from haiku.rag.circuit_breaker import CircuitBreaker
from haiku.rag.config import CircuitBreakerConfig, DoclingServeConfig
from haiku.rag.telemetry import logfire

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Statuses that mean "try again elsewhere" rather than "this request is bad".
_RETRYABLE_STATUS = frozenset({408, 429})


def _is_retryable(exc: BaseException) -> bool:
    """Whether a failure reflects an unhealthy docling-serve instance.

    True for transport errors (a crashed / unresponsive instance) and
    server-side 5xx / 408 / 429. Such failures are retried on another instance
    and counted against the instance's circuit breaker. Other 4xx and the
    task-level ValueError from `_submit_and_wait` (a bad document, not the
    instance's fault) won't succeed on a retry, so they propagate untouched.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status in _RETRYABLE_STATUS
    return isinstance(exc, httpx.TransportError)


# Process-global registries over docling-serve instances. Clients are built
# per-job by `get_converter` / `get_chunker`, so this shared state lives OUTSIDE
# the instance to persist across jobs: the round-robin cursor (keyed by the
# sorted URL set) advances across jobs, and the per-URL breakers keep a crashed
# instance skipped across jobs until its cooldown elapses. The first client to
# touch a URL fixes that breaker's config; in practice all clients share one.
_instance_rotators: dict[tuple[str, ...], "itertools.cycle[str]"] = {}
_instance_breakers: dict[str, CircuitBreaker] = {}


class DoclingServeClient:
    """Client for docling-serve async workflow.

    Handles the submit → poll → fetch pattern used by both conversion and
    chunking. Accepts a list of base URLs and round-robins jobs across them via
    a process-wide rotator keyed by the URL set. Each job's submit/poll/result
    trio stays on the same instance — task IDs are instance-local, so picking a
    different URL mid-job would 404.

    When an instance crashes or returns 5xx, the request fails over to another
    instance (up to `max_attempts`) and the failure trips that instance's
    circuit breaker, so subsequent `_pick_url` calls skip it while it recovers.
    """

    def __init__(
        self,
        base_urls: str | list[str],
        api_key: str | None = None,
        timeout: float = 300,
        transport: httpx.AsyncBaseTransport | None = None,
        circuit_breaker: CircuitBreakerConfig | None = None,
        max_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
        now_fn: Callable[[], float] = time.monotonic,
    ):
        urls = [base_urls] if isinstance(base_urls, str) else list(base_urls)
        if not urls:
            raise ValueError("DoclingServeClient requires at least one base_url")
        self.base_urls: list[str] = [u.rstrip("/") for u in urls]
        self.api_key = api_key
        self.timeout = timeout
        # transport is for testing — production callers leave it None.
        self._transport = transport
        self._breaker_config = circuit_breaker or CircuitBreakerConfig()
        self._max_attempts = max(1, max_attempts)
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._now = now_fn
        # setdefault is atomic under the GIL — concurrent constructors with
        # the same instance set will end up sharing one rotator.
        key = tuple(sorted(self.base_urls))
        self._instance_rotator = _instance_rotators.setdefault(
            key, itertools.cycle(self.base_urls)
        )

    @classmethod
    def from_config(cls, config: DoclingServeConfig) -> "DoclingServeClient":
        return cls(
            base_urls=config.base_urls,
            api_key=config.api_key,
            timeout=config.timeout,
            circuit_breaker=config.circuit_breaker,
            max_attempts=config.max_attempts,
        )

    def _httpx_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    @property
    def base_url(self) -> str:
        """First-URL view, mostly for log messages. Don't use for dispatch —
        callers should let `_pick_url` round-robin per request."""
        return self.base_urls[0]

    def _breaker_for(self, url: str) -> CircuitBreaker:
        breaker = _instance_breakers.get(url)
        if breaker is None:
            breaker = CircuitBreaker(self._breaker_config, now_fn=self._now)
            _instance_breakers[url] = breaker
        return breaker

    def _pick_url(self, exclude: frozenset[str] = frozenset()) -> str:
        """Next instance in the round-robin, skipping instances that already
        failed this request (`exclude`) or whose circuit breaker is open.

        When no healthy, not-excluded instance exists, probe one anyway (the
        loop consumed a full rotation, so one more step rotates across calls
        rather than pinning one node) — a single-instance or fully-down fleet
        still gets an attempt instead of failing with nothing to pick.
        """
        for _ in range(len(self.base_urls)):
            url = next(self._instance_rotator)
            if url in exclude:
                continue
            if not self._breaker_for(url).is_open:
                return url
        return next(self._instance_rotator)

    def _retry_delay(self, attempt_no: int) -> float:
        """Capped exponential backoff between retry attempts."""
        return min(self._retry_base_delay * (2**attempt_no), self._retry_max_delay)

    async def _run_with_retry(
        self,
        attempt: Callable[[httpx.AsyncClient, str], Awaitable[_T]],
        name: str,
    ) -> _T:
        """Run `attempt(client, base_url)` with bounded retry + failover.

        Each attempt runs the full submit → poll → fetch trio against one
        instance (task IDs are instance-local, so a trio can't be split across
        instances). A retryable failure counts against that instance's breaker
        and fails over to another instance; a successful attempt closes the
        breaker. Non-retryable errors propagate immediately.
        """
        tried: set[str] = set()
        last_exc: BaseException | None = None
        for attempt_no in range(self._max_attempts):
            base_url = self._pick_url(exclude=frozenset(tried))
            breaker = self._breaker_for(base_url)
            try:
                with logfire.span(
                    "docling_serve.request",
                    name=name,
                    url=base_url,
                    attempt=attempt_no,
                ):
                    async with self._httpx_client() as client:
                        result = await attempt(client, base_url)
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                breaker.record_failure()
                tried.add(base_url)
                last_exc = exc
                if attempt_no + 1 >= self._max_attempts:
                    raise
                logger.warning(
                    "docling-serve request for %s failed on %s "
                    "(attempt %d/%d), retrying on another instance: %s",
                    name,
                    base_url,
                    attempt_no + 1,
                    self._max_attempts,
                    exc,
                )
                await asyncio.sleep(self._retry_delay(attempt_no))
            else:
                breaker.record_success()
                return result
        raise last_exc or RuntimeError(  # pragma: no cover
            "retry loop exited without a result"
        )

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

        Retries on a different instance when an instance crashes or returns
        5xx/overload (see `_run_with_retry`). Non-retryable httpx exceptions
        (4xx other than 408/429) and the ValueError raised by `_submit_and_wait`
        (task failure / missing task_id) propagate so the ingester's pipeline
        classifier can route them (4xx → PermanentError, ValueError →
        TransientError for a whole-document retry).
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

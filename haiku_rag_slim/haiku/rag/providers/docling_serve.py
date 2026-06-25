"""Shared client for docling-serve async API."""

import asyncio
import itertools
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# HTTP statuses that count against an instance's health: 408/429 are transient
# overload, 5xx is a crashed/restarting worker (the OOM-leak failure mode).
_UNHEALTHY_STATUS = frozenset({408, 429})


def _is_instance_failure(exc: BaseException) -> bool:
    """Whether a failure reflects an unhealthy docling-serve instance and should
    count against its circuit breaker: transport errors (connection reset /
    timeout — a crashed or unresponsive instance) and 5xx/overload. Other 4xx
    are the caller's fault and the task-level ``ValueError`` from
    ``_submit_and_wait`` is a document problem — neither reflects instance
    health, so they don't trip the breaker."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in _UNHEALTHY_STATUS or status >= 500
    return isinstance(exc, httpx.TransportError)


class _InstanceBreaker:
    """Per-instance circuit breaker (closed / open with cooldown).

    Self-contained so the provider layer doesn't depend on the ingester
    package. ``is_open`` auto-probes after the cooldown elapses: a single
    request is allowed through, and its success closes the breaker while another
    failure re-opens it.
    """

    def __init__(
        self,
        failure_threshold: int,
        cooldown_s: float,
        now_fn: Callable[[], float],
    ):
        self._threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._now = now_fn
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        # Cooldown elapsed → let the next request probe the instance.
        return self._now() - self._opened_at < self._cooldown_s

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._threshold:
            self._opened_at = self._now()


# Process-global registry of round-robin rotators over docling-serve
# instances, keyed by the sorted tuple of base URLs. Clients are constructed
# per-job by `get_converter` / `get_chunker`, so the rotation cursor has to
# live OUTSIDE the instance to actually advance across jobs. Two clients
# pointing at the same instance set share one rotator; clients with
# different sets get independent rotators.
_instance_rotators: dict[tuple[str, ...], "itertools.cycle[str]"] = {}

# Process-global per-instance breakers, keyed by base URL. Like the rotators,
# these must outlive the per-job client so an instance that crashed stays
# skipped across subsequent jobs until its cooldown elapses. The first client
# to touch a URL fixes that breaker's threshold/cooldown/clock; in practice all
# clients share one config (the docling-serve provider config).
_instance_breakers: dict[str, "_InstanceBreaker"] = {}


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
        breaker_failure_threshold: int = 3,
        breaker_cooldown_s: float = 30.0,
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
        self._breaker_failure_threshold = breaker_failure_threshold
        self._breaker_cooldown_s = breaker_cooldown_s
        self._now = now_fn
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

    def _breaker_for(self, url: str) -> _InstanceBreaker:
        breaker = _instance_breakers.get(url)
        if breaker is None:
            breaker = _InstanceBreaker(
                self._breaker_failure_threshold,
                self._breaker_cooldown_s,
                self._now,
            )
            _instance_breakers[url] = breaker
        return breaker

    def _pick_url(self) -> str:
        """Next instance in the round-robin, skipping any whose circuit breaker
        is open (an instance that recently crashed / overloaded). Falls back to
        the first instance seen when every breaker is open — a single-instance
        fleet, or a fully-down one, still gets a probe rather than no attempt."""
        fallback: str | None = None
        for _ in range(len(self.base_urls)):
            url = next(self._instance_rotator)
            if fallback is None:
                fallback = url
            if not self._breaker_for(url).is_open:
                return url
        assert fallback is not None  # base_urls is non-empty (checked in __init__)
        return fallback

    def _record_outcome(self, base_url: str, exc: BaseException) -> None:
        """Record a failed request against the instance's breaker, but only when
        the failure reflects instance health (transport / 5xx). 4xx and
        task-level errors are left alone — they aren't the instance's fault."""
        if not _is_instance_failure(exc):
            return
        breaker = self._breaker_for(base_url)
        was_open = breaker.is_open
        breaker.record_failure()
        if not was_open and breaker.is_open:
            logger.warning(
                "docling-serve instance %s breaker opened after %d consecutive "
                "failure(s); skipping it for %.0fs",
                base_url,
                self._breaker_failure_threshold,
                self._breaker_cooldown_s,
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

        httpx exceptions (ConnectError, HTTPStatusError, TimeoutException,
        etc.) propagate so the ingester's pipeline classifier can route
        4xx → PermanentError and 5xx/network → TransientError. ValueError
        is raised by `_submit_and_wait` when docling-serve reports a task
        failure or returns no task_id. Instance-health failures (transport /
        5xx) trip the picked instance's circuit breaker so later requests skip
        it while it recovers.
        """
        headers = self._get_headers()
        base_url = self._pick_url()
        try:
            async with self._httpx_client() as client:
                task_id = await self._submit_and_wait(
                    client, base_url, endpoint, files, data, headers, name
                )
                result_url = f"{base_url}/v1/result/{task_id}"
                result_response = await client.get(result_url, headers=headers)
                result_response.raise_for_status()
                result = result_response.json()
        except Exception as exc:
            self._record_outcome(base_url, exc)
            raise
        self._breaker_for(base_url).record_success()
        return result

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
        ``submit_and_poll`` (including circuit-breaker bookkeeping); only the
        result-fetching step differs.
        """
        headers = self._get_headers()
        base_url = self._pick_url()
        try:
            async with self._httpx_client() as client:
                task_id = await self._submit_and_wait(
                    client, base_url, endpoint, files, data, headers, name
                )
                result_url = f"{base_url}/v1/result/{task_id}"
                result_response = await client.get(result_url, headers=headers)
                result_response.raise_for_status()
                result = result_response.content
        except Exception as exc:
            self._record_outcome(base_url, exc)
            raise
        self._breaker_for(base_url).record_success()
        return result

"""Shared client for docling-serve async API."""

import asyncio
import itertools
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Statuses signalling an unhealthy instance: 408/429 are transient overload,
# 5xx covers a crashed or restarting docling-serve worker (the OOM-leak failure
# mode). Other 4xx are the caller's fault and won't succeed elsewhere.
_RETRYABLE_STATUS = frozenset({408, 429})


def _is_retryable(exc: BaseException) -> bool:
    """Whether a failure reflects an unhealthy docling-serve instance.

    The single predicate behind both behaviours: such a failure is retried on
    another instance *and* counted against the instance's circuit breaker. True
    for transport-level failures (connection reset / timeout — an instance that
    crashed or went unresponsive) and server-side 5xx/overload; False for other
    4xx and task-level ``ValueError``\\ s, which won't succeed on a retry and
    aren't the instance's fault (a deterministically bad document is handled
    upstream, not here)."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in _RETRYABLE_STATUS or status >= 500
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


# Process-global registries over docling-serve instances. Clients are
# constructed per-job by `get_converter` / `get_chunker`, so this shared state
# has to live OUTSIDE the instance to persist across jobs: the round-robin
# cursor (keyed by the sorted tuple of base URLs) advances across jobs, and the
# per-URL breakers keep a crashed instance skipped across subsequent jobs until
# its cooldown elapses. The first client to touch a URL fixes that breaker's
# threshold/cooldown/clock; in practice all clients share one config.
_instance_rotators: dict[tuple[str, ...], "itertools.cycle[str]"] = {}
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
        max_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
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
        # Bounded retry with failover + per-instance circuit breaking:
        # docling-serve instances crash (memory leaks), so a request that hits a
        # dying instance is retried on another instance, and repeated failures
        # trip that instance's breaker so later requests skip it while it
        # recovers.
        self._max_attempts = max(1, max_attempts)
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
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

    def _pick_url(self, exclude: frozenset[str] = frozenset()) -> str:
        """Next instance in the round-robin, skipping instances that already
        failed this request (``exclude``) or whose circuit breaker is open
        (recently crashed / overloaded).

        Prefers a not-excluded, breaker-closed instance. Falls back to a
        not-excluded breaker-open one (worth a probe over re-hitting one that
        already failed this request), and finally to any instance — so a
        single-instance fleet, or one where everything is excluded/open, still
        gets an attempt rather than nothing to pick."""
        not_excluded: str | None = None
        for _ in range(len(self.base_urls)):
            url = next(self._instance_rotator)
            if url in exclude:
                continue
            if not_excluded is None:
                not_excluded = url
            if not self._breaker_for(url).is_open:
                return url
        if not_excluded is not None:
            return not_excluded
        return next(self._instance_rotator)

    def _retry_delay(self, attempt_no: int) -> float:
        """Capped exponential backoff between retry attempts."""
        return min(self._retry_base_delay * (2**attempt_no), self._retry_max_delay)

    def _record_failure(self, base_url: str) -> None:
        """Count an instance-health failure against ``base_url``'s breaker,
        logging when the breaker transitions to open."""
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
        instance that hasn't already failed this request, and the failure is
        counted against that instance's circuit breaker so later requests skip
        it while it recovers; a successful attempt closes the breaker.
        Non-retryable errors (4xx, task-level ``ValueError``) propagate
        immediately and leave the breaker untouched.
        """
        tried: set[str] = set()
        last_exc: BaseException | None = None
        for attempt_no in range(self._max_attempts):
            base_url = self._pick_url(exclude=frozenset(tried))
            try:
                async with self._httpx_client() as client:
                    result = await attempt(client, base_url)
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                # Retryable == an unhealthy instance: count it against the
                # breaker so later requests route around it.
                self._record_failure(base_url)
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
            else:
                self._breaker_for(base_url).record_success()
                return result
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

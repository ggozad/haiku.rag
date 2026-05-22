import time
from collections.abc import Callable

from haiku.rag.config import CircuitBreakerConfig


class CircuitBreaker:
    """Three-state breaker over discover() failures.

    - closed: failures are counted; threshold flips to open.
    - open: probes are blocked until cooldown elapses, then a single probe
      is allowed; success closes the breaker, another failure re-opens it.

    `now_fn` is injectable so tests don't need monkeypatching of time.time.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        *,
        now_fn: Callable[[], float] = time.monotonic,
    ):
        self._config = config or CircuitBreakerConfig()
        self._now = now_fn
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if self._now() - self._opened_at >= self._config.cooldown_s:
            # cooldown elapsed; let the next call probe
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._config.failure_threshold:
            self._opened_at = self._now()

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

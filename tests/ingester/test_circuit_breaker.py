from haiku.rag.config import CircuitBreakerConfig
from haiku.rag.ingester.pollers.circuit_breaker import CircuitBreaker


class _Clock:
    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_starts_closed():
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
    assert breaker.is_open is False
    assert breaker.consecutive_failures == 0


def test_opens_after_threshold():
    clock = _Clock()
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=3, cooldown_s=60.0), now_fn=clock
    )
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is False  # threshold not reached
    breaker.record_failure()
    assert breaker.is_open is True


def test_success_resets_failure_count():
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, cooldown_s=60.0))
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    assert breaker.consecutive_failures == 0


def test_cooldown_allows_probe():
    clock = _Clock()
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=2, cooldown_s=10.0), now_fn=clock
    )
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_open is True

    clock.advance(5.0)
    assert breaker.is_open is True  # still cooling
    clock.advance(5.5)
    assert breaker.is_open is False  # cooldown elapsed → probe allowed


def test_probe_failure_reopens():
    clock = _Clock()
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=2, cooldown_s=10.0), now_fn=clock
    )
    breaker.record_failure()
    breaker.record_failure()
    clock.advance(15.0)
    assert breaker.is_open is False

    breaker.record_failure()
    # failure_threshold=2 already exceeded by accumulating — breaker re-opens
    assert breaker.is_open is True


def test_probe_success_closes():
    clock = _Clock()
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=2, cooldown_s=10.0), now_fn=clock
    )
    breaker.record_failure()
    breaker.record_failure()
    clock.advance(15.0)
    breaker.record_success()
    assert breaker.is_open is False
    assert breaker.consecutive_failures == 0

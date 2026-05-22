import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter. Per-source override is allowed so
    one slow/flaky source doesn't drag the rest of the queue."""

    max_attempts: int = 5
    base_delay_s: float = 2.0
    max_delay_s: float = 300.0
    jitter: float = 0.25  # ±25%


def compute_backoff(
    attempt: int,
    policy: RetryPolicy,
    *,
    rng: random.Random | None = None,
) -> float:
    """Delay before the next attempt. `attempt` is the 1-indexed count of
    attempts already made (so attempt=1 → first retry delay).

    `rng` is injectable for deterministic tests.
    """
    if attempt < 1:
        attempt = 1
    raw = min(policy.base_delay_s * (2 ** (attempt - 1)), policy.max_delay_s)
    r = rng if rng is not None else random
    j = 1.0 + (r.random() * 2 - 1) * policy.jitter
    return max(0.0, raw * j)

from haiku.rag.ingester.workers.retry import RetryPolicy, compute_backoff


def test_backoff_past_the_float_range_is_the_cap():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=30.0, jitter=0.0)
    assert compute_backoff(1025, policy) == 30.0


def test_a_small_base_still_reaches_the_cap():
    policy = RetryPolicy(base_delay_s=1e-20, max_delay_s=30.0, jitter=0.0)
    assert compute_backoff(80, policy) == 30.0

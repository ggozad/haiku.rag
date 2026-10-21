import asyncio

import pytest

from haiku_rag_app import jobs


def test_a_job_finishing_after_the_request_loop_closed_is_dropped():
    runner = jobs.Jobs()
    closed = asyncio.new_event_loop()
    closed.close()
    job = jobs.Job(kind="doctor", source="alpha")
    runner._post(closed, runner._finish, job, None, "late")
    assert job.status == "queued"


def test_a_scheduling_failure_on_a_live_loop_is_raised(monkeypatch):
    def refuse(*args):
        raise RuntimeError("refused")

    live = asyncio.new_event_loop()
    monkeypatch.setattr(live, "call_soon_threadsafe", refuse)
    try:
        with pytest.raises(RuntimeError, match="refused"):
            jobs.Jobs._post(live, print)
    finally:
        live.close()


async def test_the_worker_lets_go_of_a_delivered_result():
    import gc
    import weakref

    class Result:
        pass

    runner = jobs.Jobs()
    refs = []

    async def work(progress):
        result = Result()
        refs.append(weakref.ref(result))
        return result

    job = runner.submit("alpha", "doctor", work, on_done=lambda result: "delivered")
    for _ in range(500):
        if job.status == "done":
            break
        await asyncio.sleep(0.01)
    assert job.result == "delivered"
    gc.collect()
    assert refs[0]() is None

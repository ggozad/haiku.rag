import asyncio

import pytest

from haiku.rag.client.scope import DatabaseRef
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

    job = runner.submit(
        DatabaseRef.configured("alpha", "alpha.lancedb"),
        "doctor",
        work,
        on_done=lambda result: "delivered",
    )
    for _ in range(500):
        if job.status == "done":
            break
        await asyncio.sleep(0.01)
    assert job.result == "delivered"
    gc.collect()
    assert refs[0]() is None


ALPHA = DatabaseRef.configured("alpha", "alpha.lancedb")


async def settled(job: jobs.Job) -> None:
    for _ in range(500):
        if job.status in ("done", "failed"):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"job still {job.status}")


def finished_spans(exporter, name: str):
    return [
        span
        for span in exporter.exported_spans
        if span.name == name
        and (span.attributes or {}).get("logfire.span_type") != "pending_span"
    ]


def span_id(span) -> int:
    context = span.get_span_context()
    assert context is not None
    return context.span_id


def exception_types(span) -> list[str]:
    return [
        (event.attributes or {})["exception.type"]
        for event in span.events
        if event.name == "exception"
    ]


async def test_a_job_runs_in_a_span_naming_its_kind_and_database(exporter):
    async def work(progress):
        return "report"

    job = jobs.Jobs().submit(ALPHA, "doctor", work)
    await settled(job)

    [span] = finished_spans(exporter, "app.job")
    attributes = dict(span.attributes or {})
    assert attributes["kind"] == "doctor"
    assert attributes["database"] == "alpha"
    assert exception_types(span) == []


async def test_a_failing_job_records_its_exception_on_the_span(exporter):
    async def work(progress):
        raise ValueError("no vectors")

    job = jobs.Jobs().submit(ALPHA, "doctor", work)
    await settled(job)

    assert job.status == "failed"
    assert job.error == "no vectors"
    [span] = finished_spans(exporter, "app.job")
    assert exception_types(span) == ["ValueError"]
    assert (span.attributes or {})["logfire.level_num"] == 17


async def test_each_job_span_nests_under_the_span_that_submitted_it(exporter):
    import logfire

    async def work(progress):
        return None

    runner = jobs.Jobs()
    with logfire.span("first request") as first:
        doctor = runner.submit(ALPHA, "doctor", work)
    with logfire.span("second request") as second:
        projection = runner.submit(ALPHA, "projection", work)
    await settled(doctor)
    await settled(projection)

    parents = {
        (span.attributes or {})["kind"]: span.parent.span_id
        for span in finished_spans(exporter, "app.job")
        if span.parent is not None
    }
    assert parents == {
        "doctor": span_id(first),
        "projection": span_id(second),
    }


async def test_a_failing_on_done_fails_the_job_and_frees_its_slot(exporter):
    import logfire

    async def work(progress):
        return "projection"

    def done(result):
        raise KeyError("lost generation")

    runner = jobs.Jobs()
    with logfire.span("request") as request:
        job = runner.submit(ALPHA, "projection", work, on_done=done)
    await settled(job)

    assert job.status == "failed"
    assert job.error == "'lost generation'"
    assert job.finished_at is not None
    [record] = [
        span
        for span in exporter.exported_spans
        if exception_types(span) == ["KeyError"]
    ]
    assert record.parent is not None
    assert record.parent.span_id == span_id(request)

    again = runner.submit(ALPHA, "projection", work)
    await settled(again)
    assert again.status == "done"

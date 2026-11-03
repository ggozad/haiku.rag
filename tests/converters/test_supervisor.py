import asyncio
import io
import json
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from docling_core.types.doc.document import DoclingDocument

from haiku.rag.config import AppConfig
from haiku.rag.converters import supervisor as supervisor_module
from haiku.rag.converters import supervisor_worker
from haiku.rag.converters.exceptions import (
    ConversionFailedError,
    ConversionTimeoutError,
    SupervisedConversionTimeoutError,
)
from haiku.rag.converters.supervisor import (
    ConversionSupervisor,
    UnitFailure,
    close_supervisors,
    get_supervisor,
    running_supervisors,
    set_default_gate,
)
from haiku.rag.converters.supervisor_worker import read_frame, write_frame

STUB = (sys.executable, str(Path(__file__).with_name("supervisor_stub.py")))


def _config(**supervisor) -> AppConfig:
    config = AppConfig()
    config.processing.conversion_options.ocr_engine = "ocrmac"
    settings = {
        "enabled": True,
        "workers": 2,
        "unit_timeout": 3.0,
        "retry_ocr_engine": "rapidocr",
        "background": False,
        **supervisor,
    }
    for key, value in settings.items():
        setattr(config.processing.supervisor, key, value)
    return config


def _files(tmp_path: Path, *names: str) -> list[Path]:
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths.append(path)
    return paths


def _pdf(path: Path, pages: int) -> Path:
    pdf = pdfium.PdfDocument.new()
    try:
        for _ in range(pages):
            pdf.new_page(100, 200)
        with open(path, "wb") as f:
            pdf.save(f)
    finally:
        pdf.close()
    return path


def _texts(document: DoclingDocument) -> list[str]:
    return [item.text for item in document.texts]


@pytest.mark.slow
async def test_a_hanging_document_costs_only_itself(tmp_path):
    supervisor = ConversionSupervisor(_config(), command=STUB)
    ok, hangs, recovers, also_ok = _files(
        tmp_path, "a.md", "hang-always.md", "hang-once.md", "b.md"
    )
    started = time.monotonic()
    try:
        results = await asyncio.gather(
            *(supervisor.convert_file(path) for path in (ok, hangs, recovers, also_ok)),
            return_exceptions=True,
        )
        elapsed = time.monotonic() - started

        first, failure, recovered, last = results
        assert isinstance(first, DoclingDocument)
        assert isinstance(recovered, DoclingDocument)
        assert isinstance(last, DoclingDocument)
        assert _texts(first) == ["a.md by ocrmac"]
        assert _texts(recovered) == ["hang-once.md by rapidocr"]
        assert _texts(last) == ["b.md by ocrmac"]
        assert isinstance(failure, SupervisedConversionTimeoutError)
        assert isinstance(failure, ConversionTimeoutError)
        assert not failure.converter_wedged
        assert failure.reason == (
            "timed out after 3s (ocrmac); retried with rapidocr: timed out after 3s"
        )
        # Two timeouts each for two hanging documents over two processes.
        assert elapsed < 4 * 3.0 + 10
        assert supervisor.processes <= 2

        # The pool still converts after its processes were killed.
        [again] = _files(tmp_path, "c.md")
        assert _texts(await supervisor.convert_file(again)) == ["c.md by ocrmac"]
    finally:
        await supervisor.aclose()
    assert supervisor.processes == 0


async def test_a_unit_that_overruns_is_killed(tmp_path):
    supervisor = ConversionSupervisor(
        _config(unit_timeout=0.5, retry_ocr_engine=None), command=STUB
    )
    [path] = _files(tmp_path, "hang-always.md")
    try:
        with pytest.raises(SupervisedConversionTimeoutError) as failed:
            await supervisor.convert_file(path)
    finally:
        await supervisor.aclose()
    assert failed.value.reason == (
        "timed out after 0.5s (ocrmac); retried with ocrmac: timed out after 0.5s"
    )
    assert supervisor.processes == 0


async def test_convert_routes_docling_formats_through_the_supervisor(
    tmp_path, monkeypatch
):
    from haiku.rag.client.processing import convert

    monkeypatch.setattr(supervisor_module, "WORKER_COMMAND", STUB)
    config = _config()
    [markdown] = _files(tmp_path, "a.md")
    script = tmp_path / "s.py"
    script.write_text("print('hi')", encoding="utf-8")
    try:
        assert _texts(await convert(config, markdown)) == ["a.md by ocrmac"]
        assert any("print" in text for text in _texts(await convert(config, script)))
    finally:
        await close_supervisors()


async def test_a_crash_and_an_error_fail_with_their_reasons(tmp_path):
    supervisor = ConversionSupervisor(_config(), command=STUB)
    crash, bad = _files(tmp_path, "crash.md", "bad.md")
    try:
        with pytest.raises(ConversionFailedError) as crashed:
            await supervisor.convert_file(crash)
        assert crashed.value.reason == (
            "conversion process died (exit 3) (ocrmac); retried with rapidocr: "
            "conversion process died (exit 3)"
        )
        with pytest.raises(ConversionFailedError) as failed:
            await supervisor.convert_file(bad)
        assert "ValueError: cannot parse" in failed.value.reason
        # An error leaves the process fit for the next unit.
        assert supervisor.processes == 1
    finally:
        await supervisor.aclose()


async def test_the_retry_engine_defaults_to_the_configured_one(tmp_path):
    supervisor = ConversionSupervisor(_config(retry_ocr_engine=None), command=STUB)
    [bad] = _files(tmp_path, "bad.md")
    try:
        with pytest.raises(ConversionFailedError) as failed:
            await supervisor.convert_file(bad)
        assert "(ocrmac); retried with ocrmac:" in failed.value.reason
    finally:
        await supervisor.aclose()


async def test_a_long_pdf_converts_in_batches_keeping_page_numbers(tmp_path):
    supervisor = ConversionSupervisor(_config(page_batch=3), command=STUB)
    path = _pdf(tmp_path / "book.pdf", 7)
    try:
        document = await supervisor.convert_file(path)
    finally:
        await supervisor.aclose()

    assert sorted(document.pages) == [1, 2, 3, 4, 5, 6, 7]
    assert document.pages[3].size.height == 200
    pages = [(item.text, item.prov[0].page_no) for item in document.texts]
    assert pages == [(f"page {n}", n) for n in (1, 2, 4, 5, 6, 7)]


async def test_a_short_pdf_converts_whole(tmp_path):
    supervisor = ConversionSupervisor(_config(page_batch=10), command=STUB)
    path = _pdf(tmp_path / "short.pdf", 3)
    try:
        assert _texts(await supervisor.convert_file(path)) == ["short.pdf by ocrmac"]
    finally:
        await supervisor.aclose()


async def test_a_pdf_pdfium_cannot_open_converts_whole(tmp_path):
    supervisor = ConversionSupervisor(_config(page_batch=1), command=STUB)
    [path] = _files(tmp_path, "broken.pdf")
    try:
        assert _texts(await supervisor.convert_file(path)) == ["broken.pdf by ocrmac"]
    finally:
        await supervisor.aclose()


@pytest.mark.slow
async def test_a_failed_batch_fails_the_document_and_stops_the_rest(tmp_path):
    supervisor = ConversionSupervisor(
        _config(page_batch=2, unit_timeout=2.0, retry_ocr_engine=None), command=STUB
    )
    path = _pdf(tmp_path / "hangpage.pdf", 8)
    try:
        with pytest.raises(SupervisedConversionTimeoutError) as failed:
            await supervisor.convert_file(path)
        assert "pages 3-4" in str(failed.value)
        assert supervisor.busy == 0
    finally:
        await supervisor.aclose()
    assert supervisor.processes == 0


async def test_cancelling_a_conversion_kills_its_process(tmp_path):
    supervisor = ConversionSupervisor(_config(unit_timeout=60.0), command=STUB)
    [path] = _files(tmp_path, "hang-always.md")
    task = asyncio.ensure_future(supervisor.convert_file(path))
    while supervisor.busy == 0:
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert supervisor.processes == 0


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows has no signals to send a process"
)
async def test_a_conversion_outlives_stop_signals_to_its_process(tmp_path):
    import os

    supervisor = ConversionSupervisor(_config(), command=STUB)
    [path] = _files(tmp_path, "slow.md")
    task = asyncio.ensure_future(supervisor.convert_file(path))
    while supervisor.busy == 0:
        await asyncio.sleep(0.01)
    [worker] = supervisor._pool.busy
    for sig in (signal.SIGINT, signal.SIGTERM):
        os.kill(worker.process.pid, sig)
    document = await task
    # The first attempt finished: the retry would have used rapidocr.
    assert _texts(document) == ["slow.md by ocrmac"]
    await supervisor.aclose()


def test_the_worker_ignores_stop_signals(monkeypatch):
    ignored = []
    monkeypatch.setattr(
        supervisor_worker.signal, "signal", lambda sig, handler: ignored.append(sig)
    )
    supervisor_worker.ignore_stop_signals()
    assert ignored == [signal.SIGINT, signal.SIGTERM]


async def test_idle_processes_exit(tmp_path):
    supervisor = ConversionSupervisor(_config(idle_shutdown_s=0.2), command=STUB)
    [path] = _files(tmp_path, "a.md")
    await supervisor.convert_file(path)
    assert supervisor.processes == 1
    for _ in range(100):
        if supervisor.processes == 0:
            break
        await asyncio.sleep(0.05)
    assert supervisor.processes == 0
    # A retired process is not handed out again.
    assert _texts(await supervisor.convert_file(path)) == ["a.md by ocrmac"]
    await supervisor.aclose()


async def test_a_process_is_replaced_after_its_units(tmp_path):
    supervisor = ConversionSupervisor(_config(max_units_per_worker=1), command=STUB)
    [path] = _files(tmp_path, "a.md")
    await supervisor.convert_file(path)
    assert supervisor.processes == 0
    await supervisor.aclose()


async def test_closing_lets_a_running_unit_finish(tmp_path):
    seen = asyncio.Event()
    release = asyncio.Event()

    @asynccontextmanager
    async def gate():
        seen.set()
        await release.wait()
        yield

    supervisor = ConversionSupervisor(_config(), command=STUB, gate=gate)
    [path] = _files(tmp_path, "a.md")
    task = asyncio.ensure_future(supervisor.convert_file(path))
    await seen.wait()
    await supervisor.aclose()
    release.set()
    assert _texts(await task) == ["a.md by ocrmac"]
    assert supervisor.processes == 0


async def test_the_default_gate_applies_to_new_supervisors(tmp_path):
    entered = []

    @asynccontextmanager
    async def gate():
        entered.append(1)
        yield

    set_default_gate(gate)
    try:
        supervisor = ConversionSupervisor(_config(), command=STUB)
    finally:
        set_default_gate(None)
    [path] = _files(tmp_path, "a.md")
    try:
        await supervisor.convert_file(path)
    finally:
        await supervisor.aclose()
    assert entered == [1]
    assert ConversionSupervisor(_config(), command=STUB).gate is None


async def test_closing_cancels_idle_timers(tmp_path):
    supervisor = ConversionSupervisor(_config(), command=STUB)
    [path] = _files(tmp_path, "a.md")
    await supervisor.convert_file(path)
    await supervisor.aclose()
    assert supervisor.processes == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
async def test_a_process_that_will_not_exit_is_killed(tmp_path):
    supervisor = ConversionSupervisor(_config(), command=STUB)
    [path] = _files(tmp_path, "a.md")
    await supervisor.convert_file(path)
    [worker] = supervisor._pool.idle
    supervisor._pool.timers.pop(worker).cancel()
    supervisor._pool.idle.clear()
    worker.process.send_signal(signal.SIGSTOP)
    await worker.close(grace=0.2)
    assert worker.process.returncode == -signal.SIGKILL
    await worker.close()


@pytest.mark.parametrize(
    "command",
    [
        (sys.executable, "-c", "pass"),
        (sys.executable, "-c", "import time; time.sleep(60)"),
    ],
    ids=["exits", "never-ready"],
)
async def test_a_process_that_fails_to_start_fails_the_unit(
    tmp_path, monkeypatch, command
):
    monkeypatch.setattr(supervisor_module, "START_TIMEOUT", 0.5)
    supervisor = ConversionSupervisor(_config(), command=command)
    [path] = _files(tmp_path, "a.md")
    with pytest.raises(UnitFailure, match="failed to start"):
        await supervisor._attempt(supervisor_module._Unit(path, None), None)
    assert supervisor.processes == 0


def test_a_unit_label_names_its_pages():
    unit = supervisor_module._Unit(Path("/x/book.pdf"), None, (3, 4))
    assert unit.label == "/x/book.pdf pages 3-4"


async def test_the_registry_keeps_one_supervisor_per_configuration():
    config = _config()
    first = get_supervisor(config)
    assert get_supervisor(config.model_copy(deep=True)) is first
    other = _config(workers=1)
    assert get_supervisor(other) is not first
    assert set(running_supervisors()) == {first, get_supervisor(other)}
    await close_supervisors()
    assert running_supervisors() == []


def test_no_supervisors_run_outside_a_loop():
    assert running_supervisors() == []


async def test_the_default_command_runs_the_worker_module(monkeypatch):
    monkeypatch.setattr(supervisor_module, "WORKER_COMMAND", STUB)
    assert ConversionSupervisor(_config()).command == STUB


# --- the conversion process's side of the protocol ---


def _requests(*frames: dict) -> io.BytesIO:
    stream = io.BytesIO()
    for frame in frames:
        write_frame(stream, json.dumps(frame).encode())
    stream.seek(0)
    return stream


def _replies(stream: io.BytesIO) -> list[bytes]:
    stream.seek(0)
    frames = []
    while (frame := read_frame(stream)) is not None:
        frames.append(frame)
    return frames


def test_the_worker_answers_each_unit():
    def convert(config, path, source_uri, page_range, ocr_engine):
        if path.name == "bad.md":
            raise ValueError("nope")
        doc = DoclingDocument(name=path.stem)
        doc.add_text(
            label="text",  # ty: ignore[invalid-argument-type]
            text=f"{source_uri} {page_range} {ocr_engine}",
        )
        return doc

    setup = {"config": AppConfig().model_dump(mode="json"), "background": False}
    replies = io.BytesIO()
    supervisor_worker.serve(
        _requests(
            setup,
            {"path": "/a.md", "source_uri": "file:///a.md", "page_range": [2, 3]},
            {"path": "/bad.md"},
        ),
        replies,
        convert,
    )
    ready, status, body, error = _replies(replies)
    assert json.loads(ready) == {"status": "ready"}
    assert json.loads(status) == {"status": "ok"}
    document = DoclingDocument.model_validate_json(body)
    assert _texts(document) == ["file:///a.md (2, 3) None"]
    assert json.loads(error) == {"status": "error", "message": "ValueError: nope"}


def test_the_worker_stops_without_a_setup():
    replies = io.BytesIO()
    supervisor_worker.serve(io.BytesIO(), replies)
    assert replies.getvalue() == b""


def test_a_truncated_frame_ends_the_input():
    stream = io.BytesIO()
    write_frame(stream, b"0123456789")
    stream = io.BytesIO(stream.getvalue()[:-2])
    assert read_frame(stream) is None


def test_the_worker_lowers_its_priority_when_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(supervisor_worker, "lower_priority", lambda: calls.append(1))
    setup = {"config": AppConfig().model_dump(mode="json"), "background": True}
    supervisor_worker.serve(_requests(setup), io.BytesIO())
    assert calls == [1]


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_lowering_priority_nices_and_backgrounds_on_macos(monkeypatch, platform):
    import ctypes

    calls = []

    class Libc:
        def setpriority(self, *args):
            calls.append(("setpriority", *args))

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(
        supervisor_worker.os, "nice", lambda n: calls.append(("nice", n))
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: Libc())
    supervisor_worker.lower_priority()
    expected = [("nice", 10)]
    if platform == "darwin":
        expected.append(("setpriority", 4, 0, 0x1000))
    assert calls == expected


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("notes.md", b"# Title\n\nBody text", "Body text"),
        ("notes16.md", "# Title\n\nWide text".encode("utf-16"), "Wide text"),
        ("script.py", b"print('hi')", "print('hi')"),
    ],
)
def test_docling_convert_converts_in_the_calling_thread(
    tmp_path, name, content, expected
):
    path = tmp_path / name
    path.write_bytes(content)
    document = supervisor_worker.docling_convert(
        AppConfig(), path, None, None, "rapidocr"
    )
    assert any(expected in text for text in _texts(document))


def test_docling_convert_reports_a_file_it_cannot_parse(tmp_path):
    path = tmp_path / "empty.docx"
    path.write_bytes(b"not a docx")
    with pytest.raises(ValueError, match="Failed to parse file"):
        supervisor_worker.docling_convert(AppConfig(), path, None, None, None)


@pytest.mark.slow
async def test_real_docling_conversion_keeps_page_numbers_across_batches(
    docling_local_models,  # noqa: ARG001
):
    config = AppConfig()
    config.processing.supervisor.enabled = True
    config.processing.supervisor.page_batch = 4
    config.processing.supervisor.background = False
    options = config.processing.conversion_options
    options.do_ocr = False
    options.do_table_structure = False
    options.generate_page_images = False
    config.processing.pictures = "none"
    supervisor = ConversionSupervisor(config)
    path = Path(__file__).parents[1] / "data" / "doclaynet.pdf"
    try:
        document = await supervisor.convert_file(path)
    finally:
        await supervisor.aclose()
    assert sorted(document.pages) == list(range(1, 10))
    assert {prov.page_no for item in document.texts for prov in item.prov} == set(
        range(1, 10)
    )

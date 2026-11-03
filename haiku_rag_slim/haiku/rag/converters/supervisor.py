import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
import weakref
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from haiku.rag.config import AppConfig
from haiku.rag.converters.exceptions import (
    ConversionFailedError,
    SupervisedConversionTimeoutError,
)
from haiku.rag.converters.supervisor_worker import _LENGTH
from haiku.rag.telemetry import logfire

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)

WORKER_COMMAND = (sys.executable, "-m", "haiku.rag.converters.supervisor_worker")

# Seconds a new conversion process has to read its setup and answer ready. A
# unit's own deadline starts after that.
START_TIMEOUT = 120.0

# Every library a conversion process may spread threads over.
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "DOCLING_NUM_THREADS",
)


class UnitFailure(Exception):
    """One attempt at one unit failed. `timed_out` when its process was killed
    for overrunning."""

    def __init__(self, reason: str, *, timed_out: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.timed_out = timed_out


@dataclass(eq=False)
class _Worker:
    process: asyncio.subprocess.Process
    units: int = 0

    @property
    def stdin(self) -> asyncio.StreamWriter:
        assert self.process.stdin is not None
        return self.process.stdin

    @property
    def stdout(self) -> asyncio.StreamReader:
        assert self.process.stdout is not None
        return self.process.stdout

    async def send(self, payload: bytes) -> None:
        self.stdin.write(_LENGTH.pack(len(payload)) + payload)
        await self.stdin.drain()

    async def receive(self) -> bytes:
        (length,) = _LENGTH.unpack(await self.stdout.readexactly(_LENGTH.size))
        return await self.stdout.readexactly(length)

    async def kill(self) -> None:
        if self.process.returncode is None:
            self.process.kill()
        await self.process.wait()

    async def close(self, grace: float = 5.0) -> None:
        """Ask the process to exit by ending its input, killing it if it does
        not within `grace` seconds."""
        if self.process.returncode is None:
            self.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), grace)
            except TimeoutError:
                await self.kill()
        await self.process.wait()


@dataclass
class _Unit:
    path: Path
    source_uri: str | None
    page_range: tuple[int, int] | None = None

    @property
    def label(self) -> str:
        if self.page_range is None:
            return str(self.path)
        return f"{self.path} pages {self.page_range[0]}-{self.page_range[1]}"


@dataclass
class _Pool:
    """The live conversion processes, idle ones first in line."""

    idle: list[_Worker] = field(default_factory=list)
    busy: set[_Worker] = field(default_factory=set)
    timers: dict[_Worker, asyncio.TimerHandle] = field(default_factory=dict)


class ConversionSupervisor:
    """Runs docling-local conversions in a bounded pool of worker processes.

    A unit that overruns `unit_timeout` has its process killed, and a unit that
    fails in any way gets one retry, with `retry_ocr_engine` as the OCR engine.
    A PDF longer than `page_batch` pages converts as batches of pages spread over
    the pool, and the batches are joined keeping every page's number.
    `gate`, when set, is entered around each unit: a host uses it to hold
    conversions back.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        command: Sequence[str] | None = None,
        gate: "Callable[[], AbstractAsyncContextManager[None]] | None" = None,
    ):
        self.config = config
        self.settings = config.processing.supervisor
        self.command = tuple(command or WORKER_COMMAND)
        self.gate = gate
        self._slots = asyncio.Semaphore(self.settings.workers)
        self._pool = _Pool()
        self._closing = False
        self._setup = json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "background": self.settings.background,
            }
        ).encode()

    @property
    def processes(self) -> int:
        """Conversion processes alive now, busy or idle."""
        return len(self._pool.idle) + len(self._pool.busy)

    @property
    def busy(self) -> int:
        return len(self._pool.busy)

    async def convert_file(
        self, path: Path, source_uri: str | None = None
    ) -> "DoclingDocument":
        """Convert one file, in page batches when it is a long PDF."""
        sizes = None
        batch = self.settings.page_batch
        if path.suffix.lower() == ".pdf" and batch > 0:
            sizes = await asyncio.to_thread(_pdf_page_sizes, path)
        if sizes is None or len(sizes) <= batch:
            return await self._convert_unit(_Unit(path, source_uri))

        ranges = [
            (start, min(start + batch - 1, len(sizes)))
            for start in range(1, len(sizes) + 1, batch)
        ]
        with logfire.span(
            "document.convert_batches",
            path=str(path),
            pages=len(sizes),
            batches=len(ranges),
        ):
            tasks = [
                asyncio.ensure_future(
                    self._convert_unit(_Unit(path, source_uri, page_range))
                )
                for page_range in ranges
            ]
            try:
                parts = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        for part, page_range in zip(parts, ranges, strict=True):
            _keep_page_numbers(part, page_range, sizes)
        from docling_core.types.doc.document import DoclingDocument

        return await asyncio.to_thread(DoclingDocument.concatenate, parts)

    async def _convert_unit(self, unit: _Unit) -> "DoclingDocument":
        """One unit, retried once with the retry OCR engine."""
        configured = self.config.processing.conversion_options.ocr_engine
        retry_engine = self.settings.retry_ocr_engine
        try:
            return await self._attempt(unit, None)
        except UnitFailure as first:
            logger.warning(
                "Converting %s failed (%s), retrying with OCR engine %s",
                unit.label,
                first.reason,
                retry_engine or configured,
            )
            try:
                return await self._attempt(unit, retry_engine)
            except UnitFailure as second:
                reason = (
                    f"{first.reason} ({configured}); retried with "
                    f"{retry_engine or configured}: {second.reason}"
                )
                message = f"Failed to convert {unit.label}: {reason}"
                if second.timed_out:
                    raise SupervisedConversionTimeoutError(
                        message, reason=reason
                    ) from None
                raise ConversionFailedError(message, reason=reason) from None

    async def _attempt(self, unit: _Unit, ocr_engine: str | None) -> "DoclingDocument":
        from docling_core.types.doc.document import DoclingDocument

        request = json.dumps(
            {
                "path": str(unit.path),
                "source_uri": unit.source_uri,
                "page_range": unit.page_range,
                "ocr_engine": ocr_engine,
            }
        ).encode()
        timeout = self.settings.unit_timeout
        async with self._slots, (self.gate or contextlib.nullcontext)():
            worker = await self._checkout()
            reusable = False
            try:
                with logfire.span(
                    "document.convert_unit",
                    unit=unit.label,
                    ocr_engine=ocr_engine,
                    pid=worker.process.pid,
                ):
                    try:
                        await worker.send(request)
                        status = json.loads(
                            await asyncio.wait_for(worker.receive(), timeout)
                        )
                        if status["status"] != "ok":
                            reusable = True
                            raise UnitFailure(status["message"])
                        body = await asyncio.wait_for(worker.receive(), timeout)
                    except TimeoutError:
                        raise UnitFailure(
                            f"timed out after {timeout:g}s", timed_out=True
                        ) from None
                    except (asyncio.IncompleteReadError, ConnectionError):
                        await worker.kill()
                        raise UnitFailure(
                            f"conversion process died (exit {worker.process.returncode})"
                        ) from None
                    reusable = True
                worker.units += 1
                return await asyncio.to_thread(
                    DoclingDocument.model_validate_json, body
                )
            finally:
                await self._checkin(worker, reusable)

    async def _checkout(self) -> _Worker:
        if self._pool.idle:
            worker = self._pool.idle.pop()
            self._pool.timers.pop(worker).cancel()
        else:
            worker = await self._spawn()
        self._pool.busy.add(worker)
        return worker

    async def _checkin(self, worker: _Worker, reusable: bool) -> None:
        self._pool.busy.discard(worker)
        if (
            not reusable
            or self._closing
            or worker.process.returncode is not None
            or worker.units >= self.settings.max_units_per_worker
        ):
            # Shielded: a cancelled caller must still not leave a process behind.
            closing = worker.close() if reusable else worker.kill()
            await asyncio.shield(asyncio.ensure_future(closing))
            return
        loop = asyncio.get_running_loop()
        self._pool.timers[worker] = loop.call_later(
            self.settings.idle_shutdown_s, self._retire, worker
        )
        self._pool.idle.append(worker)

    def _retire(self, worker: _Worker) -> None:
        """An idle worker's timer ran out. Checkout and closing cancel it, so
        the worker is still idle."""
        del self._pool.timers[worker]
        self._pool.idle.remove(worker)
        asyncio.ensure_future(worker.close()).add_done_callback(_retrieve)

    async def _spawn(self) -> _Worker:
        threads = str(self.settings.threads_per_worker)
        env = {
            **os.environ,
            **dict.fromkeys(_THREAD_VARIABLES, threads),
            "TOKENIZERS_PARALLELISM": "false",
        }
        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=env,
        )
        worker = _Worker(process)
        try:
            await worker.send(self._setup)
            await asyncio.wait_for(worker.receive(), START_TIMEOUT)
        except (TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            await worker.kill()
            raise UnitFailure(
                f"conversion process failed to start (exit {process.returncode})"
            ) from None
        return worker

    async def aclose(self) -> None:
        """Stop every idle process. A busy one stops when its unit ends."""
        self._closing = True
        idle, self._pool.idle = self._pool.idle, []
        for worker in idle:
            self._pool.timers.pop(worker).cancel()
        await asyncio.gather(*(worker.close() for worker in idle))


def _retrieve(task: "asyncio.Future[None]") -> None:
    with contextlib.suppress(asyncio.CancelledError):
        task.exception()


def _pdf_page_sizes(path: Path) -> list[tuple[float, float]] | None:
    """Each page's width and height in points, or None when pdfium cannot open
    the file, which then converts as one unit for docling to judge."""
    import pypdfium2 as pdfium

    from haiku.rag.converters.pdf_split import PDFIUM_LOCK

    with PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(str(path))
        except pdfium.PdfiumError:
            return None
        try:
            return [pdf.get_page_size(i) for i in range(len(pdf))]
        finally:
            pdf.close()


def _keep_page_numbers(
    document: "DoclingDocument",
    page_range: tuple[int, int],
    sizes: list[tuple[float, float]],
) -> None:
    """Give a batch every page of its range: concatenation numbers a part's
    pages on from the highest page of the parts before it."""
    from docling_core.types.doc.base import Size

    start, end = page_range
    for page_no in range(start, end + 1):
        if page_no not in document.pages:
            width, height = sizes[page_no - 1]
            document.add_page(page_no=page_no, size=Size(width=width, height=height))


_SUPERVISORS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, ConversionSupervisor]]" = weakref.WeakKeyDictionary()

# The sections a conversion reads: equal ones convert alike.
_CONVERSION_SECTIONS = {"processing", "prompts", "providers"}


def get_supervisor(config: AppConfig) -> ConversionSupervisor:
    """The running loop's supervisor for this configuration's conversions."""
    loop = asyncio.get_running_loop()
    key = hashlib.md5(
        config.model_dump_json(include=_CONVERSION_SECTIONS).encode(),
        usedforsecurity=False,
    ).hexdigest()
    supervisors = _SUPERVISORS.setdefault(loop, {})
    supervisor = supervisors.get(key)
    if supervisor is None:
        supervisor = supervisors[key] = ConversionSupervisor(config)
    return supervisor


def running_supervisors() -> list[ConversionSupervisor]:
    with contextlib.suppress(RuntimeError):
        return list(_SUPERVISORS.get(asyncio.get_running_loop(), {}).values())
    return []


async def close_supervisors() -> None:
    """Close the running loop's supervisors and forget them."""
    supervisors = _SUPERVISORS.pop(asyncio.get_running_loop(), {})
    await asyncio.gather(*(s.aclose() for s in supervisors.values()))

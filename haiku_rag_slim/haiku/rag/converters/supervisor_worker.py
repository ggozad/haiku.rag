import json
import os
import signal
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument

    from haiku.rag.config import AppConfig

# A frame is an 8-byte big-endian length, then that many bytes.
_LENGTH = struct.Struct(">Q")

# <sys/resource.h>: put the whole process in the background band.
_PRIO_DARWIN_PROCESS = 4
_PRIO_DARWIN_BG = 0x1000

Convert = Callable[
    ["AppConfig", Path, str | None, tuple[int, int] | None, str | None],
    "DoclingDocument",
]


def write_frame(stream: IO[bytes], payload: bytes) -> None:
    stream.write(_LENGTH.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def read_frame(stream: IO[bytes]) -> bytes | None:
    """The next frame, or None at end of input."""
    header = stream.read(_LENGTH.size)
    if len(header) < _LENGTH.size:
        return None
    (length,) = _LENGTH.unpack(header)
    payload = stream.read(length)
    if len(payload) < length:
        return None
    return payload


def lower_priority() -> None:
    """Nice the process, and on macOS move it to background QoS."""
    os.nice(10)
    if sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.setpriority(_PRIO_DARWIN_PROCESS, 0, _PRIO_DARWIN_BG)


def ignore_stop_signals() -> None:
    """Leave this process's stop to its supervisor, which ends its input or
    kills it. A terminal's Ctrl+C and a service manager signal every process of
    the service, and the host stops by draining or releasing its jobs."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)


def docling_convert(
    config: "AppConfig",
    path: Path,
    source_uri: str | None,
    page_range: tuple[int, int] | None,
    ocr_engine: str | None,
) -> "DoclingDocument":
    from haiku.rag.converters.docling_local import DoclingLocalConverter

    if ocr_engine is not None:
        config = config.model_copy(deep=True)
        options = config.processing.conversion_options
        config.processing.conversion_options = options.model_copy(
            update={"ocr_engine": ocr_engine}
        )
    return DoclingLocalConverter(config).convert_file_sync(path, source_uri, page_range)


def serve(
    requests: IO[bytes], replies: IO[bytes], convert: Convert = docling_convert
) -> None:
    """Convert the units read from `requests` until it ends.

    The first frame is the setup: the configuration and whether to run at
    background priority, answered with a ready frame. Each later frame is one
    unit; each answer is a status frame, followed by the document's JSON when
    the status is ok.
    """
    from haiku.rag.config import AppConfig

    setup = read_frame(requests)
    if setup is None:
        return
    settings: dict[str, Any] = json.loads(setup)
    if settings.get("background"):
        lower_priority()
    config = AppConfig.model_validate(settings["config"])
    write_frame(replies, json.dumps({"status": "ready"}).encode())

    while (frame := read_frame(requests)) is not None:
        unit = json.loads(frame)
        page_range = unit.get("page_range")
        try:
            document = convert(
                config,
                Path(unit["path"]),
                unit.get("source_uri"),
                tuple(page_range) if page_range else None,
                unit.get("ocr_engine"),
            )
            body = document.model_dump_json().encode()
        except Exception as exc:
            status = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            write_frame(replies, json.dumps(status).encode())
            continue
        write_frame(replies, json.dumps({"status": "ok"}).encode())
        write_frame(replies, body)


def main(
    convert: Convert = docling_convert,
) -> None:  # pragma: no cover - runs in the conversion process
    """Serve over stdin and stdout. The protocol owns the real stdout, and
    whatever the converter prints goes to stderr."""
    ignore_stop_signals()
    replies = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    serve(sys.stdin.buffer, replies, convert)


if __name__ == "__main__":  # pragma: no cover - runs in the conversion process
    main()

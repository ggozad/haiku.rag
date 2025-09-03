import logging

from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.WARNING)


def get_logger() -> logging.Logger:
    logger = logging.getLogger("haiku.rag")

    handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
    )
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.setLevel("INFO")

    # Remove any existing handlers to avoid duplicates on reconfiguration
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    logger.addHandler(handler)
    return logger

import logging
import warnings

from rich.console import Console
from rich.logging import RichHandler


def get_logger() -> logging.Logger:
    """Return the library logger configured with a Rich handler."""
    logger = logging.getLogger("haiku.rag")

    handler = RichHandler(
        console=Console(stderr=True),
        rich_tracebacks=True,
    )
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicates on reconfiguration
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    logger.addHandler(handler)
    # Do not let messages propagate to the root logger
    logger.propagate = False
    return logger


def configure_cli_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for CLI runs.

    - Silence ALL non-haiku.rag loggers by detaching root handlers and setting
      their level to ERROR.
    - Attach a Rich handler only to the "haiku.rag" logger.
    - Prevent propagation so only our logger prints in the CLI.
    """
    # Silence root logger completely
    root = logging.getLogger()
    for hdlr in root.handlers[:]:
        root.removeHandler(hdlr)
    root.setLevel(logging.ERROR)

    # Optionally silence some commonly noisy libraries explicitly as a safeguard
    for noisy in ("httpx", "httpcore", "docling", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
        logging.getLogger(noisy).propagate = False

    # Configure and return our app logger
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False

    warnings.filterwarnings("ignore")
    return logger

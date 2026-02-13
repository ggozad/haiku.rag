import logging

from haiku.rag.logging import configure_cli_logging, get_logger


def test_get_logger():
    logger = get_logger()
    assert logger.name == "haiku.rag"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert logger.propagate is False


def test_get_logger_idempotent():
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2
    assert len(logger2.handlers) == 1


def test_configure_cli_logging():
    logger = configure_cli_logging()
    assert logger.name == "haiku.rag"
    assert logger.propagate is False

    root = logging.getLogger()
    assert root.level == logging.ERROR
    assert len(root.handlers) == 0

    for name in ("httpx", "httpcore", "docling", "urllib3", "asyncio"):
        noisy = logging.getLogger(name)
        assert noisy.level == logging.ERROR
        assert noisy.propagate is False


def test_configure_cli_logging_custom_level():
    logger = configure_cli_logging(level=logging.DEBUG)
    assert logger.level == logging.DEBUG

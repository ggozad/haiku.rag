class IngesterError(Exception):
    """Base class for ingester errors that the worker classifies on."""


class PermanentError(IngesterError):
    """The job will never succeed without intervention (bad URI, unsupported
    content type, 410 Gone, etc.). Goes straight to dead — no retry.

    `fatal_to_process` marks a failure that also left this process unable to
    convert, so the worker terminates once the job is recorded dead.
    """

    def __init__(self, message: str, *, fatal_to_process: bool = False):
        super().__init__(message)
        self.fatal_to_process = fatal_to_process


class TransientError(IngesterError):
    """The job might succeed on a future attempt (network hiccup, 5xx, DB
    busy). Worker reschedules with backoff up to max_attempts, then dead."""

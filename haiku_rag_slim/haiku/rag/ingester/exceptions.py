class IngesterError(Exception):
    """Base class for ingester errors that the worker classifies on."""


class PermanentError(IngesterError):
    """The job will never succeed without intervention (bad URI, unsupported
    content type, 410 Gone, etc.). Goes straight to dead — no retry."""


class TransientError(IngesterError):
    """The job might succeed on a future attempt (network hiccup, 5xx, DB
    busy). Worker reschedules with backoff up to max_attempts, then dead."""

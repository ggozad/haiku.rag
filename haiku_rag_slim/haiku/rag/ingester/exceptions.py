class IngesterError(Exception):
    """Base class for ingester errors that the worker classifies on."""


class PermanentError(IngesterError):
    """The job will never succeed without intervention (bad URI, unsupported
    content type, 410 Gone, etc.). Goes straight to dead — no retry.

    `conversion_stalled` marks a document whose conversion exceeded its
    deadline, so the same bytes stall again and discovery must not rediscover
    them. `fatal_to_process` marks a failure that also left this process unable
    to convert, so the worker terminates once the job is recorded dead.
    """

    def __init__(
        self,
        message: str,
        *,
        conversion_stalled: bool = False,
        fatal_to_process: bool = False,
    ):
        super().__init__(message)
        self.conversion_stalled = conversion_stalled
        self.fatal_to_process = fatal_to_process


class TransientError(IngesterError):
    """The job might succeed on a future attempt (network hiccup, 5xx, DB
    busy). Worker reschedules with backoff up to max_attempts, then dead."""


class BlockingTombstoneError(IngesterError):
    """A retry collided with the stalled-conversion row holding that slot.

    `blocking_job_id` is the row to deal with first: retry it, or delete the
    document.
    """

    def __init__(self, job_id: str, blocking_job_id: str):
        super().__init__(
            f"Job {job_id!r} cannot be retried while job {blocking_job_id!r} "
            f"holds the slot for the same source, URI, op and revision after a "
            f"stalled conversion."
        )
        self.job_id = job_id
        self.blocking_job_id = blocking_job_id

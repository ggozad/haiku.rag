class IngesterError(Exception):
    """Base class for ingester errors that the worker classifies on."""


class PermanentError(IngesterError):
    """The job will never succeed without intervention (bad URI, unsupported
    content type, 410 Gone, etc.). Goes straight to dead — no retry.

    `conversion_stalled` marks a document whose conversion exceeded its
    deadline, so the same bytes stall again and discovery must not rediscover
    them. `fatal_to_process` marks a failure that also left this process unable
    to convert, so the worker terminates once the job is recorded dead.
    `rejected` marks a document a hook refused, recorded `rejected` instead of
    dead.
    """

    def __init__(
        self,
        message: str,
        *,
        conversion_stalled: bool = False,
        fatal_to_process: bool = False,
        rejected: bool = False,
    ):
        super().__init__(message)
        self.conversion_stalled = conversion_stalled
        self.fatal_to_process = fatal_to_process
        self.rejected = rejected


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


class QueueError(IngesterError):
    """The queue cannot be used as configured."""


class EmptyListingError(IngesterError):
    """A source listed nothing while URIs it listed before remain known."""

    def __init__(self, source_id: str, known: int):
        super().__init__(
            f"Source {source_id!r} listed no documents, and {known} URIs it "
            f"listed before are still known. Nothing was deleted. If the source "
            f"was emptied on purpose, remove it from the configuration, and "
            f"delete its documents by hand if they should go."
        )
        self.source_id = source_id
        self.known = known

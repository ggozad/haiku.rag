class ConversionTimeoutError(TimeoutError):
    """A conversion exceeded `processing.conversion_timeout`.

    `converter_wedged` marks the abandoned conversion as one that held the
    shared docling converter, so nothing in this process can convert through
    it again.
    """

    def __init__(self, message: str, *, converter_wedged: bool):
        super().__init__(message)
        self.converter_wedged = converter_wedged


class ConverterWedgedError(RuntimeError):
    """The shared docling converter was abandoned by a timed-out conversion
    and this process cannot convert through it again. The document is not at
    fault."""


class SupervisedConversionTimeoutError(ConversionTimeoutError):
    """A supervised conversion overran its deadline on every attempt.

    Each attempt's process was killed, so nothing in this process is wedged.
    `reason` names each attempt's failure.
    """

    def __init__(self, message: str, *, reason: str):
        super().__init__(message, converter_wedged=False)
        self.reason = reason


class ConversionFailedError(ValueError):
    """A supervised conversion failed on every attempt, by an error or by its
    process dying. `reason` names each attempt's failure."""

    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason = reason

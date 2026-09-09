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

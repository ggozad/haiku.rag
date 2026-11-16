from collections.abc import Iterator


def causes(exc: BaseException) -> Iterator[BaseException]:
    """The exceptions behind `exc`, including every member of a group, such as
    anyio's group of one failed connection per resolved address."""
    seen: set[int] = set()
    pending = [exc.__cause__ or exc.__context__]
    while pending:
        cause = pending.pop()
        if cause is None or id(cause) in seen:
            continue
        seen.add(id(cause))
        yield cause
        pending.append(cause.__cause__ or cause.__context__)
        if isinstance(cause, BaseExceptionGroup):
            pending.extend(cause.exceptions)

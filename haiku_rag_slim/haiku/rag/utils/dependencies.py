from typing import NoReturn


def raise_missing_extra(module: str, extra: str, exc: ModuleNotFoundError) -> NoReturn:
    """Report `module` as a missing optional dependency, naming its extra."""
    if exc.name != module:
        raise exc
    raise ImportError(
        f"{module} is not installed. Install it with "
        f"`uv pip install 'haiku.rag-slim[{extra}]'`."
    ) from exc

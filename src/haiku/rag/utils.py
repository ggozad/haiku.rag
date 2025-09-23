import asyncio
import importlib
import importlib.util
import sys
from collections.abc import Callable
from functools import wraps
from importlib import metadata
from io import BytesIO
from pathlib import Path
from types import ModuleType

from packaging.version import Version, parse


def debounce(wait: float) -> Callable:
    """
    A decorator to debounce a function, ensuring it is called only after a specified delay
    and always executes after the last call.

    Args:
        wait (float): The debounce delay in seconds.

    Returns:
        Callable: The decorated function.
    """

    def decorator(func: Callable) -> Callable:
        last_call = None
        task = None

        @wraps(func)
        async def debounced(*args, **kwargs):
            nonlocal last_call, task
            last_call = asyncio.get_event_loop().time()

            if task:
                task.cancel()

            async def call_func():
                await asyncio.sleep(wait)
                if asyncio.get_event_loop().time() - last_call >= wait:  # type: ignore
                    await func(*args, **kwargs)

            task = asyncio.create_task(call_func())

        return debounced

    return decorator


def get_default_data_dir() -> Path:
    """Get the user data directory for the current system platform.

    Linux: ~/.local/share/haiku.rag
    macOS: ~/Library/Application Support/haiku.rag
    Windows: C:/Users/<USER>/AppData/Roaming/haiku.rag

    Returns:
        User Data Path.
    """
    home = Path.home()

    system_paths = {
        "win32": home / "AppData/Roaming/haiku.rag",
        "linux": home / ".local/share/haiku.rag",
        "darwin": home / "Library/Application Support/haiku.rag",
    }

    data_path = system_paths[sys.platform]
    return data_path


async def is_up_to_date() -> tuple[bool, Version, Version]:
    """Check whether haiku.rag is current.

    Returns:
        A tuple containing a boolean indicating whether haiku.rag is current,
        the running version and the latest version.
    """

    # Lazy import to avoid pulling httpx (and its deps) on module import
    import httpx

    async with httpx.AsyncClient() as client:
        running_version = parse(metadata.version("haiku.rag"))
        try:
            response = await client.get("https://pypi.org/pypi/haiku.rag/json")
            data = response.json()
            pypi_version = parse(data["info"]["version"])
        except Exception:
            # If no network connection, do not raise alarms.
            pypi_version = running_version
    return running_version >= pypi_version, running_version, pypi_version


def text_to_docling_document(text: str, name: str = "content.md"):
    """Convert text content to a DoclingDocument.

    Args:
        text: The text content to convert.
        name: The name to use for the document stream (defaults to "content.md").

    Returns:
        A DoclingDocument created from the text content.
    """
    # Lazy import docling deps to keep import-time light
    from docling.document_converter import DocumentConverter  # type: ignore
    from docling_core.types.io import DocumentStream  # type: ignore

    bytes_io = BytesIO(text.encode("utf-8"))
    doc_stream = DocumentStream(name=name, stream=bytes_io)
    converter = DocumentConverter()
    result = converter.convert(doc_stream)
    return result.document


def load_callable(path: str):
    """Load a callable from a dotted path or file path.

    Supported formats:
    - "package.module:func" or "package.module.func"
    - "path/to/file.py:func"

    Returns the loaded callable. Raises ValueError on failure.
    """
    if not path:
        raise ValueError("Empty callable path provided")

    module_part = None
    func_name = None

    if ":" in path:
        module_part, func_name = path.split(":", 1)
    else:
        # split by last dot for module.attr
        if "." in path:
            module_part, func_name = path.rsplit(".", 1)
        else:
            raise ValueError(
                "Invalid callable path format. Use 'module:func' or 'module.func' or 'file.py:func'."
            )

    # Try file path first
    mod: ModuleType | None = None
    module_path = Path(module_part)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
    else:
        # Import as a module path
        try:
            mod = importlib.import_module(module_part)
        except Exception as e:
            raise ValueError(f"Failed to import module '{module_part}': {e}")

    if not hasattr(mod, func_name):
        raise ValueError(f"Callable '{func_name}' not found in module '{module_part}'")
    func = getattr(mod, func_name)
    if not callable(func):
        raise ValueError(
            f"Attribute '{func_name}' in module '{module_part}' is not callable"
        )
    return func


def prefetch_models():
    """Prefetch runtime models (Docling + Ollama as configured)."""
    import httpx
    from docling.utils.model_downloader import download_models

    from haiku.rag.config import Config

    download_models()

    # Collect Ollama models from config
    required_models: set[str] = set()
    if Config.EMBEDDINGS_PROVIDER == "ollama":
        required_models.add(Config.EMBEDDINGS_MODEL)
    if Config.QA_PROVIDER == "ollama":
        required_models.add(Config.QA_MODEL)
    if Config.RESEARCH_PROVIDER == "ollama":
        required_models.add(Config.RESEARCH_MODEL)
    if Config.RERANK_PROVIDER == "ollama":
        required_models.add(Config.RERANK_MODEL)

    if not required_models:
        return

    base_url = Config.OLLAMA_BASE_URL

    with httpx.Client(timeout=None) as client:
        for model in sorted(required_models):
            with client.stream(
                "POST", f"{base_url}/api/pull", json={"model": model}
            ) as r:
                for _ in r.iter_lines():
                    pass

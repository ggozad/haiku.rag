import importlib
import importlib.util
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType

from packaging.version import Version, parse


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
        running_version = parse(metadata.version("haiku.rag-slim"))
        try:
            response = await client.get("https://pypi.org/pypi/haiku.rag/json")
            data = response.json()
            pypi_version = parse(data["info"]["version"])
        except Exception:
            # If no network connection, do not raise alarms.
            pypi_version = running_version
    return running_version >= pypi_version, running_version, pypi_version


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
    """Prefetch runtime models (Docling + Ollama + HuggingFace tokenizer as configured)."""
    import httpx

    from haiku.rag.config import Config

    try:
        from docling.utils.model_downloader import download_models

        download_models()
    except ImportError:
        # Docling not installed, skip downloading docling models
        pass

    # Download HuggingFace tokenizer
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(Config.processing.chunking_tokenizer)

    # Collect Ollama models from config
    required_models: set[str] = set()
    if Config.embeddings.provider == "ollama":
        required_models.add(Config.embeddings.model)
    if Config.qa.provider == "ollama":
        required_models.add(Config.qa.model)
    if Config.research.provider == "ollama":
        required_models.add(Config.research.model)
    if Config.reranking.provider == "ollama":
        required_models.add(Config.reranking.model)

    if not required_models:
        return

    base_url = Config.providers.ollama.base_url

    with httpx.Client(timeout=None) as client:
        for model in sorted(required_models):
            with client.stream(
                "POST", f"{base_url}/api/pull", json={"model": model}
            ) as r:
                for _ in r.iter_lines():
                    pass

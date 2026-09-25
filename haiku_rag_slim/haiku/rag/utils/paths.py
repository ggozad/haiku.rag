import sys
from pathlib import Path


def locate_database(location: str) -> Path | str:
    """A configured location as a URI, or as a local path."""
    if "://" in location:
        return location
    return Path(location)


def get_default_data_dir() -> Path:
    """Get the user data directory for the current system platform."""
    home = Path.home()

    system_paths = {
        "win32": home / "AppData/Roaming/haiku.rag",
        "linux": home / ".local/share/haiku.rag",
        "darwin": home / "Library/Application Support/haiku.rag",
    }

    data_path = system_paths[sys.platform]
    return data_path

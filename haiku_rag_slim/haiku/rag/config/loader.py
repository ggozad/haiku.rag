import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def find_config_file(cli_path: Path | None = None) -> Path | None:
    """Find the YAML config file using the search path.

    Search order:
    1. CLI-provided path (via HAIKU_RAG_CONFIG_PATH env var or parameter)
    2. ./haiku.rag.yaml (current directory)
    3. Platform-specific user config directory

    Returns None if no config file is found.
    """
    # Check environment variable first (set by CLI --config flag)
    if not cli_path:
        env_path = os.getenv("HAIKU_RAG_CONFIG_PATH")
        if env_path:
            cli_path = Path(env_path).expanduser()

    if cli_path:
        if cli_path.exists():
            return cli_path
        raise FileNotFoundError(f"Config file not found: {cli_path}")

    cwd_config = Path.cwd() / "haiku.rag.yaml"
    if cwd_config.exists():
        return cwd_config

    # Use same directory as data storage for config
    from haiku.rag.utils import get_default_data_dir

    data_dir = get_default_data_dir()
    user_config = data_dir / "haiku.rag.yaml"
    if user_config.exists():
        return user_config

    return None


def load_yaml_config(path: Path) -> dict:
    """Load and parse a YAML config file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data or {}


def generate_default_config() -> dict:
    """Generate a default YAML config structure from AppConfig defaults."""
    from haiku.rag.config.models import AppConfig

    default_config = AppConfig()
    return default_config.model_dump(mode="json", exclude_none=False)


_SECRET_KEY_HINTS = ("key", "password", "token", "secret")


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_KEY_HINTS)


def redact_secrets(data: Any) -> Any:
    """Recursively mask secret-bearing values in a config dump. Any scalar
    whose key contains key/password/token/secret becomes "***" when set or
    None when unset; everything else is preserved."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if (
                isinstance(key, str)
                and _is_secret_key(key)
                and not isinstance(value, (dict, list))
            ):
                result[key] = "***" if value else None
            else:
                result[key] = redact_secrets(value)
        return result
    if isinstance(data, list):
        return [redact_secrets(item) for item in data]
    return data

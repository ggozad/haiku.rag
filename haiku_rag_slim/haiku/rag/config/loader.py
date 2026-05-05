import logging
import os
from pathlib import Path

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
    data = data or {}
    _drop_legacy_generate_picture_images(data)
    return data


def _drop_legacy_generate_picture_images(data: dict) -> None:
    """Drop ``processing.conversion_options.generate_picture_images`` from
    loaded YAML and log that it is no longer needed.

    Picture bytes are always extracted now, so the flag has no effect.
    Old YAMLs keep working; users see one log line telling them they can
    remove the entry.
    """
    processing = data.get("processing")
    if not isinstance(processing, dict):
        return
    opts = processing.get("conversion_options")
    if isinstance(opts, dict) and "generate_picture_images" in opts:
        opts.pop("generate_picture_images", None)
        logger.warning(
            "Config: 'processing.conversion_options.generate_picture_images' "
            "no longer has any effect (picture bytes are always extracted). "
            "Remove it from your haiku.rag.yaml."
        )


def generate_default_config() -> dict:
    """Generate a default YAML config structure from AppConfig defaults."""
    from haiku.rag.config.models import AppConfig

    default_config = AppConfig()
    return default_config.model_dump(mode="json", exclude_none=False)

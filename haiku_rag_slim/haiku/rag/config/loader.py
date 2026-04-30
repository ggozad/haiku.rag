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
    _translate_legacy_picture_fields(data)
    return data


def _translate_legacy_picture_fields(data: dict) -> None:
    """Map pre-A4 picture knobs onto ``processing.pictures``.

    Pre-A4 the same intent was expressed by two booleans on
    ``conversion_options``: ``generate_picture_images`` and
    ``picture_description.enabled``. Translation, in priority order:

    - ``picture_description.enabled = true`` (regardless of the image flag)
      → ``pictures = "description"``. Mirrors the original behavior where
      enabling the VLM implicitly forced docling to produce picture bytes.
    - ``generate_picture_images = true`` (and no description) → ``"image"``.
    - both false / missing → no translation; default ``"none"`` applies.

    If ``pictures`` is already set on the loaded YAML it wins — users who
    have migrated keep their explicit choice. Mutates ``data`` in-place
    and emits one warning per legacy field encountered.
    """
    processing = data.get("processing")
    if not isinstance(processing, dict):
        return

    if "pictures" in processing:
        # User has migrated; legacy fields may still be present but should not
        # override the explicit choice. Drop them silently to avoid confusion.
        opts = processing.get("conversion_options")
        if isinstance(opts, dict):
            opts.pop("generate_picture_images", None)
            pic = opts.get("picture_description")
            if isinstance(pic, dict):
                pic.pop("enabled", None)
        return

    opts = processing.get("conversion_options")
    if not isinstance(opts, dict):
        return

    pic = opts.get("picture_description") if isinstance(opts, dict) else None
    legacy_describe = pic.pop("enabled", None) if isinstance(pic, dict) else None
    legacy_image = opts.pop("generate_picture_images", None)

    if legacy_describe:
        processing["pictures"] = "description"
        logger.warning(
            "Config: 'processing.conversion_options.picture_description.enabled=true' is "
            "deprecated; mapped to 'processing.pictures: description'. Please update your "
            "haiku.rag.yaml."
        )
    elif legacy_image:
        processing["pictures"] = "image"
        logger.warning(
            "Config: 'processing.conversion_options.generate_picture_images=true' is "
            "deprecated; mapped to 'processing.pictures: image'. Please update your "
            "haiku.rag.yaml."
        )


def generate_default_config() -> dict:
    """Generate a default YAML config structure from AppConfig defaults."""
    from haiku.rag.config.models import AppConfig

    default_config = AppConfig()
    return default_config.model_dump(mode="json", exclude_none=False)

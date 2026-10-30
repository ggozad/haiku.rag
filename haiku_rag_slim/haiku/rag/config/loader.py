import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_INTERPOLATION_PATTERN = re.compile(
    r"\$\$|\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}|\$\{file:(?P<file>[^}]*)\}"
)


class MissingEnvVarError(ValueError):
    """A ${VAR} in the config references an unset or empty environment variable."""


def _expand_str(value: str, reserved: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        if match.group(0) == "$$":
            return "$"
        if (file_name := match.group("file")) is not None:
            path = Path(reserved.get("HAIKU_RAG_CONFIG_DIR", ".")) / file_name
            try:
                contents = path.read_bytes().decode("utf-8").rstrip("\r\n")
            except OSError as error:
                raise ValueError(
                    f"Config file reference {str(path)!r} cannot be read: "
                    f"{error.strerror}."
                ) from None
            except UnicodeError:
                raise ValueError(
                    f"Config file reference {str(path)!r} is not UTF-8 text."
                ) from None
            if not contents:
                raise ValueError(f"Config file reference {str(path)!r} is empty.")
            return contents
        name, default = match.group(1), match.group(2)
        if name in reserved:
            return reserved[name]
        if os.environ.get(name, "") != "":
            return os.environ[name]
        if default is not None:
            return default
        raise MissingEnvVarError(
            f"Config references unset or empty environment variable ${{{name}}}. "
            f"Set it, or use ${{{name}:-default}} to provide a fallback."
        )

    return _INTERPOLATION_PATTERN.sub(replace, value)


def expand_env_vars(data: Any, reserved: Mapping[str, str] | None = None) -> Any:
    """Expand environment and ${file:path} references in string values.
    Keys and non-string scalars are left untouched; $$ collapses to a literal $.
    Reserved values override the environment; HAIKU_RAG_CONFIG_DIR roots file paths."""
    reserved = reserved or {}
    if isinstance(data, dict):
        return {key: expand_env_vars(value, reserved) for key, value in data.items()}
    if isinstance(data, list):
        return [expand_env_vars(item, reserved) for item in data]
    if isinstance(data, str):
        return _expand_str(data, reserved)
    return data


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
    from haiku.rag.utils.paths import get_default_data_dir

    data_dir = get_default_data_dir()
    user_config = data_dir / "haiku.rag.yaml"
    if user_config.exists():
        return user_config

    return None


def _read_yaml(path: Path, config_dir: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return expand_env_vars(data or {}, {"HAIKU_RAG_CONFIG_DIR": str(config_dir)})


def load_yaml_config(path: Path) -> dict:
    """Load YAML with environment and file references expanded, merging the
    database fragments in the `databases_include` directory."""
    config_dir = path.resolve().parent
    data = _read_yaml(path, config_dir)
    if isinstance(data, dict) and (include := data.pop("databases_include", None)):
        _merge_fragments(data, path, config_dir / include, config_dir)
    return data


_FRAGMENT_SECTIONS = ("lancedb", "ingester")


def _merge_fragments(data: dict, path: Path, directory: Path, config_dir: Path) -> None:
    """Add the database maps of every `*.yaml` in `directory` to `data`.
    A fragment's `${HAIKU_RAG_CONFIG_DIR}` is the main file's directory."""
    defined = {
        (section, name): path
        for section in _FRAGMENT_SECTIONS
        for name in ((data.get(section) or {}).get("databases") or {})
    }
    if not directory.is_dir():
        return
    for fragment in sorted(directory.glob("*.yaml")):
        if fragment.name.startswith(".") or not fragment.is_file():
            continue
        try:
            content = _read_yaml(fragment, config_dir)
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise ValueError(f"{fragment}: {error}") from None
        if not isinstance(content, dict) or not all(
            section in _FRAGMENT_SECTIONS
            and isinstance(value, dict)
            and set(value) <= {"databases"}
            and isinstance(value.get("databases") or {}, dict)
            for section, value in content.items()
        ):
            raise ValueError(
                f"{fragment}: a database fragment may set only lancedb.databases "
                "and ingester.databases"
            )
        for section, value in content.items():
            for name, entry in (value.get("databases") or {}).items():
                if (earlier := defined.get((section, name))) is not None:
                    raise ValueError(
                        f"{section}.databases {name!r} is defined in both "
                        f"{earlier} and {fragment}"
                    )
                defined[section, name] = fragment
                target = data[section] = data.get(section) or {}
                target["databases"] = target.get("databases") or {}
                target["databases"][name] = entry


def generate_default_config() -> dict:
    """Generate a default YAML config structure from AppConfig defaults."""
    from haiku.rag.config.models import AppConfig

    default_config = AppConfig()
    return default_config.model_dump(mode="json", exclude_none=False)


# Never part of an innocent key name, so matched anywhere in it.
_SECRET_KEY_FRAGMENTS = ("password", "passwd", "secret", "credential")
# Found inside innocent names (max_tokens, keywords), so matched as whole words.
_SECRET_KEY_WORDS = frozenset({"key", "apikey", "token"})
_HEADER_KEYS = frozenset({"headers", "fetch_headers"})
# SQLAlchemy's userinfo: a password runs to the first `@` and may hold `/` or `#`.
_SQLALCHEMY_USERINFO = re.compile(r"[^:/@]*(?::[^@]*)?@")


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS):
        return True
    words = re.split(r"[^a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower())
    return not _SECRET_KEY_WORDS.isdisjoint(words)


def _mask(value: Any) -> str | None:
    return "***" if value else None


def _mask_userinfo(value: str) -> str:
    """`value` with any URL userinfo replaced by `***`."""
    scheme, separator, rest = value.partition("://")
    if not separator:
        return value
    # urllib and HTTPX end userinfo at the authority's last `@`. Outside HTTP,
    # mask to the later end of either reading, so neither parser's password shows.
    authority = re.split(r"[/?#]", rest, maxsplit=1)[0]
    end = authority.rfind("@") + 1
    if scheme.lower() not in ("http", "https") and (
        match := _SQLALCHEMY_USERINFO.match(rest)
    ):
        end = max(end, match.end())
    return f"{scheme}://***@{rest[end:]}" if end else value


def redact_secrets(data: Any) -> Any:
    """Recursively mask secret-bearing values in a config dump.

    Masks scalars under a key naming a key, password, token, secret or
    credential, every value under `headers` or `fetch_headers`, and the
    userinfo of any URL. Set values become "***", unset ones None.
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key in _HEADER_KEYS and isinstance(value, dict):
                result[key] = {name: _mask(item) for name, item in value.items()}
            elif (
                isinstance(key, str)
                and _is_secret_key(key)
                and not isinstance(value, (dict, list))
            ):
                result[key] = _mask(value)
            else:
                result[key] = redact_secrets(value)
        return result
    if isinstance(data, list):
        return [redact_secrets(item) for item in data]
    if isinstance(data, str):
        return _mask_userinfo(data)
    return data

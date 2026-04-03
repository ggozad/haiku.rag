import pathlib
import shutil
from importlib.metadata import version

from jinja2 import Environment, PackageLoader

AVAILABLE_TOOLS: set[str] = {
    "list_documents",
    "get_document",
    "search",
    "ask",
    "research",
    "analyze",
}

DEFAULT_PREAMBLE = (
    "You are a RAG (Retrieval Augmented Generation) assistant "
    "with access to a document knowledge base.\n"
    "Use your tools to search and answer questions. "
    "Never make up information — always use tools to get facts "
    "from the knowledge base."
)

DEFAULT_DESCRIPTION = (
    "Search, retrieve and analyze documents using RAG (Retrieval Augmented Generation)."
)


def _get_env() -> Environment:
    return Environment(
        loader=PackageLoader("haiku.rag.skill_generator", "templates"),
        autoescape=False,
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
    )


def validate_metadata(name: str, description: str) -> None:
    from haiku.skills import SkillMetadata

    SkillMetadata(name=name, description=description)


def validate_tools(tools: list[str]) -> None:
    if not tools:
        raise ValueError("tools must contain at least one tool")
    unknown = set(tools) - AVAILABLE_TOOLS
    if unknown:
        raise ValueError(
            f"Unknown tools: {', '.join(sorted(unknown))}."
            f" Available: {', '.join(sorted(AVAILABLE_TOOLS))}"
        )


def validate_db_path(db_path: pathlib.Path) -> None:
    if not db_path.exists():
        raise ValueError(f"db_path does not exist: {db_path}")
    if not db_path.is_dir():
        raise ValueError(f"db_path is not a directory: {db_path}")


def validate_output_dir(output_dir: pathlib.Path, name: str) -> None:
    if not output_dir.exists():
        raise ValueError(f"output_dir does not exist: {output_dir}")
    target = output_dir / f"{name}-skill"
    if target.exists():
        raise ValueError(f"Target directory already exists: {target}")


def render_templates(
    output_dir: pathlib.Path,
    name: str,
    description: str,
    tool_names: list[str],
    preamble: str | None = None,
    remote: bool = False,
) -> pathlib.Path:
    if preamble is None:
        preamble = DEFAULT_PREAMBLE

    pkg_name = name.replace("-", "_")
    rag_version = version("haiku.rag-slim")
    env = _get_env()
    context = {
        "name": name,
        "pkg_name": pkg_name,
        "description": description,
        "tool_names": tool_names,
        "preamble": preamble,
        "rag_version": rag_version,
        "remote": remote,
    }

    result_dir = output_dir / f"{name}-skill"
    pkg_dir = result_dir / f"{pkg_name}_skill"
    assets_dir = pkg_dir / "assets"
    assets_dir.mkdir(parents=True)

    # Render pyproject.toml
    template = env.get_template("pyproject.toml.j2")
    (result_dir / "pyproject.toml").write_text(template.render(context))

    # Render README.md
    template = env.get_template("README.md.j2")
    (result_dir / "README.md").write_text(template.render(context))

    # Render __init__.py
    template = env.get_template("__init__.py.j2")
    (pkg_dir / "__init__.py").write_text(template.render(context))

    # Render SKILL.md
    template = env.get_template("SKILL.md.j2")
    (pkg_dir / "SKILL.md").write_text(template.render(context))

    return result_dir


def generate_skill(
    db_path: pathlib.Path | None,
    output_dir: pathlib.Path,
    name: str,
    description: str,
    tool_names: list[str],
    config_path: pathlib.Path | None = None,
    preamble: str | None = None,
) -> pathlib.Path:
    validate_metadata(name, description)
    validate_tools(tool_names)

    if db_path is None:
        if config_path is None:
            raise ValueError(
                "config_path is required when db_path is not provided "
                "(remote storage needs connection config)"
            )
    else:
        validate_db_path(db_path)

    validate_output_dir(output_dir, name)

    result = render_templates(
        output_dir=output_dir,
        name=name,
        description=description,
        tool_names=tool_names,
        preamble=preamble,
        remote=db_path is None,
    )

    pkg_name = name.replace("-", "_")
    assets_dir = result / f"{pkg_name}_skill" / "assets"

    if db_path is not None:
        shutil.copytree(db_path, assets_dir / f"{name}.lancedb")

    if config_path is not None:
        shutil.copy2(config_path, assets_dir / "haiku.rag.yaml")

    return result

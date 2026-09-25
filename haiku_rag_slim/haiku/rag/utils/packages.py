from importlib import metadata

from packaging.version import Version, parse


def get_package_versions() -> dict[str, str]:
    """Get versions of haiku.rag and its dependencies."""
    from docling_core.types.doc.document import DoclingDocument

    versions = {
        "haiku_rag": metadata.version("haiku.rag-slim"),
        "lancedb": metadata.version("lancedb"),
        "pydantic_ai": metadata.version("pydantic-ai-slim"),
        "docling_document_schema": DoclingDocument.model_construct().version,
    }
    try:
        versions["docling"] = metadata.version("docling")
    except metadata.PackageNotFoundError:
        versions["docling"] = "not installed"
    return versions


async def is_up_to_date() -> tuple[bool, Version, Version]:
    """Check whether haiku.rag is current."""

    # Lazy import to avoid pulling httpx (and its deps) on module import
    import httpx

    async with httpx.AsyncClient() as client:
        running_version = parse(metadata.version("haiku.rag-slim"))
        try:
            response = await client.get("https://pypi.org/pypi/haiku.rag/json")
            data = response.json()
            pypi_version = parse(data["info"]["version"])
        except Exception:  # pragma: no cover
            # If no network connection, do not raise alarms.
            pypi_version = running_version
    return running_version >= pypi_version, running_version, pypi_version

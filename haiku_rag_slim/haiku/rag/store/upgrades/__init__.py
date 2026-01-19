import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from packaging.version import Version, parse

if TYPE_CHECKING:
    from haiku.rag.store.engine import Store

logger = logging.getLogger(__name__)


@dataclass
class Upgrade:
    """Represents a database upgrade step."""

    version: str
    apply: Callable[["Store"], None]
    description: str = ""


# Registry of upgrade steps (ordered by version)
upgrades: list[Upgrade] = []


def get_pending_upgrades(from_version: str) -> list[Upgrade]:
    """Get pending upgrades from the given version.

    Returns:
        List of Upgrade objects where from_version < upgrade.version,
        sorted by version in ascending order.
    """
    v_from: Version = parse(from_version)
    sorted_steps = sorted(upgrades, key=lambda u: parse(u.version))
    return [s for s in sorted_steps if v_from < parse(s.version)]


def run_pending_upgrades(store: "Store", from_version: str) -> list[str]:
    """Run upgrades where from_version < step.version.

    Returns:
        List of descriptions of applied upgrades.
    """
    applicable = get_pending_upgrades(from_version)

    if applicable:
        logger.info("%d upgrade step(s) pending", len(applicable))

    applied: list[str] = []

    # Apply in ascending order
    for idx, step in enumerate(applicable, start=1):
        logger.info(
            "Applying upgrade %s: %s (%d/%d)",
            step.version,
            step.description or "",
            idx,
            len(applicable),
        )
        step.apply(store)
        logger.info("Completed upgrade %s", step.version)
        applied.append(
            f"{step.version}: {step.description}" if step.description else step.version
        )

    return applied


# Import upgrade modules AFTER Upgrade class is defined to avoid circular imports
# ruff: noqa: E402, I001
from haiku.rag.store.upgrades.v0_20_0 import (
    upgrade_add_docling_document as upgrade_0_20_0_docling,
)
from haiku.rag.store.upgrades.v0_23_1 import (
    upgrade_contextualize_chunks as upgrade_0_23_1_contextualize,
)
from haiku.rag.store.upgrades.v0_25_0 import (
    upgrade_compress_docling_document as upgrade_0_25_0_compress,
)

upgrades.append(upgrade_0_20_0_docling)
upgrades.append(upgrade_0_23_1_contextualize)
upgrades.append(upgrade_0_25_0_compress)

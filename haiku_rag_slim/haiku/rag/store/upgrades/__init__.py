import logging
from collections.abc import Callable
from dataclasses import dataclass

from packaging.version import Version, parse

from haiku.rag.store.engine import Store

logger = logging.getLogger(__name__)


@dataclass
class Upgrade:
    """Represents a database upgrade step."""

    version: str
    apply: Callable[[Store], None]
    description: str = ""


# Registry of upgrade steps (ordered by version)
upgrades: list[Upgrade] = []


def run_pending_upgrades(store: Store, from_version: str, to_version: str) -> None:
    """Run upgrades where from_version < step.version <= to_version."""
    v_from: Version = parse(from_version)
    v_to: Version = parse(to_version)

    # Ensure that tests/development run available code upgrades even if the
    # installed package version hasn't been bumped to include them yet.
    if upgrades:
        highest_step_version: Version = max(parse(u.version) for u in upgrades)
        if highest_step_version > v_to:
            v_to = highest_step_version

    # Determine applicable steps
    sorted_steps = sorted(upgrades, key=lambda u: parse(u.version))
    applicable = [s for s in sorted_steps if v_from < parse(s.version) <= v_to]
    if applicable:
        logger.info("%d upgrade step(s) pending", len(applicable))

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

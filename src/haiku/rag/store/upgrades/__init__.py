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


from .v0_9_3 import upgrade_fts_phrase as upgrade_0_9_3_fts  # noqa: E402
from .v0_9_3 import upgrade_order as upgrade_0_9_3_order  # noqa: E402
from .v0_10_1 import upgrade_add_title as upgrade_0_10_1_add_title  # noqa: E402

upgrades.append(upgrade_0_9_3_order)
upgrades.append(upgrade_0_9_3_fts)
upgrades.append(upgrade_0_10_1_add_title)

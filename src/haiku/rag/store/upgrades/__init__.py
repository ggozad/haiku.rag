from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from packaging.version import Version, parse

from haiku.rag.store.engine import Store


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

    # Ensure upgrades are applied in ascending version order
    for step in sorted(upgrades, key=lambda u: parse(u.version)):
        v_step = parse(step.version)
        if v_from < v_step <= v_to:
            step.apply(store)


# Import concrete upgrade modules (module names cannot start with a digit)
from .v0_9_3 import upgrade_order as upgrade_0_9_3_order  # noqa: E402

upgrades.append(upgrade_0_9_3_order)

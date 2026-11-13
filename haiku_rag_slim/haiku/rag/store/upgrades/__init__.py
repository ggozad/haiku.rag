import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from packaging.version import Version, parse

if TYPE_CHECKING:
    from haiku.rag.store.engine import Store

logger = logging.getLogger(__name__)


@dataclass
class Upgrade:
    """Represents a database upgrade step."""

    version: str
    apply: Callable[["Store"], Coroutine[Any, Any, None]]
    description: str = ""


# Registry of upgrade steps (ordered by version)
upgrades: list[Upgrade] = []

# A database recorded below this version needs a step this release does not carry.
OLDEST_MIGRATABLE_VERSION = "0.64.0"
# The last release carrying every upgrade step.
FULL_CHAIN_RELEASE = "0.111.1"


def migratable(version: str) -> bool:
    """Whether the upgrade steps this release carries can migrate `version`."""
    return parse(version) >= parse(OLDEST_MIGRATABLE_VERSION)


def get_pending_upgrades(from_version: str) -> list[Upgrade]:
    """Get pending upgrades from the given version.

    Returns:
        List of Upgrade objects where from_version < upgrade.version,
        sorted by version in ascending order.
    """
    v_from: Version = parse(from_version)
    sorted_steps = sorted(upgrades, key=lambda u: parse(u.version))
    return [s for s in sorted_steps if v_from < parse(s.version)]


def latest_schema_version() -> str:
    """The version of the newest upgrade step: the schema this release writes."""
    return max((u.version for u in upgrades), key=parse)


async def run_pending_upgrades(store: "Store", from_version: str) -> list[str]:
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
        await step.apply(store)
        logger.info("Completed upgrade %s", step.version)
        applied.append(
            f"{step.version}: {step.description}" if step.description else step.version
        )

    return applied


# Import upgrade modules AFTER Upgrade class is defined to avoid circular imports
# ruff: noqa: E402, I001
from haiku.rag.store.upgrades.v0_75_0 import (
    upgrade_index_hot_lookup_keys as upgrade_0_75_0_index_hot_lookup_keys,
)
from haiku.rag.store.upgrades.v0_89_0 import (
    upgrade_record_embedder_only as upgrade_0_89_0_record_embedder_only,
)
from haiku.rag.store.upgrades.v0_91_0 import (
    upgrade_fts_format_v2 as upgrade_0_91_0_fts_format_v2,
)

upgrades.append(upgrade_0_75_0_index_hot_lookup_keys)
upgrades.append(upgrade_0_89_0_record_embedder_only)
upgrades.append(upgrade_0_91_0_fts_format_v2)

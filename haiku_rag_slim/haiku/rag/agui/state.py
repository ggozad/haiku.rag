"""Generic AG-UI state utilities for any Pydantic BaseModel."""

from typing import Any

from pydantic import BaseModel


def compute_state_delta(
    old_state: BaseModel, new_state: BaseModel
) -> list[dict[str, Any]]:
    """Compute JSON Patch (RFC 6902) operations from old state to new state.

    Args:
        old_state: Previous state (any Pydantic BaseModel)
        new_state: Current state (same type as old_state)

    Returns:
        List of JSON Patch operations
    """
    operations: list[dict[str, Any]] = []

    # Convert states to dicts for comparison
    old_dict = old_state.model_dump()
    new_dict = new_state.model_dump()

    # Compare each field and generate patches
    for key, new_value in new_dict.items():
        old_value = old_dict.get(key)

        if old_value != new_value:
            # Simple replace operation
            operations.append({"op": "replace", "path": f"/{key}", "value": new_value})

    return operations

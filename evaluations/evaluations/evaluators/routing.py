from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

COUNTS = ("n_search_calls", "n_sandbox_search_calls", "n_executions", "n_requests")
"""Run counts scored beside the routing scores, so a pairing reports them."""


def _names(call: Any) -> set[str] | None:
    """The collections a call named, or None for a value that is not a list of names."""
    if isinstance(call, list) and all(isinstance(name, str) for name in call):
        return set(call)
    return None


def _invalid(call: Any, collections: set[str]) -> bool:
    """Whether a sent selection read nothing: not a list of names, or a name outside the run."""
    named = _names(call)
    return named is None or not named <= collections


def _selections(calls: list[Any], collections: set[str]) -> list[set[str]]:
    """What each search read: the whole run for an omitted `sources`, nothing
    for a selection that is not a list of names or names a collection outside
    the run."""
    selections: list[set[str]] = []
    for call in calls:
        if call is None:
            selections.append(set(collections))
            continue
        named = _names(call)
        selections.append(
            named if named is not None and named <= collections else set()
        )
    return selections


@dataclass
class CollectionRoutingEvaluator(Evaluator):
    """How a run's searches selected among the collections it covers.

    Reads ``search_sources`` and ``searched_uris`` from the attributes, and
    ``collections``, ``expected_sources`` and ``relevant_uris`` from the case
    metadata. A case without ``collections`` is not a routing case and gets no
    score.
    """

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float | int]:
        metadata = ctx.metadata or {}
        if "collections" not in metadata:
            return {}
        collections = set(metadata["collections"])
        expected = set(metadata.get("expected_sources", []))
        calls = list(ctx.attributes.get("search_sources") or [])
        selections = _selections(calls, collections)
        searched: set[str] = set().union(*selections)
        first = selections[0] if selections else set()
        first_covers = bool(selections) and expected <= first
        scores: dict[str, float | int] = {
            "first_search_covers": float(first_covers),
            "first_search_exact": float(bool(selections) and first == expected),
            "sources_recall": len(expected & searched) / len(expected)
            if expected
            else 0.0,
            "effective_collections": sum(len(s) for s in selections),
            "n_broad_searches": sum(1 for s in selections if s == collections),
            "n_invalid_selections": sum(
                1 for c in calls if c is not None and _invalid(c, collections)
            ),
            "n_broadenings": sum(1 for a, b in pairwise(selections) if a < b),
        }
        if not first_covers:
            scores["recovered_after_miss"] = float(expected <= searched)
        relevant = set(metadata.get("relevant_uris", []))
        if relevant:
            found = set(ctx.attributes.get("searched_uris") or [])
            scores["searched_recall"] = len(relevant & found) / len(relevant)
        for name in COUNTS:
            if name in ctx.attributes:
                scores[name] = int(ctx.attributes[name])
        return scores

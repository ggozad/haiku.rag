import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from evaluations.evaluators.number_match import _answer_segment

# A single numeric literal: optional sign, optional $, digits with thousands
# separators, optional decimal, optional trailing percent. Scale words
# (million/billion) are deliberately NOT expanded — T² gold answers are bare
# numbers, so expanding would mis-scale (e.g. "688 million" must stay 688).
_NUM_RE = re.compile(r"[-−]?\$?\s*\d[\d,]*(?:\.\d+)?\s*%?")


def _format_number(value: float) -> str:
    """Render without a trailing ``.0`` for integers; plain decimal otherwise."""
    if value == int(value):
        return str(int(value))
    return repr(value)


def extract_prediction(output: str | None) -> str:
    """Pull the primary numeric answer from a skill output, for submission.

    Restricts to a declared ``ANSWER:`` line when present (via ``_answer_segment``)
    so reasoning numbers don't leak. Strips ``$`` and thousands separators,
    converts a trailing ``%`` to a fraction (T² gold stores percentages as
    decimals), and normalizes the unicode minus. Returns ``""`` for empty/no-number
    outputs (nulls) — the leaderboard counts those as wrong.

    NOTE: the exact normalization the leaderboard's NM applies is unconfirmed;
    validate against their scorer before a final submission.
    """
    if not output:
        return ""
    match = _NUM_RE.search(_answer_segment(output))
    if match is None:
        return ""
    token = match.group(0).replace("$", "").replace(",", "").replace(" ", "")
    token = token.replace("−", "-")
    if token.endswith("%"):
        return _format_number(float(token[:-1]) / 100)
    return _format_number(float(token))


def build_submission_rows(
    predictions: Iterable[Mapping[str, Any]],
    retrieval_by_question: Mapping[str, Sequence[str]],
    subset: str,
    topk: int = 3,
) -> list[dict[str, Any]]:
    """Assemble T² leaderboard submission rows.

    Args:
        predictions: rows with ``id``, ``question`` and ``output`` (the QA run).
        retrieval_by_question: question text -> ranked retrieved context ids.
        subset: dataset subset name (e.g. ``"FinQA"``).
        topk: how many ranked context ids to include. ``context_id`` is a single
            string when ``topk == 1``, else a list of up to ``topk`` ids.

    Returns one dict per prediction: ``{id, subset, context_id, prediction}``.
    """
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        ranked = list(retrieval_by_question.get(pred["question"], []))[:topk]
        context_id: Any = (ranked[0] if ranked else None) if topk == 1 else ranked
        rows.append(
            {
                "id": pred["id"],
                "subset": subset,
                "context_id": context_id,
                "prediction": extract_prediction(pred.get("output")),
            }
        )
    return rows

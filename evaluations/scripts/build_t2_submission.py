#!/usr/bin/env python
"""Build a T²-RAGBench leaderboard submission file (JSONL) for one subset.

Joins QA predictions with retrieval rankings by question text and emits one
object per question: ``{id, subset, context_id, prediction}`` — the format the
leaderboard expects (NM is scored on ``prediction``, MRR@3 on ``context_id``).
See https://t2ragbench.demo.hcds.uni-hamburg.de/submission.html.

Predictions come from a QA-run CSV (the merged master export: needs columns
``id``, ``question``, ``output``). Retrieval rankings come from a Logfire
retrieval trace (case spans store the ranked retrieved context ids in
``output`` and the question in ``inputs``) or, with --retrieval-csv, a CSV with
``question`` and a JSON-list ``output``.

Logfire access needs LOGFIRE_READ_TOKEN in the environment (EU project).

Example:
    LOGFIRE_READ_TOKEN=... uv run python scripts/build_t2_submission.py \
        --predictions ../t2_finqa_qwen3.6_019e982d.csv \
        --retrieval-trace 019e9296e64d38b33f3592beff6660a9 \
        --subset FinQA --topk 3 --out finqa_submission.jsonl
"""

import argparse
import csv
import datetime
import json
import os
import sys

from evaluations.submission import build_submission_rows


def _load_predictions(path: str) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _retrieval_from_csv(path: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["question"]] = json.loads(row["output"])
    return out


def _retrieval_from_trace(trace_id: str) -> dict[str, list[str]]:
    from logfire.experimental.query_client import LogfireQueryClient

    token = os.environ.get("LOGFIRE_READ_TOKEN")
    if not token:
        sys.exit("LOGFIRE_READ_TOKEN not set (needed to read the retrieval trace).")
    client = LogfireQueryClient(
        read_token=token, base_url="https://logfire-eu.pydantic.dev"
    )
    rows = client.query_json_rows(
        sql=(
            "SELECT attributes->>'inputs' AS question, attributes->>'output' AS ranked "
            f"FROM records WHERE trace_id = '{trace_id}' AND span_name LIKE 'case:%'"
        ),
        min_timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        limit=10000,
    )["rows"]
    return {r["question"]: json.loads(r["ranked"]) for r in rows if r.get("ranked")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="QA-run CSV.")
    parser.add_argument("--retrieval-trace", help="Logfire retrieval trace id.")
    parser.add_argument("--retrieval-csv", help="Retrieval CSV (question, output).")
    parser.add_argument("--subset", required=True, help="Subset name, e.g. FinQA.")
    parser.add_argument("--topk", type=int, default=3, help="Ranked context ids.")
    parser.add_argument("--out", required=True, help="Output .jsonl path.")
    args = parser.parse_args()

    if bool(args.retrieval_trace) == bool(args.retrieval_csv):
        sys.exit("Pass exactly one of --retrieval-trace or --retrieval-csv.")

    predictions = _load_predictions(args.predictions)
    retrieval = (
        _retrieval_from_csv(args.retrieval_csv)
        if args.retrieval_csv
        else _retrieval_from_trace(args.retrieval_trace)
    )

    rows = build_submission_rows(
        predictions, retrieval, subset=args.subset, topk=args.topk
    )
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    answered = sum(1 for r in rows if r["prediction"])
    matched = sum(1 for r in rows if r["context_id"])
    print(
        f"wrote {args.out}: {len(rows)} rows "
        f"({answered} with a prediction, {matched} with a retrieved context_id)"
    )


if __name__ == "__main__":
    main()

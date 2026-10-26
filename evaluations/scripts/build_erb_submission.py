#!/usr/bin/env python
"""Build an EnterpriseRAG-Bench `answers.jsonl` from a per-case results file.

Writes one object per case, `{question_id, answer, document_ids}`, the input of
Onyx's scorer (`src.scripts.answer_evaluation.metrics_based_eval` in
github.com/onyx-dot-app/EnterpriseRAG-Bench). `document_ids` are the documents
the answer cited.

With --uuid-index (their `generated_data/uuid_index.json`), exits 1 when a cited
id is missing from it: their scorer stops at an interactive prompt for one.

Example:
    uv run python scripts/build_erb_submission.py \
        ~/.local/share/haiku.rag/evaluations/results/<name>.<trace>.jsonl \
        answers.jsonl --uuid-index ../EnterpriseRAG-Bench/generated_data/uuid_index.json
"""

import argparse
import json
import sys
from pathlib import Path

from evaluations.submission import enterprise_rag_answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="Per-case results jsonl.")
    parser.add_argument("out", type=Path, help="answers.jsonl to write.")
    parser.add_argument("--uuid-index", type=Path, help="Their uuid_index.json.")
    args = parser.parse_args()

    results = [
        json.loads(line) for line in args.results.read_text().splitlines() if line
    ]
    answers = enterprise_rag_answers(results)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in answers))

    empty = sum(1 for row in answers if not row["answer"] and not row["document_ids"])
    print(f"{len(answers)} rows, {empty} with neither answer nor documents")
    if args.uuid_index:
        known = json.loads(args.uuid_index.read_text())
        unknown = sorted(
            {doc for row in answers for doc in row["document_ids"]} - known.keys()
        )
        if unknown:
            sys.exit(f"{len(unknown)} cited ids missing from the index: {unknown[:20]}")


if __name__ == "__main__":
    main()

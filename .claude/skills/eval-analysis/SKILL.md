---
name: eval-analysis
description: Read and compare haiku.rag evaluation results. Use when asked what an arm scored, whether a branch or config is better than another, to pair two runs, to interpret accuracy, cite rate or cited_map, or to query Logfire for per-case evidence without the accessor, row-cap and window traps.
---

# Analysing evaluation results

The registry is the record of what ran. Logfire holds the per-case data. This
skill is how to turn the two into a claim that survives a second reader.

## 1. The standard table

```bash
evaluations arms pair <arm A> <arm B>
```

Treated and baseline come from the recorded comparator, never from argument
order or from the order of a result listing. The command refuses an arm that
is not a completed valid run, a commit that either row leaves unrecorded, two
datasets, two kinds, a pairing key that is NULL on either side, and a pair
whose rows do not show the differences the arm file names. The paired tests
are over per-case verdicts, so only QA arms pair: a retrieval or build arm is
refused, and read its numbers from the rows instead.
If it refuses, the pair is not ready; do not compute it by hand.

Per-case rows come from the run's result file under the evaluations data
directory when it exists, and from Logfire otherwise.

Per arm: cases, accuracy over judged cases, floor over all cases, cite rate
over all cases, mean `cited_map` over the cases that carry one, aborts,
unjudged. The table prints those denominators under it, and the registry column
is `cite_rate_all_cases`. Published tables have quoted a cite rate over scored
cases instead, so never compare one against the other without converting:
scored rate times scored cases, over all cases. For the pair: discordant counts with the
exact McNemar p-value, the `cited_map` sign test, the smallest |b - c| the
test rejects at p < 0.05 with that many discordant pairs, and the recorded
decision rule. Quote the table whole. The resolution line is the power
statement: a gate written for the full set cannot be evaluated on a prefix,
and saying so is part of the report.

## 2. Reading a pair

- Cases join on the dataset's pairing key, which the harness records in the
  experiment metadata. Never choose the column yourself. A join on a key the
  cases do not carry is NULL on both sides and returns nothing or a cross
  product that looks like a plausible table.
- Read the first line before the p-value. A join that matched a fraction of
  the cases prints a normal-looking table, so compare the paired count with
  each arm's cases and with the case selection the arm file names. The
  one-sided counts beside it say which arm the missing cases came from.
- Read the sign test's tie count the same way. A sign test drops every tie, and
  scores on a coarse lattice tie often, so the surviving pairs can be a small
  fraction of the cases. Forty surviving pairs print exactly like eight
  hundred.
- Check that the paired count equals the cases you expected. A join that
  silently matches a fraction produces a verdict that looks normal.
- A pair is unreadable without a null pair beside it. Identical code takes a
  different trajectory on most cases while flipping few verdicts, so
  `n_searches` or `n_executions` deltas between two arms are noise unless the
  null pair says otherwise. One null replicate bounds nothing; several are
  needed before a delta is inside or outside the noise.
- Do not back-derive a flip rate from the treatment and then use it to excuse
  the treatment.
- At a handful of discordant pairs the exact test has no power. Open every
  discordant case and find its mechanism. Causal evidence on three of three
  beats a p-value over a hundred.
- Timing: medians and interquartile ranges, never means, and always against a
  null pair. A mean delta is two outliers.
- Cite rate is confounded by task success. Measure it among correct answers,
  split by whether the case executed code. `citation_status` and `cited_map`
  answer different questions; do not conflate them.
- Ratios take numerator and denominator from one cohort, and the report names
  it.

## 3. Eras and corpora

A corpus rebuild ends an era. Numbers from before and after do not compare,
and a trace cannot tell you which corpus it used unless its metadata carries
the fingerprint. Every registry row carries the database path, document and
chunk counts, the stored embedder and `db_written_at`, the newest table
version time. Counts alone do not separate two states of one corpus: a
content fix at a constant document count, or a schema migration that rewrites
no rows, leave them identical while `db_written_at` moves. `db_version` is the
schema marker and says nothing about content. State them with any number you
quote, and never quote a benchmark number without naming the corpus it came
from.
`reference.md` in the `haiku.rag-evaluations-data` repository lists the
current eras.

## 4. Logfire, without the traps

The registry and the result files are the record; nothing in the standard
table needs Logfire. Logfire is where you read a transcript: what the model
saw, what a tool returned, why a case died. Project `haiku`, service `evals`.
The `debug-evals` skill has the queries; these are the rules that make them
true.

- Read a value as text with `attributes->>'key'`. The subtree form
  `attributes->'key'` returns NULL on some span attributes with no error, so
  a phrase count over thousands of spans comes back a clean zero and looks
  like a finding. Before reporting any zero or near-zero from a text scan, run
  a control on the same expression that must match, and check
  `max(length(...))` is not NULL.
- Per-case data hangs off the case span through a parent chain: `case:` then
  `execute {task}` then `invoke_agent agent` then `execute_tool *`. Tool
  results are in `gen_ai.tool.call.result`. Model output is in
  `gen_ai.output.messages` and never contains tool results.
- Exclude the sibling `invoke_agent judge_input_output_expected` subtree from
  any evidence search. The judge is given the gold answer.
- Queries return at most 100 rows and do not say so. Aggregate server side,
  one arm per query, or page with `LIMIT 100 OFFSET n` and check the total
  against `count(*)`.
- The time window applies to every table scan in the query, including a
  subquery that supplies ids. A window tight enough for one arm silently
  returns no ids for another. Scope by full 32-character trace id.
- Count `count(*)` against `count(distinct case_name)` per trace. A
  duplicated export inflates n by one and is invisible otherwise.
- Large image payloads can drop case spans without an error. For any vision
  arm, compare the distinct case count with the dataset size before trusting
  an aggregate.
- Assign arms from the recorded run name and metadata, never from position in
  a result set.

## 5. Claims

- Before asserting a mechanism, read the code path end to end and name one
  observation that would disprove the claim. If none can be named, the claim
  is not ready.
- When evidence points one way, measure the other direction before
  withdrawing a concern.
- A claim about a population is measured on the population.
- Every mechanism claim is verified by a second session, against the code
  path or the population, before it reaches Yiorgis.
- Retract in the durable record: the registry note, the memory note, the plan.

## Amending this skill

Same procedure as `eval-launch`: a new trap or convention is written here in
the session that learns it, facts go to `reference.md` in the data repository
or to the registry, every amendment is reviewed by a second session or by
Yiorgis as a commit to this repository and dated below, and an amendment that
adds a must-know item extends the acceptance test.
`evaluations/tests/test_skills.py` refuses a commit sha, trace id or run date
above this section.

## Change log

- 2026-09-17: created from the operator plan, sections 1.6, 1.7, 2 and 3.4.

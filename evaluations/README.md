# haiku.rag - Evaluations

Internal benchmarking and evaluation scripts for haiku.rag.

This package is not published to PyPI and is only used for development and testing purposes.

## Overview

Contains evaluation scripts for benchmarking RAG retrieval and QA performance. Available datasets:

- HotpotQA (`hotpotqa`) — multi-hop QA over Wikipedia paragraphs (distractor validation split, 7,405 questions, two gold documents per question)
- MTRAG ClapNQ (`mtrag_clapnq`, `mtrag_clapnq_rewrite`) — IBM's multi-turn RAG benchmark, ClapNQ (Wikipedia) domain: 183,408 passages, 208 retrieval queries with binary qrels, 224 generation tasks. The base key retrieves with the raw last user turn; the `_rewrite` variant uses the human standalone rewrites (both share one database). Retrieval reports Recall@5/@10, nDCG@5/@10, and MAP against IBM's published setup. QA replays each task's reference conversation prefix as message history and answers the final turn; the judge sees the conversation as a transcript, citation MAP is scored only on turns with gold passages, and refusal precision/recall is reported against the answerability labels. Generation scores are internal (our judge and rubric), not comparable with IBM's published generation numbers. The `mtrag_clapnq_live` key replays whole conversations (one case per conversation, `--limit` counts conversations) through a single capability session, carrying the model's own answers and tool history across turns; it reports the same outcomes per turn plus micro (per-turn) and macro (per-conversation) aggregates.
- FRAMES (`frames`) — multi-hop QA (822 questions, 2-23 gold Wikipedia articles per question; 2 of the original 824 questions are excluded because a linked article has been deleted from Wikipedia). The corpus is the union of the 2,521 linked articles, fetched from the Wikipedia REST API at current revision (revision id and fetch date recorded in the article cache) with navigation chrome stripped. There is no official FRAMES evaluation setup; numbers here correspond to the paper's multi-step retrieval setting (fixed corpus, agentic retrieval, judged accuracy) and are not comparable to its closed-book, oracle-prompt, or web-search settings. Answers were authored against ~2024 revisions and may have drifted with article content.
- OpenRAG Bench, two variants:
  - `orb_text` — text embedder (`qwen3-embedding:4b`, 2560-dim) with VLM picture descriptions baked into chunk content at ingest. Use for text-only retrieval/QA against figure-rich corpora.
  - `orb_multimodal` — multimodal embedder (`qwen3-vl-embedding-8b`, 4096-dim) with picture vectors in the same space as text. Use for cross-modal retrieval (text-as-query → figure hits, image-as-query) and vision QA where the figure itself is the answer.

## Usage

After installing the package, you can run evaluations using the `evaluations` command:

```bash
# Run retrieval + QA benchmarks
evaluations run hotpotqa
evaluations run orb_text

# Use a custom config file
evaluations run hotpotqa --config /path/to/haiku.rag.yaml

# Override the database path
evaluations run hotpotqa --db /path/to/custom.lancedb

# Skip database population and run only benchmarks
evaluations run hotpotqa --skip-db

# Skip specific benchmarks
evaluations run hotpotqa --skip-retrieval
evaluations run hotpotqa --skip-qa

# Limit the number of test cases
evaluations run hotpotqa --limit 100
```

Population prints a line every 50 ingested documents and one at the end:
documents seen and ingested, elapsed time, documents per minute over the
whole run, and the ETA at that rate. Documents skipped on resume count as
seen, not ingested. The progress bar renders only on a terminal, so a
redirected log carries these lines and nothing else. Python buffers redirected
output, so run with `PYTHONUNBUFFERED=1` to read them as they appear; the
queue sets it.

### Choosing the capability model

`evaluations run` benchmarks the RAG capability end to end:

```bash
evaluations run hotpotqa
```

The capability runs on `qa.model` and the judge on `evaluations.judge`, both
from the config, so a run's models travel with the file its hash covers. A
citation retrieval metric (`cited_map`) is computed alongside QA accuracy from
the URIs the capability registered via the `cite` tool.

### Declaring and checking an arm

An arm file declares one run: the dataset, the checkout and its pinned commit,
the config, the database, the case selection and the flags passed through to
`evaluations run`. A paired arm also names its comparator, the differences the
operator expects between the two, and the decision rule.

```yaml
name: frames-main-150
dataset: frames
worktree: ~/wt/frames-main
sha: 0123456789ab
config: configs/frames.yaml
db: ~/.local/share/haiku.rag/evaluations/dbs/frames.lancedb
limit: 150
flags: [--skip-db, --skip-retrieval]
comparator: frames-branch-150.yaml
differences: [sha]
decision_rule: McNemar exact on answer_equivalent, two-sided, p < 0.05 fails
```

Relative paths resolve against the arm file. `flags` may not carry `--config`,
`--name`, `--db`, `--limit` or `--filter-ids`: the fields above pin them, and a
flag would run something else than the registry records. `evaluations preflight
ARM` prints one line per check and exits 1 when any fails: the checkout is at
the pinned commit with no uncommitted changes to tracked files, its `.env`
carries `LOGFIRE_TOKEN` unless the arm passes `--no-telemetry`, the config
validates, every local database the run reads exists and stores the config's
embedder (a configured location carrying a scheme is a URI, which the preflight
names and does not open), the filter files exist, and every difference from the comparator (arm fields, flags by
option, config keys by dotted path) is named in `differences`. A named
difference that does not differ fails too. An arm whose config fills
`lancedb.databases` evaluates that set: it carries `--skip-db` and no `db`,
which is what the run itself requires.

### The registry of arms

Every arm gets one row in `registry.sqlite` under the evaluations data
directory (`--registry PATH` points elsewhere). `evaluations preflight ARM
--register` writes the launch row when every check passes: dataset, commit,
config path and hash, database path and fingerprint, the capability model and
the endpoint it opens, judge, reranker, embedder, case selection, comparator,
decision rule and operator. A run that skips QA records no capability and no
judge. Completion fills the trace id, cases, accuracy, cite rate, mean
`cited_map`, aborts and wall time and sets the status to `valid`. A void arm
keeps its row and carries the reason, and its numbers are never paired.
`evaluations arms complete NAME` fills those fields from Logfire, finding the
trace by run name within the launch window or by `--trace ID`. With no trace
the arm is void with reason `no telemetry`.

```bash
evaluations arms list [--dataset frames] [--db PATH] [--status void]
evaluations arms show frames-main-150
evaluations arms void frames-main-150 --reason "wrong target"
evaluations arms export arms.jsonl     # sorted keys, one arm per line, diffable
evaluations arms import arms.jsonl     # upserts by name
```

### Running a queue of arms

`evaluations queue a.yaml b.yaml` runs the arm files in order. For each arm it
runs the preflight, runs the smoke set named by `smoke_ids` under the name
`<name>-smoke` and requires it to exit cleanly with at least one case,
registers the launch,
runs the arm from its worktree with output appended to `logs/<name>.log` under
the evaluations data directory, kills the run at `deadline_hours`, and
completes the registry row from the trace. A failed step ends that arm and the
queue continues; an arm that fails after it was registered keeps its launched
row, to complete by hand. When the arm's worktree does not exist and `--repo` names a
checkout, the queue provisions it at the pinned sha with `uv sync` and copies
`--env` into it. `--detach NAME` starts the queue inside a detached tmux
session, so it survives a lost ssh connection.

```bash
evaluations queue arms/frames-main.yaml arms/frames-branch.yaml \
  --repo ~/dev/haiku.rag --env ~/dev/haiku.rag/.env --detach night
```

### Per-case result files

`evaluations run` writes one JSON line per case to
`<name>.<trace>.jsonl` under `results/` in the evaluations data directory
(`--results DIR` or `HAIKU_RAG_EVAL_RESULTS` point elsewhere): case name,
pairing key, verdict, whether the case cited anything, `cited_map`, whether
it aborted, the trace id, the answer, the judge's reason, the run's per-case
attributes and the task duration. A run without a trace writes
`<name>.notrace-<time>.jsonl`. `arms pair`, `arms complete` and the queue's
smoke check and completion read this file when it exists and fall back to
Logfire, so the registry and the result files are the record and Logfire is
where you read a transcript. Live conversation runs write no file.

### Pairing two arms

`evaluations arms pair A B` prints the standard table for two registered
arms. Treated and baseline are read from the recorded comparator, not from
argument order. The command refuses an arm that is not a completed valid run,
a commit that either row leaves unrecorded, two datasets, a pairing key that is
NULL on either side, and a pair whose rows do not show the differences the arm
file names.

Cases join on the dataset's `pair_key`: `question_id` for FRAMES and
HotpotQA, `query_id` for ORB, `id` for T2, `task_id` and `conversation_id`
for MTRAG. Per-case outcomes come from each run's result file, and from
Logfire through the read key in `~/.logfire-read-key` or `LOGFIRE_READ_KEY`
when there is none. The table shows per arm the
cases, accuracy over judged cases, floor over all cases, cite rate, mean
`cited_map`, aborts and unjudged cases. For the pair it shows the discordant
counts with the exact McNemar p-value, the `cited_map` sign test, the
smallest |b - c| the test rejects at p < 0.05 with that many discordant
pairs, and the recorded decision rule.

### Debugging runs in Logfire

With `LOGFIRE_TOKEN` set, runs ship spans under `service_name = 'evals'`. The
`debug-evals` skill in `.claude/skills/` turns these into ready-made Logfire
queries (recent runs, per-case pass rate and `cited_map`, failing and slowest
cases) for use from Claude Code.

A run refuses to start when `LOGFIRE_TOKEN` is not set. `--no-telemetry` runs
without it and leaves Logfire unconfigured, so the run sends nothing whatever
the environment holds and its result file is the record. A live dataset and
`--skip-qa` write no result file, so the preflight refuses the flag for them.
Every run prints its git commit and the SHA-256 of its resolved config at start
and records them in the experiment metadata (`git_sha`, `git_dirty`,
`config_hash`) together with the database it read (`db_path`, `db_documents`,
`db_chunks`, `db_embedder_provider`, `db_embedder_model`, `db_embedder_dim`,
`db_version`).

### Pre-built Databases

Download pre-built evaluation databases from HuggingFace:

```bash
evaluations download hotpotqa
evaluations download all
evaluations download hotpotqa --force
```

Upload databases (maintainer only):

```bash
evaluations upload hotpotqa
evaluations upload all
```

## Database Storage

By default, evaluation databases are stored in the haiku.rag data directory:
- **Linux**: `~/.local/share/haiku.rag/evaluations/dbs/`
- **macOS**: `~/Library/Application Support/haiku.rag/evaluations/dbs/`
- **Windows**: `C:/Users/<USER>/AppData/Roaming/haiku.rag/evaluations/dbs/`

You can override this with the `--db` option.

### Evaluating over Multiple Databases

With [`lancedb.databases`](https://ggozad.github.io/haiku.rag/configuration/storage/#multiple-databases) configured, `evaluations run <dataset> --skip-db` benchmarks the full set. Retrieval, QA, and live conversations preserve the database name on results and citations. A configured set of one follows the same path and retains its name.

Population writes one database and therefore requires `--db`:

```bash
evaluations run hotpotqa --db /path/to/one.lancedb   # populate, then benchmark
evaluations run hotpotqa --skip-db                   # benchmark the configured set
```

`--db` overrides the configured set for both population and benchmarks.

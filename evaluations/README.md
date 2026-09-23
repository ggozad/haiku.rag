# haiku.rag - Evaluations

Benchmarking for haiku.rag: retrieval, QA accuracy and citation retrieval, scored end to end through the RAG capability. Not published to PyPI. Methodology and current results are on the [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/) page.

## Datasets

- `hotpotqa`: HotpotQA multi-hop QA over Wikipedia paragraphs, distractor validation split, 7,405 questions with two gold documents each.
- `frames`: FRAMES multi-hop QA, 822 of the 824 questions (two link a deleted article) over a fixed corpus of the 2,500 linked articles, fetched from the Wikipedia REST API at current revision with navigation removed. The revision id and fetch date are in the article cache. Numbers correspond to the paper's multi-step retrieval setting, not its closed-book, oracle-prompt or web-search settings. Answers were written against 2024 revisions and may have drifted.
- `orb_text`, `orb_multimodal`, `orb_multimodal_nemotron`: OpenRAG Bench. `orb_text` uses `qwen3-embedding:4b` (2560) with VLM picture descriptions in the chunk text, for text-only retrieval over figure-rich papers. `orb_multimodal` uses `qwen3-vl-embedding-8b` (4096) and `orb_multimodal_nemotron` `nvidia/llama-nemotron-embed-vl-1b-v2`, both with pictures in the text vector space, for cross-modal retrieval and vision QA.
- `t2_finqa`, `t2_tatdqa`: T²-RAGBench financial-report QA, scored by numeric match instead of an LLM judge.
- `mtrag_clapnq`, `mtrag_clapnq_rewrite`, `mtrag_clapnq_live`, `mtrag_clapnq_live_uncompacted`: IBM's MTRAG multi-turn benchmark, ClapNQ domain (183,408 passages, 208 retrieval queries with binary qrels, 224 generation tasks), all four on one database.
  - `mtrag_clapnq` retrieves with the last user turn as written, and answers each task after replaying its reference conversation prefix as message history. The judge sees the conversation as a transcript. Citation MAP is scored only on turns with gold passages, and refusal precision and recall against the answerability labels.
  - `mtrag_clapnq_rewrite` retrieves with the human standalone rewrites.
  - `mtrag_clapnq_live` replays whole conversations through one capability session, carrying the model's own answers and tool history, and reports per-turn outcomes with micro (per-turn) and macro (per-conversation) aggregates. `--limit` counts conversations.
  - `mtrag_clapnq_live_uncompacted` is the same replay without evidence compaction.
  - Retrieval reports Recall@5/@10, nDCG@5/@10 and MAP, comparable with IBM's published setup. Generation scores use our judge and rubric and are not comparable with IBM's.

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

# Restrict every search to a subset of the database
evaluations run orb_text --skip-db --filter "uri LIKE '2407%'"

# Re-run only the QA cases listed in a file, one id per line
evaluations run hotpotqa --skip-db --filter-ids failed.txt

# Only the queries that need image understanding
evaluations run orb_multimodal --multimodal-only

# Name the run, and vacuum every N documents while populating (default 100)
evaluations run hotpotqa --name my-arm --vacuum-interval 500
```

### Choosing the capability model

`evaluations run` benchmarks the RAG capability end to end:

```bash
evaluations run hotpotqa
```

The capability runs on `qa.model` and the judge on `evaluations.judge`, both
from the config. A citation retrieval metric (`cited_map`) is computed
alongside QA accuracy from the URIs the capability registered via the `cite` tool.

### Per-case results

A QA run writes one JSON line per case to
`<data dir>/evaluations/results/<name>.<trace id>.jsonl` (`--results DIR`
to place it elsewhere): case name, pairing key, verdict, citation flag,
`cited_map`, abort flag, trace id, answer, judge reason, the per-case attributes
and task duration. Rows are appended to `<name>.<run id>.partial.jsonl` as
each case finishes, so a run that is killed keeps the cases it completed. The
run id keeps concurrent runs of one name apart, and a result file is never
overwritten. Live
conversation runs write no file. `--name` must be a file name: letters, digits,
dot, dash and underscore.

Two result files over the same cases pair with

```bash
evaluations pair <treated>.jsonl <baseline>.jsonl
```

which joins them on each case's pairing key (`DatasetSpec.pair_key`) and prints
accuracy, floor, cite rate, mean `cited_map`, aborts and unjudged for each arm,
the discordant counts with the exact McNemar p-value, the `cited_map` sign test,
and the smallest discordance split the exact test would reject. It refuses a
file with a missing or duplicate key, and two files with no case in common.

A run refuses to start when Logfire finds no token, whether from
`LOGFIRE_TOKEN` or a credentials file. `--no-telemetry` runs without Logfire,
leaving the result file as the only record.

### Debugging runs in Logfire

With `LOGFIRE_TOKEN` set, runs ship spans under `service_name = 'evals'`. The
`debug-evals` skill in `.claude/skills/` turns these into ready-made Logfire
queries (recent runs, per-case pass rate and `cited_map`, failing and slowest
cases) for use from Claude Code.

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

With [`lancedb.databases`](https://ggozad.github.io/haiku.rag/configuration/multiple-databases/) configured, `evaluations run <dataset> --skip-db` benchmarks the full set. Retrieval, QA, and live conversations preserve the database name on results and citations. A configured set of one follows the same path and retains its name.

Population writes one database, so it runs only without `lancedb.databases`. `--db` beside a configured set is an error:

```bash
evaluations run hotpotqa --db /path/to/one.lancedb   # no lancedb.databases: populate, then benchmark
evaluations run hotpotqa --skip-db                   # lancedb.databases: benchmark the configured set
```

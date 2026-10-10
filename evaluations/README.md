# haiku.rag evaluations

The benchmark harness for haiku.rag. It builds an evaluation database per dataset, scores retrieval, and runs every question end to end through the RAG capability, scoring answer accuracy and citations. It is a workspace package of this repository and is not published to PyPI. Results and the scoring methodology are on the [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/) page.

## Installation

Runs use the full haiku.rag install, docling included. From the repository root:

```bash
uv sync --all-packages
uv run evaluations --help
```

## Datasets

| Key | Benchmark | Database | QA scored by | Retrieval metrics | Pair key | Reference config |
|---|---|---|---|---|---|---|
| `hotpotqa` | HotpotQA | `hotpotqa.lancedb` | judge | MAP | `question_id` | `hotpotqa.yaml` |
| `frames` | FRAMES | `frames.lancedb` | judge | MAP | `question_id` | `frames.yaml` |
| `orb_text` | OpenRAG Bench | `open_rag_bench_text.lancedb` | judge | MAP | `query_id` | `orb_text.yaml` |
| `orb_multimodal` | OpenRAG Bench | `open_rag_bench_multimodal.lancedb` | judge | MAP | `query_id` | `orb_multimodal.yaml` |
| `orb_multimodal_nemotron` | OpenRAG Bench | `open_rag_bench_multimodal_nemotron.lancedb` | judge | MAP | `query_id` | `orb_multimodal_nemotron.yaml` |
| `t2_finqa` | T²-RAGBench FinQA | `t2_ragbench_finqa.lancedb` | numeric match | MAP | `id` | `t2_finqa.yaml` |
| `t2_tatdqa` | T²-RAGBench TAT-DQA | `t2_ragbench_tatdqa.lancedb` | numeric match | MAP | `id` | `t2_finqa.yaml` |
| `mtrag_clapnq` | MTRAG ClapNQ | `mtrag_clapnq.lancedb` | judge, refusal | Recall@5/10, nDCG@5/10, MAP | `task_id` | `mtrag_clapnq.yaml` |
| `mtrag_clapnq_rewrite` | MTRAG ClapNQ | `mtrag_clapnq.lancedb` | judge, refusal | Recall@5/10, nDCG@5/10, MAP | `task_id` | `mtrag_clapnq.yaml` |
| `mtrag_clapnq_live` | MTRAG ClapNQ | `mtrag_clapnq.lancedb` | judge per turn | none | `conversation_id` | `mtrag_clapnq.yaml` |
| `mtrag_clapnq_live_uncompacted` | MTRAG ClapNQ | `mtrag_clapnq.lancedb` | judge per turn | none | `conversation_id` | `mtrag_clapnq.yaml` |

Reference configs are in `evaluations/configs/`. Every QA run also scores citations (`cited_map`).

- `hotpotqa`: the distractor validation split, 7,405 questions with two gold documents each.
- `frames`: 822 of FRAMES' 824 questions over a fixed corpus of the 2,500 linked Wikipedia articles. Two questions are excluded because they link a deleted article. The articles are fetched from the Wikipedia REST API at their current revision with navigation removed, and the revision id and fetch date are kept in the article cache. The gold answers date from 2024 revisions and may no longer match the current articles.
- `orb_text` embeds with `qwen3-embedding:4b` (2560) and puts VLM picture descriptions in the chunk text. `orb_multimodal` (`qwen3-vl-embedding-8b`, 4096) and `orb_multimodal_nemotron` (`nvidia/llama-nemotron-embed-vl-1b-v2`, 2048) embed pictures and text in one vector space.
- `t2_finqa`, `t2_tatdqa`: financial-report questions with numeric answers, scored by `NumberMatchEvaluator` (relative tolerance 0.01). No judge runs.
- The four MTRAG keys share one database of 183,408 ClapNQ passages, with 208 retrieval queries and 224 generation tasks.
  - `mtrag_clapnq` retrieves with the last user turn as written. It answers each task after replaying the reference conversation before it as message history, and the judge sees the conversation as a transcript. Citation MAP is scored on the turns with gold passages, refusal precision and recall against the answerability labels.
  - `mtrag_clapnq_rewrite` retrieves with the human standalone rewrites.
  - `mtrag_clapnq_live` replays each conversation through one capability session, carrying the model's own answers, tool history and state, with `EvidenceCompactionCapability` registered. It reports per-turn outcomes with micro (per turn) and macro (per conversation) aggregates.
  - `mtrag_clapnq_live_uncompacted` is the same replay without compaction.

## Running a benchmark

```bash
uv run evaluations run hotpotqa                              # build the database, then retrieval and QA
uv run evaluations run hotpotqa --skip-db                    # use the database as it is
uv run evaluations run hotpotqa --skip-db --skip-retrieval   # QA only
uv run evaluations run hotpotqa --skip-db --limit 10 --name smoke
```

Without `--skip-db`, the run first populates the dataset's database. A document already stored with its chunks is skipped, so an interrupted build resumes.

| Option | Effect |
|---|---|
| `--config PATH` | The `haiku.rag.yaml` to use. Without it, `./haiku.rag.yaml`, then the user config directory, then the defaults |
| `--db PATH` | The database path. Default: `evaluations/dbs/<database>` in the haiku.rag data directory |
| `--skip-db` | Do not populate the database |
| `--skip-retrieval`, `--skip-qa` | Skip one benchmark |
| `--limit N` | Keep the first N questions of each benchmark's dataset. Live MTRAG counts conversations |
| `--name NAME` | Name the run. A file name: letters, digits, dot, dash and underscore. Default `<key>_qa_evaluation` and `<key>_retrieval_evaluation` |
| `--filter CLAUSE`, `-f CLAUSE` | Restrict every benchmark search, see [Restricting the corpus](#restricting-the-corpus) |
| `--filter-ids PATH` | Run only the QA cases with ids listed in the file, one per line. Retrieval is unaffected |
| `--multimodal-only` | Retrieval only: keep the queries whose source type includes an image (OpenRAG Bench) |
| `--vacuum-interval N` | Vacuum every N documents while populating. Default 100 |
| `--results DIR` | Directory for the per-case result file. Default `evaluations/results/` in the haiku.rag data directory |
| `--no-telemetry` | Run without Logfire |

QA cases run one at a time, so a run's length is the number of cases times the mean case time. A `--limit` prefix checks the wiring and is a poor estimate of the full run's case time.

### Models

The capability runs on `qa.model` and the judge on `evaluations.judge`, both from the config. Without `evaluations.judge`, the judge is `ollama:qwen3.8`. Every reference config of a judged dataset carries the same judge block, and `evaluations/tests/test_reference_configs.py` keeps it identical across them. The reference configs name vLLM endpoints at `http://vllm:<port>`. Point their `base_url`s at your own OpenAI-compatible servers.

`evaluations.system_one` puts a System One decision model in front of the judge. The [Benchmarks](https://ggozad.github.io/haiku.rag/benchmarks/#faster-judging-with-a-system-one-model) page describes it and its measurements.

## Run identity and telemetry

A run prints the code revision and the config hash when it starts:

```
Code: 56bb99b1828efc9ba9741b819e424aff80f36c50 | config hash: cd96ca0b1836de767771f0a220023cd48533a57242865f88870c5063f40a285f
```

The same values are in the run's experiment metadata, with the corpus fingerprint and the settings that change results:

- `git_sha`, `git_dirty` (untracked files ignored) and `config_hash`, the SHA-256 of the resolved `AppConfig`.
- `db_path`, `db_documents`, `db_chunks`, `db_embedder_provider`, `db_embedder_model`, `db_embedder_dim`, `db_version`, and `db_written_at`, the newest table version time. All are None when `lancedb.databases` places the databases.
- The embedder, reranker, chunk size, `search_limit`, `max_context_chars`, `qa_max_searches`, `qa_max_executions`, `sandbox_code_timeout`, `sandbox_max_output_chars`, `document_filter` and `pair_key`.
- `judge_*` and, on a QA run, `capability_*`: provider, model, temperature, max tokens, thinking and `extra_body`.

Runs send spans to Logfire under `service_name = 'evals'`. A run refuses to start when Logfire finds no token, from `LOGFIRE_TOKEN` or a credentials file. `--no-telemetry` runs without Logfire, and the result file is then the only record of the run.

## Per-case results

A QA run writes one JSON line per case to `<results dir>/<name>.<trace id>.jsonl`, and prints the path at the end. Each row holds the case name, the pairing key, the verdict, whether the answer cited anything, `cited_map`, the abort flag, the trace id, the answer, the judge's reason, the per-case attributes and the task duration. A run gated by `evaluations.system_one` also records which model decided each verdict, the probability and the served model.

While the run is in progress, rows are appended to `<name>.<run id>.partial.jsonl`, so a killed run keeps the cases it completed. The run id keeps concurrent runs of one name apart, and a result file is never overwritten. Live MTRAG runs write no file.

The per-case attributes are counted from the message history:

- `n_searches`: distinct search queries, with every in-code `search()` filed under one `_sandbox` key.
- `n_search_calls`, `n_sandbox_search_calls`, `n_rejected_searches` (search budget spent), `n_failed_tools`, `n_executions` and `n_requests`.
- `cited_uris`, `cited_chunk_ids` and `searched_uris`.
- `citation_status`: `grounded`, `ungrounded` (the model called `cite([])`) or `missing`.

## Comparing two runs

```bash
uv run evaluations pair <treated>.jsonl <baseline>.jsonl
```

`pair` joins two result files on each case's pair key and prints, for each run, accuracy, floor, cite rate, mean `cited_map`, aborts and unjudged cases. It then prints the discordant counts with the exact McNemar p-value, the sign test on `cited_map`, and the smallest discordant split the exact test would reject. It refuses a file with a missing or duplicate key and two files with no case in common. It warns when only one file comes from a run gated by `evaluations.system_one`, or when the two runs used different System One models.

A comparison measures one change only when everything else matches:

- The same database file, unchanged between the runs. Rebuilding, re-embedding or migrating a database starts a new baseline, and so does adding a vector index, which makes vector search approximate.
- One code revision, or two whose difference is the change under test. Check `git_sha` and `git_dirty` on both runs.
- The same judge. A run gated by a System One model and an ungated run differ by the judge as well.
- The same `--filter`, or none on both.

Two runs of identical code and config answer some cases differently: the capability model samples, and vLLM batching reorders near-tied embedder and reranker scores. A null pair, the same arm run twice, measures that noise. Read a difference against it, and against the smallest significant split in the output of `pair`. Accuracy over judged cases excludes aborted cases, so report the floor, passed over all cases, beside it.

## Restricting the corpus

When a database holds several corpora and a dataset's questions come from one, `--filter` restricts every benchmark search to it. It takes the same SQL `WHERE` clause as `haiku-rag search --filter`, over the document columns `id`, `uri`, `title`, `created_at`, `updated_at` and `metadata`. Each dataset writes its own URIs: `orb_text` uses arXiv ids such as `2407.01528v3`, `hotpotqa` page titles.

```bash
uv run evaluations run orb_text --skip-db --filter "uri LIKE '2407%'"
```

`metadata` is stored as a `json.dumps` string with no subfield access, so match the serialized pair, including the space after the colon:

```bash
uv run evaluations run orb_text --skip-db --filter "metadata LIKE '%\"corpus\": \"orb_text\"%'"
```

The clause applies to the retrieval benchmark and to every search by the capability during QA, and is recorded as `document_filter`. It restricts searches only: a run without `--skip-db` still populates the full corpus.

## Databases

Evaluation databases are stored under `evaluations/dbs/` in the haiku.rag data directory:

- Linux: `~/.local/share/haiku.rag/evaluations/dbs/`
- macOS: `~/Library/Application Support/haiku.rag/evaluations/dbs/`
- Windows: `C:/Users/<USER>/AppData/Roaming/haiku.rag/evaluations/dbs/`

Pre-built databases are on HuggingFace, in `ggozad/haiku-rag-eval-dbs`:

```bash
uv run evaluations download hotpotqa
uv run evaluations download all
uv run evaluations download hotpotqa --force   # overwrite a local copy
```

A database opens only against the embedder recorded in it, so run a downloaded database with `--skip-db` and its reference config:

```bash
uv run evaluations run orb_multimodal_nemotron --skip-db --config evaluations/configs/orb_multimodal_nemotron.yaml
```

`evaluations upload <key|all>` replaces the hosted copy and needs write access to the HuggingFace repository.

### Multiple databases

With [`lancedb.databases`](https://ggozad.github.io/haiku.rag/configuration/multiple-databases/) configured, `evaluations run <key> --skip-db` benchmarks the configured set, and results and citations keep each database's name. A configured set of one takes the same path. Population writes one database, so it runs only without `lancedb.databases`, and `--db` beside a configured set is an error.

## Debugging runs in Logfire

The `debug-evals` skill in `.claude/skills/` gives Claude Code ready-made Logfire queries over the `evals` service: recent runs, per-case pass rate and `cited_map`, failing and slowest cases, and the agent activity inside one case.

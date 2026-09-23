# Benchmarks

haiku.rag is evaluated on OpenRAG Bench (ORB), T²-RAGBench, HotpotQA, FRAMES and MTRAG. Retrieval, QA accuracy and citation retrieval are scored end to end through the RAG capability, see [Methodology](#methodology).

Each footnote names the haiku.rag version and the judge a row was measured with. Rows are not re-run when either changes, so compare rows that share both. Rows before v0.87.0 ran the capability architecture that release replaced.

No benchmark database has a vector index, so every number reflects exact brute-force kNN. For the effect of an index, see [Vector indexing](configuration/storage.md#vector-indexing).

## OpenRAG Bench (ORB)

[OpenRAG Bench](https://huggingface.co/datasets/vectara/open_ragbench) is arXiv papers with multimodal question-answer pairs, text-based and image-based, over figures, charts and diagrams. Each query maps to one relevant document.

Two approaches are measured:

- **Multimodal embedder** (`Qwen/Qwen3-VL-Embedding-8B` or `nvidia/llama-nemotron-embed-vl-1b-v2`, served by vLLM): pictures and text share one vector space, and no VLM runs at ingest.
- **Text embedder with VLM picture descriptions** (`qwen3-embedding:4b` with `ollama/ministral-3`): pictures are described at ingest and the descriptions join the chunk text. See [Picture handling](configuration/processing.md#picture-handling).

### Multimodal embedder

#### Retrieval (MAP)

| Embedding Model                          | Reranker                                             | Cases | MAP    |
|------------------------------------------|------------------------------------------------------|------:|-------:|
| `Qwen/Qwen3-VL-Embedding-8B`             | none                                                 |  3045 | 0.9774 |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | none                                                 |  3045 | 0.9798 |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | `nvidia/llama-nemotron-rerank-vl-1b-v2` (multimodal) |  3045 | 0.9899 |

*The reranked row: v0.88.0, `reranking.multimodal: true`, the stack of the QA rows below. The Qwen3-VL row: v0.52.0. The Nemotron row without a reranker was measured earlier.*

#### QA accuracy and citation retrieval

| Capability model | Cases | QA accuracy | Mean `cited_map` |
|------------------|------:|-------------|------------------|
| `vllm:Muse-Glimmer-30B-NVFP4-W4A4` | 3045 | 0.9620 | 0.9847 |
| `vllm:Qwen3.8-27B-NVFP4` | 3045 | 0.9688 | 0.9907 |

*v0.88.0, the reranked retrieval stack above, judged by `Qwen3.8-27B` at `reasoning_effort: low`. Accuracy is over judged cases, 3028 and 3043 of 3045, the rest lost to the search tool exhausting its retry limit. Counting the unjudged as failures gives 0.9566 and 0.9682. Mean 26.15 s and 13.95 s per case. Muse-Glimmer runs at `chat_template_kwargs.reasoning_strength: high`, Qwen3.8 at `reasoning_effort: low`.*

*Paired on the 3023 cases judged in both, accuracy does not separate the models (McNemar exact p = 0.09). `cited_map` favours Qwen3.8, better on 44 cases and worse on 19 (sign test p = 0.0025). The Qwen3.8 row is self-judged. An independent judge agreed on 95% of a 120-case sample and was never stricter.*

### Text embedder with VLM picture descriptions

#### Retrieval (MAP)

| Embedding Model                          | VLM                  | Reranker               | Cases | MAP    |
|------------------------------------------|----------------------|------------------------|------:|-------:|
| `qwen3-embedding:4b`                     | Ollama / ministral-3 | `mxbai-rerank-base-v2` |  3045 | 0.9834 |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | Ollama / ministral-3 | `mxbai-rerank-base-v2` |  3045 | 0.9863 |

*v0.50.0.*

#### QA accuracy and citation retrieval

| Embedding Model                          | VLM                  | Capability model                  | Cases | QA accuracy | Mean `cited_map` |
|------------------------------------------|----------------------|------------------------------|------:|-------------|------------------|
| `qwen3-embedding:4b`                     | Ollama / ministral-3 | `vllm:Gemma-4-26B-A4B-NVFP4` |  3045 | 0.92        | 0.80             |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | Ollama / ministral-3 | `vllm:Gemma-4-26B-A4B-NVFP4` |  2836 | 0.96        | 0.81             |

*v0.50.0 with `mxbai-rerank-base-v2`, judged by `vllm:Qwen3.6-35B-A3B-NVFP4`. The Nemotron row covers 2836 of 3045 cases.*

## T²-RAGBench (FinQA)

[T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench) turns financial-report QA into context-independent questions with short numeric answers and one gold document each. The FinQA subset is 2,789 single-page PDFs and 8,281 questions. QA is scored by exact numeric match (`NumberMatchEvaluator`, relative tolerance 0.01), not by a judge.

#### QA accuracy and citation retrieval

| Embedding Model      | Reranker               | Capability model             | Cases | QA accuracy | Mean `cited_map` |
|----------------------|------------------------|------------------------------|------:|-------------|------------------|
| `qwen3-embedding:4b` | `mxbai-rerank-base-v2` | `vllm:Qwen3.6-35B-A3B-NVFP4` |  7939 | 0.77        | 0.78             |

*v0.55.0, `qwen3-embedding:4b` (vLLM, 2560) with `mxbai-rerank-base-v2`. 341 of 8281 cases produced no answer, and accuracy and `cited_map` are over the 7939 scored. Mean 16.0 s per case.*

## HotpotQA

[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) is multi-hop QA over Wikipedia: each question combines facts from two paragraphs, among distractor paragraphs. The distractor validation split has 7,405 questions over about 66k paragraphs, two gold documents per question.

#### Retrieval (MAP)

| Embedding Model      | Reranker            | Cases | MAP    |
|----------------------|---------------------|------:|-------:|
| `qwen3-embedding:4b` | `Qwen3-Reranker-4B` |  7405 | 0.8202 |
| `qwen3-embedding:4b` | none                |  7405 | 0.6995 |

Hybrid search usually ranks the first-hop document first. The second-hop document often needs the reranker to reach the result window.

#### QA accuracy and citation retrieval

| Capability model             | Reranker            | QA accuracy | Mean `cited_map` |
|------------------------------|---------------------|-------------|------------------|
| `vllm:Gemma-4-26B-A4B-NVFP4` | `Qwen3-Reranker-4B` | 0.85        | 0.80             |
| `vllm:Gemma-4-26B-A4B-NVFP4` | none                | 0.83        | 0.75             |

*v0.66.0, `qwen3-embedding:4b` (vLLM, 2560), judged by `vllm:Qwen3.6-35B-A3B-NVFP4`, 7,405 cases. Without a reranker, `cited_map` (0.75) exceeds retrieval MAP (0.70): the capability searches several times and recovers second-hop documents one query misses.*

## FRAMES

[FRAMES](https://huggingface.co/datasets/google/frames-benchmark) is Google's multi-hop QA benchmark: 824 questions, each grounded in 2 to 23 Wikipedia articles, with temporal, numerical and tabular reasoning. 822 are evaluated, 2 being excluded because a linked article was deleted, over a fixed corpus of the 2,500 linked articles fetched at their current revision. FRAMES has no official evaluation setup. A fixed corpus with agentic retrieval and judged accuracy corresponds to the paper's multi-step retrieval setting, where [the paper](https://arxiv.org/abs/2409.12941) reports 0.66 with Gemini-Pro-1.5 (0.729 given the gold articles). Answers were written against 2024 revisions and may have drifted with the articles.

Gold sets span 2 to 23 articles, more than a result window holds or an answer cites, so MAP and `cited_map` stay well below 1.

#### Retrieval (MAP)

| Embedding Model                        | Reranker                                 | Cases | MAP    |
|----------------------------------------|------------------------------------------|------:|-------:|
| `nvidia/llama-nemotron-embed-vl-1b-v2` | `nvidia/llama-nemotron-rerank-vl-1b-v2` |   822 | 0.5966 |

*v0.88.0, the stack of the QA rows below. Single-query retrieval scores zero on 11 questions, those whose gold article's subject the question never names. The agentic run answers 6 of them correctly.*

#### QA accuracy and citation retrieval

| Capability model | Cases | QA accuracy | Mean `cited_map` |
|------------------|------:|-------------|------------------|
| `vllm:Muse-Glimmer-30B-NVFP4-W4A4` | 822 | 0.8938 | 0.7342 |
| `vllm:Qwen3.8-27B-NVFP4` | 822 | 0.9077 | 0.7312 |

*v0.88.0, `nvidia/llama-nemotron-embed-vl-1b-v2` (vLLM, 2048) with `nvidia/llama-nemotron-rerank-vl-1b-v2`, judged by `Qwen3.8-27B` at `reasoning_effort: low`. Accuracy is over judged cases, 810 and 802 of 822, the rest lost to the search tool exhausting its retry limit. Counting the unjudged as failures gives 0.8808 and 0.8856. Mean 62.56 s and 30.21 s per case. The Qwen3.8 row is self-judged.*

*Paired on the 791 cases judged in both, neither metric separates the models (accuracy McNemar exact p = 0.80, `cited_map` sign test p = 0.16). Not comparable with FRAMES results before v0.87.0, which used a different corpus build, embedder, reranker and capability.*

## MTRAG (ClapNQ)

[MTRAG](https://github.com/IBM/mt-rag-benchmark) is IBM's multi-turn RAG benchmark: human-written conversations with per-turn answerability labels and binary relevance judgments. The ClapNQ (Wikipedia) domain has 183,408 passages, 29 conversations, 224 turns and 208 retrieval queries.

Four dataset keys share one database:

- `mtrag_clapnq` retrieves with the last user turn as written, and answers each turn after replaying the reference conversation before it.
- `mtrag_clapnq_rewrite` retrieves with the human standalone rewrites.
- `mtrag_clapnq_live` replays whole conversations through one capability session, carrying the model's own answers, tools and state, with `EvidenceCompactionCapability` registered.
- `mtrag_clapnq_live_uncompacted` is the same replay without compaction. It is the only evaluation where compaction acts.

#### Retrieval (Recall@k / nDCG@k)

Comparable with [IBM's published results](https://github.com/IBM/mt-rag-benchmark/tree/main/mtrag-human/retrieval_tasks). Elser is IBM's strongest reported retriever.

| Retriever | Queries | R@5 | R@10 | nDCG@5 | nDCG@10 |
|-----------|---------|----:|-----:|-------:|--------:|
| Elser (IBM) | lastturn | 0.49 | 0.58 | 0.45 | 0.49 |
| `haiku.rag` | lastturn | 0.501 | 0.600 | 0.455 | 0.497 |
| Elser (IBM) | rewrite | 0.52 | 0.64 | 0.48 | 0.54 |
| `haiku.rag` | rewrite | 0.548 | 0.668 | 0.503 | 0.556 |

#### QA accuracy and citation retrieval

| Mode | Capability model | Turns | QA accuracy | Mean `cited_map` |
|------|------------------|------:|-------------|------------------|
| Gold-prefix (`mtrag_clapnq`) | `vllm:Muse-Glimmer-30B-NVFP4` | 224 | 0.76 | 0.35 |
| Live compacted (`mtrag_clapnq_live`) | `vllm:Muse-Glimmer-30B-NVFP4` | 224/224 scored | 0.83 micro / 0.84 macro | 0.42 |
| Live uncompacted (`mtrag_clapnq_live_uncompacted`) | `vllm:Muse-Glimmer-30B-NVFP4` | 224/224 scored | 0.78 micro / 0.79 macro | 0.42 |

*v0.74.0, `qwen3-embedding:4b` (vLLM, 2560), `Qwen3-Reranker-4B`, `reasoning_strength: high`, judged by `vllm:Qwen3.6-35B-A3B-NVFP4` (temperature 0.6, thinking). QA uses our judge and rubric and is not comparable with IBM's generation metrics. Gold-prefix and live rates answer different judge questions and are not comparable with each other. `cited_map` is over the 208 turns with gold passages.*

The two live arms replay the same 224 turns and differ only in compaction:

| | Compacted | Uncompacted |
|---|---:|---:|
| Input tokens per model request | 7,461 | 13,539 |
| Turns passed | 185 | 175 |
| Citation MAP (macro, 208 eligible turns) | 0.4174 | 0.4230 |
| Refusal precision / recall (16 unanswerable turns) | 0.33 / 0.44 | 0.23 / 0.31 |

Compaction cut input tokens per request by 44.9%. Of the 18 turns where the arms disagree, 14 pass only compacted (McNemar exact p = 0.031, paired difference +4.5 points, 95% CI +0.8 to +8.1). The gold-prefix arm's refusal precision and recall are 0.24 and 0.44.

## Methodology

### Retrieval metrics

**Mean Average Precision (MAP)** scores the ranked results against the gold `expected_uris`:

- At each relevant document's position k, precision@k is the relevant documents in the top k divided by k.
- Average precision (AP) is the sum of those precisions divided by the number of relevant documents.
- MAP is the mean AP over queries, from 0 to 1. For a single-document query it is `1/rank`.

MTRAG also reports Recall@k and nDCG@k.

### QA accuracy

A `pydantic-evals` LLM judge decides whether each answer is correct, and accuracy is the fraction judged correct. The default judge is `ollama:qwen3.8`, pinned so that changing the capability model does not change the judge. `evaluations.judge` in `haiku.rag.yaml` overrides it, including any OpenAI-compatible endpoint. Against the previous judge on 120 ORB cases, `qwen3.8` agreed on 95% (Cohen's κ 0.90).

A dataset with its own deterministic evaluator is scored by it, and no judge runs. T²-RAGBench is the only one, with `NumberMatchEvaluator`.

### Citation retrieval

`cited_map` scores the URIs the capability registered with `cite` against the gold `expected_uris`, with the same MAP arithmetic. It comes from the same run as QA accuracy. Where retrieval MAP measures whether the retriever returned the gold document, `cited_map` measures whether the answer was grounded on it. The console also reports the cite rate, the share of cases with at least one citation, and mean citations per case.

## Running evaluations

The `evaluations` CLI runs the benchmarks, orchestrated with [`pydantic-evals`](https://github.com/pydantic/pydantic-ai/tree/main/libs/pydantic-evals):

```bash
evaluations run hotpotqa
evaluations run orb_text
```

### Pre-built databases

Building an evaluation database takes long, especially for OpenRAG Bench. Pre-built ones are on HuggingFace:

```bash
evaluations download hotpotqa
evaluations download all
evaluations download hotpotqa --force   # overwrite
```

| Dataset | Size |
|---------|------|
| `orb_text`: OpenRAG Bench, `qwen3-embedding:4b` with VLM picture descriptions in the chunk text | ~15.8 GB |
| `orb_multimodal`: OpenRAG Bench, multimodal `qwen3-vl-embedding-8b` | ~16.7 GB |
| `orb_multimodal_nemotron`: OpenRAG Bench, multimodal `nvidia/llama-nemotron-embed-vl-1b-v2` | ~15.2 GB |
| `t2_finqa`: T²-RAGBench FinQA, `qwen3-embedding:4b` | ~2.0 GB |
| `t2_tatdqa`: T²-RAGBench TAT-DQA, `qwen3-embedding:4b` | ~1.8 GB |
| `hotpotqa`: HotpotQA, `qwen3-embedding:4b` | ~1.2 GB |
| `frames`: FRAMES, `nvidia/llama-nemotron-embed-vl-1b-v2` | ~7.2 GB |
| `mtrag_clapnq`: MTRAG ClapNQ, `qwen3-embedding:4b`, shared by the `_rewrite`, `_live` and `_live_uncompacted` keys | ~2.7 GB |

The hosted `frames` and `orb_multimodal_nemotron` databases are the corpora behind the current FRAMES and ORB rows.

After downloading, run with `--skip-db` and the database's reference config from `evaluations/configs/`, since a database opens only against the embedder it was built with:

```bash
evaluations run orb_multimodal_nemotron --skip-db --config configs/orb_multimodal_nemotron.yaml
```

`t2_tatdqa` has no reference config of its own. `configs/t2_finqa.yaml` uses the same embedder and opens it.

The configs use vLLM endpoints. Point their `base_url` at your own OpenAI-compatible servers to run them.

### Options

```bash
evaluations run hotpotqa --config /path/to/haiku.rag.yaml --db /path/to/custom.lancedb
```

- `--config PATH`: the `haiku.rag.yaml` to use. Without it, `./haiku.rag.yaml`, then the user config directory, then the defaults
- `--db PATH`: the database path (default: the platform data directory)
- `--skip-db`: do not update the evaluation database
- `--skip-retrieval`, `--skip-qa`: skip one benchmark
- `--limit N`: limit the number of cases
- `--name NAME`: name the run, as a file name (letters, digits, dot, dash, underscore)
- `--filter CLAUSE` / `-f CLAUSE`: restrict every benchmark search, see [Restricting the corpus](#restricting-the-corpus)
- `--filter-ids PATH`: run only the QA cases whose ids the file lists, one per line. Retrieval is unaffected
- `--multimodal-only`: only the queries that need image understanding
- `--vacuum-interval N`: vacuum every N documents while populating (default 100)
- `--results DIR`: directory for the per-case result file (default `evaluations/results/` in the haiku.rag data directory)
- `--no-telemetry`: run without Logfire. Without it, a run refuses to start when Logfire finds no token (`LOGFIRE_TOKEN` or a credentials file)

The capability runs on `qa.model` and the judge on `evaluations.judge`, both from the configuration.

A QA run writes one JSON line per case to `<name>.<trace id>.jsonl` in the results directory. `evaluations pair TREATED BASELINE` joins two such files on their cases and prints the paired comparison, with exact McNemar on verdicts and the sign test on `cited_map`. See the [evaluations README](https://github.com/ggozad/haiku.rag/blob/main/evaluations/README.md#per-case-results).

The recommended judge settings, pinned in YAML:

```yaml
evaluations:
  judge:
    provider: openai
    name: Inferact/Qwen3.8-27B-NVFP4
    base_url: http://localhost:8000/v1   # optional, for OpenAI-compatible servers
    temperature: 0.6
    max_tokens: 16384
    extra_body:
      top_p: 0.95
      top_k: 20
      min_p: 0
      chat_template_kwargs:
        reasoning_effort: low   # qwen3.8: low | medium | xhigh (default)
```

### Restricting the corpus

When a database holds several corpora and a dataset's questions come from one, `--filter` restricts every benchmark search to it. It takes the SQL `WHERE` clause `haiku-rag search --filter` takes, over the document columns (`id`, `uri`, `title`, `created_at`, `updated_at`, `metadata`). Each dataset writes its own URIs: `orb_text` uses arXiv ids such as `2407.01528v3`, `hotpotqa` page titles.

```bash
evaluations run orb_text --skip-db --config haiku.rag.s3.yaml \
  --filter "uri LIKE '2407%'"
```

A corpus distinguished by a tag rather than a URI can carry it as document metadata. `metadata` is stored as a `json.dumps` string with no subfield access, so match the serialized pair, including the space after the colon:

```bash
evaluations run orb_text --skip-db --filter "metadata LIKE '%\"corpus\": \"orb_text\"%'"
```

The clause applies to the retrieval benchmark and to every search the capability runs during QA, so both score the same subset. It is recorded as `document_filter` in the run's metadata. It restricts searches only: a run without `--skip-db` still populates the full corpus.

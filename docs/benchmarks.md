# Benchmarks

We evaluate `haiku.rag` on a small set of datasets that exercise different parts of the pipeline. OpenRAG Bench (ORB), T²-RAGBench, HotpotQA, and MTRAG are the datasets we currently track. Retrieval, QA accuracy, and citation retrieval are scored end-to-end through the RAG and analysis capabilities.

## Current results

Numbers below were measured on a recent `haiku.rag` version. Most rows were judged by `Qwen3.6-35B-A3B-NVFP4`; the `Qwen3.8-27B` rows were judged by the currently pinned `qwen3.8`, as their footnote states. Rows are not re-judged when the pinned judge changes, so compare rows judged by the same judge and treat cross-judge differences as unmeasured.

### OpenRAG Bench (ORB)

[OpenRAG Bench](https://huggingface.co/datasets/vectara/open_ragbench) contains ArXiv research papers with multimodal question-answering pairs. Queries include both text-based and image-based questions, testing retrieval and reasoning over visual content like figures, charts, and diagrams. Each query maps to one relevant document.

Two approaches are benchmarked separately:

- **Multimodal embedder** (`Qwen/Qwen3-VL-Embedding-8B`, served via vLLM): picture bytes and text live in a shared vector space, no VLM is run at ingest.
- **Text embedder + VLM picture descriptions** (`qwen3-embedding:4b` + `ollama/ministral-3`): pictures are described at ingest and the descriptions are woven into chunk text. Retrieval runs over text only. See [Picture handling configuration](configuration/processing.md#picture-handling).

#### Multimodal embedder

##### Retrieval (MAP)

| Embedding Model                          | Reranker                                             | Cases | MAP    |
|------------------------------------------|------------------------------------------------------|------:|-------:|
| `Qwen/Qwen3-VL-Embedding-8B`             | none                                                 |  3045 | 0.9774 |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | none                                                 |  3045 | 0.9798 |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | `nvidia/llama-nemotron-rerank-vl-1b-v2` (multimodal) |  3045 | 0.9913 |

*The nemotron row without a reranker is measured on this release. The reranked row uses `reranking.multimodal: true`: picture chunks reach the vision reranker as images alongside their description text, measured on haiku.rag main post-v0.67.3.*

##### QA accuracy + citation retrieval

| Embedding Model                          | Target          | Capability model                       | Cases | QA accuracy | Mean `cited_map` |
|------------------------------------------|-----------------|-----------------------------------|------:|-------------|------------------|
| `Qwen/Qwen3-VL-Embedding-8B`             | `rag-capability`     | `vllm:Gemma-4-26B-A4B-NVFP4`      |  1409 | 0.89        | —                |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | `rag-capability`     | `vllm:Gemma-4-26B-A4B-NVFP4`      |  3039 | 0.9263      | 0.9761           |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | `analysis-capability`| `vllm:Gemma-4-26B-A4B-NVFP4`      |  3040 | 0.9362      | 0.9343           |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | `analysis-capability`| `vllm:Qwen3.6-35B-A3B-NVFP4`      |  3045 | 0.95        | 0.93             |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | `rag-capability`     | `vllm:Muse-Glimmer-30B-NVFP4`     |  3045 | 0.9494      | 0.9771           |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | `analysis-capability`| `vllm:Muse-Glimmer-30B-NVFP4`     |  3017 | 0.9718      | 0.9837           |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | `rag-capability`     | `vllm:Qwen3.8-27B-NVFP4`          |  3045 | 0.9514      | 0.9817           |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | `analysis-capability`| `vllm:Qwen3.8-27B-NVFP4`          |  3042 | 0.9629      | 0.9835           |

*The `Muse-Glimmer-30B` rows run at `chat_template_kwargs.reasoning_strength: high`, no reranker, same judge.*

*The `Qwen3.8-27B` rows run at `chat_template_kwargs.reasoning_effort: low`, no reranker, and are **judged by `Qwen3.8-27B` itself** — the pinned judge, and the same model that produced the answers. A 120-case cross-check by an independent judge agreed on 95% and was never stricter, but the incumbent `Qwen3.6` judge is no longer hosted, so the older rows cannot be re-judged for a like-for-like comparison. `rag-capability` cites on 99.34% of cases with 1.09 citations each; `analysis-capability` on 99.70% with 1.12, at 0.13 code executions per case. Case counts exclude 0 and 3 provider errors respectively.*

*Both nemotron `Gemma-4` rows are measured on this release, no reranker, judged by `vllm:Qwen3.6-35B-A3B-NVFP4` with thinking on, and exclude the cases that errored (6 of 3045 for `rag-capability`, 5 for `analysis-capability`). The `rag-capability` row cites at 99.64% with a mean of 1.08 citations per case, at a median 4.7s per case against 5.0s for `analysis-capability`. Citation coverage is what moved on this release: 4.9% of analysis cases register no citation, against 26.3% before, at unchanged searches and code executions per case. The remaining rows are from haiku.rag v0.52.0, where Qwen3-VL covered 1409 / 3045 cases.*

#### Text embedder + VLM picture descriptions

##### Retrieval (MAP)

| Embedding Model                          | VLM                  | Reranker               | Cases | MAP    |
|------------------------------------------|----------------------|------------------------|------:|-------:|
| `qwen3-embedding:4b`                     | Ollama / ministral-3 | `mxbai-rerank-base-v2` |  3045 | 0.9834 |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | Ollama / ministral-3 | `mxbai-rerank-base-v2` |  3045 | 0.9863 |

*Measured on haiku.rag v0.50.0.*

##### QA accuracy + citation retrieval

| Embedding Model                          | VLM                  | Capability model                  | Cases | QA accuracy | Mean `cited_map` |
|------------------------------------------|----------------------|------------------------------|------:|-------------|------------------|
| `qwen3-embedding:4b`                     | Ollama / ministral-3 | `vllm:Gemma-4-26B-A4B-NVFP4` |  3045 | 0.92        | 0.80             |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | Ollama / ministral-3 | `vllm:Gemma-4-26B-A4B-NVFP4` |  2836 | 0.96        | 0.81             |

*Measured on haiku.rag v0.50.0 with `mxbai-rerank-base-v2`, judged by `vllm:Qwen3.6-35B-A3B-NVFP4`. Nemotron covered 2836 / 3045 cases.*

### T²-RAGBench (FinQA)

[T²-RAGBench](https://huggingface.co/datasets/G4KMU/t2-ragbench) reformulates financial-report QA into context-independent questions with short numeric answers and a 1:1 gold document mapping. The FinQA subset is 2,789 single-page PDFs / 8,281 questions, ingested via docling. Unlike the other datasets, QA is scored deterministically with `NumberMatchEvaluator` (relative tolerance 0.01) instead of an LLM judge, so QA accuracy here is exact numeric match rather than a judged fraction.

##### QA accuracy + citation retrieval

| Embedding Model      | Reranker               | Target           | Capability model                  | Cases | QA accuracy | Mean `cited_map` |
|----------------------|------------------------|------------------|------------------------------|------:|-------------|------------------|
| `qwen3-embedding:4b` | `mxbai-rerank-base-v2` | `analysis-capability` | `vllm:Qwen3.6-35B-A3B-NVFP4` |  7939 | 0.77        | 0.78             |

*Measured on haiku.rag v0.55.0, deterministic Number-Match scoring (ε=0.01), 2560-dim `qwen3-embedding:4b` (vLLM) with `mxbai-rerank-base-v2`. 341 / 8281 cases excluded as nulls (analysis spirals from the request limit and in-generation loops). Accuracy and `cited_map` are over the 7939 scored cases. Mean 16.0s/case.*

### HotpotQA

[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) is multi-hop question answering over Wikipedia: each question requires combining facts from two supporting paragraphs, with distractor paragraphs in the corpus. We use the distractor validation split: 7,405 questions over ~66k unique paragraphs, each question mapping to two gold documents.

##### Retrieval (MAP)

| Embedding Model      | Reranker            | Cases | MAP    |
|----------------------|---------------------|------:|-------:|
| `qwen3-embedding:4b` | `Qwen3-Reranker-4B` |  7405 | 0.8202 |
| `qwen3-embedding:4b` | none                |  7405 | 0.6995 |

The reranker's contribution is larger here than on the single-doc datasets: hybrid search usually surfaces the first-hop document at rank 1, while the second-hop document often needs the reranker to climb into the result window.

##### QA accuracy + citation retrieval

| Skill model                  | Reranker            | QA accuracy | Mean `cited_map` |
|------------------------------|---------------------|-------------|------------------|
| `vllm:Gemma-4-26B-A4B-NVFP4` | `Qwen3-Reranker-4B` | 0.85        | 0.80             |
| `vllm:Gemma-4-26B-A4B-NVFP4` | none                | 0.83        | 0.75             |

*Measured on haiku.rag v0.66.0 with `qwen3-embedding:4b` (vLLM, dim 2560), judged by `vllm:Qwen3.6-35B-A3B-NVFP4`, 7,405 cases. The reranker lifts QA accuracy +2.7pts and `cited_map` +4.6pts. Without a reranker, `cited_map` (0.75) still exceeds the no-reranker retrieval MAP (0.70): the skill reformulates queries across search calls, partially recovering second-hop documents that a single query misses.*

### MTRAG (ClapNQ)

[MTRAG](https://github.com/IBM/mt-rag-benchmark) is IBM's multi-turn RAG benchmark (TACL 2025, SemEval-2026 Task 8): human-authored conversations with per-turn answerability labels and binary relevance judgments. We evaluate the ClapNQ (Wikipedia) domain: 183,408 passages, 29 conversations, 224 turns, 208 retrieval queries.

Four dataset keys share one database. `mtrag_clapnq` retrieves with the raw last user turn and runs QA by replaying each task's reference conversation prefix as message history. `mtrag_clapnq_rewrite` retrieves with the human standalone rewrites. `mtrag_clapnq_live` replays whole conversations through a single capability session, carrying the model's own answers, tool history and capability state across turns, with `EvidenceCompactionCapability` registered. `mtrag_clapnq_live_uncompacted` is the same replay without compaction, isolating what compaction contributes. This is the only multi-turn evaluation, so it is the only one where compaction acts at all.

##### Retrieval (Recall@k / nDCG@k)

Directly comparable with [IBM's published results](https://github.com/IBM/mt-rag-benchmark/tree/main/mtrag-human/retrieval_tasks). Elser is IBM's strongest reported retriever.

| Retriever | Queries | R@5 | R@10 | nDCG@5 | nDCG@10 |
|-----------|---------|----:|-----:|-------:|--------:|
| Elser (IBM) | lastturn | 0.49 | 0.58 | 0.45 | 0.49 |
| `haiku.rag` | lastturn | 0.501 | 0.600 | 0.455 | 0.497 |
| Elser (IBM) | rewrite | 0.52 | 0.64 | 0.48 | 0.54 |
| `haiku.rag` | rewrite | 0.548 | 0.668 | 0.503 | 0.556 |

##### QA accuracy + citation retrieval

| Mode | Capability model | Turns | QA accuracy | Mean `cited_map` |
|------|------------------|------:|-------------|------------------|
| Gold-prefix (`mtrag_clapnq`) | `vllm:Muse-Glimmer-30B-NVFP4` | 224 | 0.76 | 0.35 |
| Live compacted (`mtrag_clapnq_live`) | `vllm:Muse-Glimmer-30B-NVFP4` | 224/224 scored | 0.83 micro / 0.84 macro | 0.42 |
| Live uncompacted (`mtrag_clapnq_live_uncompacted`) | `vllm:Muse-Glimmer-30B-NVFP4` | 224/224 scored | 0.78 micro / 0.79 macro | 0.42 |

*Measured on haiku.rag v0.74.0 with `qwen3-embedding:4b` (vLLM, dim 2560), `Qwen3-Reranker-4B`, stock capability instructions with `reasoning_strength: high`, judged by the pinned `vllm:Qwen3.6-35B-A3B-NVFP4` (temperature 0.6, thinking). QA numbers are internal (our judge and rubric) and not comparable with IBM's published generation metrics. Gold-prefix and live rates answer different judge questions and are not comparable with each other. The dataset is text-only (ClapNQ passages), so it exercises no multimodal paths.*

The two live arms replay the same 29 conversations (224 turns) and differ only in registering `EvidenceCompactionCapability`, so they are compared as paired observations:

- Input tokens per model request, computed as total input tokens divided by model requests across the whole arm: 7,461 compacted (5,207,627 tokens over 698 requests) vs 13,539 uncompacted (9,707,530 over 717 requests). The uncompacted arm used 1.81x as many tokens per request, a 44.9% reduction under compaction.
- Answer pass rate: 185/224 vs 175/224 turns. Of the 18 turns where the arms disagree, 14 pass only compacted and 4 only uncompacted. McNemar exact two-sided p = 0.031. The paired difference is +4.5pp with a Wald 95% CI of +0.8 to +8.1pp, so the honest claim is an improvement of roughly 1 to 8 points, not the point estimate.
- Citation MAP, macro-averaged over conversations with 208 of 224 turns eligible (turns with gold passages) in each arm: 0.4174 compacted vs 0.4230 uncompacted. The gold-prefix 0.35 is over 208 of 224 eligible cases.
- Refusal precision and recall against the answerability labels (16 UNANSWERABLE turns per arm): compacted 0.33 precision and 0.44 recall (21 refusals), uncompacted 0.23 and 0.31 (22 refusals). Gold-prefix: 0.24 and 0.44 (29 refusals).

## Methodology

### Retrieval Metrics

**Mean Average Precision (MAP)** scores ranked retrieval results against the gold `expected_uris`.

- For each relevant document at position k, calculate precision@k = (relevant docs in top k) / k
- Average Precision (AP) = sum of these precision values / total relevant documents
- MAP is the mean of AP scores across all queries
- Range: 0 to 1. Rewards ranking relevant documents higher
- For single-doc queries this collapses to `1/rank` (i.e. reciprocal rank)

### QA Accuracy

`pydantic-evals` coordinates an LLM judge to determine whether the capability's answer is correct. The default judge is `ollama:qwen3.8`, pinned so changes to the capability model don't change the judge underneath. Set `evaluations.judge` in `haiku.rag.yaml` to override (including a custom `base_url` for any OpenAI-compatible endpoint). Accuracy is the fraction of correctly answered questions.

A dataset that brings its own deterministic evaluator is scored by that evaluator instead, and no judge runs. T²-RAGBench is the only such dataset today, scored by `NumberMatchEvaluator`.

`qwen3.8` replaced `qwen3.6` after a 120-case calibration on ORB, stratified 60 pass / 60 fail: agreement 0.950, Cohen's κ 0.900, and in all 6 disagreements it matched or beat `qwen3.6` (4 were `qwen3.6` failing answers that were equivalent in different notation). It emits no reasoning content, so it avoids the thinking spirals that made `qwen3.6` exceed its output budget and drop verdicts. `reasoning_effort` changes its verdicts in 1 case per 120, so the cheaper `low` is pinned.

Before that, we picked `qwen3.6` over the previously-pinned `gpt-oss` after a 4-cell calibration (gpt-oss / qwen3.6 as both answerer and judge, with Claude Opus 4.7 as a reference). `qwen3.6` had κ ≥ 0.66 vs the reference on both same-family and cross-family answerers (vs ~0.39–0.55 for `gpt-oss`) and showed no measurable self-preference bias, while `gpt-oss` was ~10 pp more lenient on its own outputs.

### Citation Retrieval

Alongside QA accuracy, a second metric scores the URIs the capability registered via the `cite` tool against each dataset's gold `expected_uris`, using the same MAP math as raw retrieval. The score key is `cited_map`. Console output also includes the cite rate (% of cases with at least one citation) and the mean number of citations per case.

This is computed alongside QA accuracy from the same capability run, no extra invocations. The signal complements raw retrieval: where raw retrieval measures whether the retriever surfaced the gold document at any rank, citation retrieval measures whether the capability grounded its answer on it.

## Running Evaluations

You can run evaluations with the `evaluations` CLI:

```bash
evaluations run hotpotqa
evaluations run orb_text
```

The evaluation flow is orchestrated with [`pydantic-evals`](https://github.com/pydantic/pydantic-ai/tree/main/libs/pydantic-evals), which we leverage for dataset management, scoring, and report generation.

### Pre-built Databases

Building evaluation databases from scratch can take a long time, especially for large datasets like OpenRAG Bench. Pre-built databases are available on HuggingFace:

```bash
# Download a specific dataset
evaluations download hotpotqa

# Download all datasets
evaluations download all

# Force re-download (overwrite existing)
evaluations download hotpotqa --force
```

Active datasets:

| Dataset | Size |
|---------|------|
| `orb_text` — OpenRAG Bench, text embedder (`qwen3-embedding:4b`) with VLM picture descriptions baked into chunk content | ~18 GB |
| `orb_multimodal` — OpenRAG Bench, multimodal embedder (`qwen3-vl-embedding-8b`); picture vectors live in the same space as text for cross-modal retrieval | ~16 GB |
| `orb_multimodal_nemotron` — OpenRAG Bench, multimodal embedder (`nvidia/llama-nemotron-embed-vl-1b-v2`), the embedder behind the published headline results | ~16 GB |
| `t2_finqa` — T²-RAGBench (FinQA) financial QA, text embedder (`qwen3-embedding:4b`); scored by exact numeric match, run with `--target analysis-capability` | ~2 GB |
| `hotpotqa` — HotpotQA multi-hop QA over Wikipedia paragraphs, text embedder (`qwen3-embedding:4b`) | ~1.5 GB |
| `mtrag_clapnq` — MTRAG multi-turn RAG, ClapNQ (Wikipedia) passages, text embedder (`qwen3-embedding:4b`); also serves the `mtrag_clapnq_rewrite`, `mtrag_clapnq_live` and `mtrag_clapnq_live_uncompacted` keys | ~2.8 GB |

After downloading, run benchmarks with `--skip-db`. Each database is built with a specific embedder, so pass its reference config from `evaluations/configs/` (a database only opens against a config whose embedder matches):

```bash
evaluations run orb_multimodal_nemotron --skip-db --config configs/orb_multimodal_nemotron.yaml
```

The configs use `vllm` as the model host. Point `base_url` at your own OpenAI-compatible endpoints to reproduce the numbers.

### Configuration

The benchmark script accepts several options:

```bash
evaluations run hotpotqa --config /path/to/haiku.rag.yaml --db /path/to/custom.lancedb
```

**Options:**

- `--config PATH` - Specify a custom `haiku.rag.yaml` configuration file
- `--db PATH` - Override the database path (default: platform-specific user data directory)
- `--skip-db` - Skip updating the evaluation database
- `--skip-retrieval` - Skip retrieval benchmark
- `--skip-qa` - Skip QA benchmark
- `--limit N` - Limit number of test cases
- `--name NAME` - Override the evaluation name
- `--target {rag-capability,analysis-capability}` - Choose which [capability](capabilities/index.md) to benchmark end-to-end (default: `rag-capability`). The target names remain stable dataset identifiers.
- `--capability-model PROVIDER:NAME` - Override the capability model independently from the judge (default: `config.qa.model`, or `config.analysis.model` when set for `--target analysis-capability`).
- `--filter CLAUSE` / `-f CLAUSE` - Restrict every benchmark search to a subset of the database (see [Restricting the corpus](#restricting-the-corpus)).

If no config file is specified, the script searches standard locations: `./haiku.rag.yaml`, user config directory, then falls back to defaults.

To pin the LLM judge in YAML (rather than the default `ollama:qwen3.8`). These are the recommended settings:

```yaml
evaluations:
  judge:
    provider: openai
    name: Inferact/Qwen3.8-27B-NVFP4
    base_url: http://localhost:8000/v1   # optional, for OpenAI-compatible servers (vLLM, LM Studio, etc.)
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

When a database holds documents from several corpora — only some of which a dataset's questions are drawn from — `--filter` restricts every benchmark search to a subset. It takes the same SQL `WHERE` clause as `haiku-rag search --filter`, over document columns (`id`, `uri`, `title`, `created_at`, `updated_at`, `metadata`). Each dataset writes its own URIs: `orb_text` uses bare arXiv ids such as `2407.01528v3`, `hotpotqa` uses page titles.

```bash
evaluations run orb_text --skip-db --config haiku.rag.s3.yaml \
  --filter "uri LIKE '2407%'"
```

If the corpora are distinguished by a tag rather than by URI, attach it at ingest time as document metadata and match it with `LIKE`. `metadata` is stored as a `json.dumps` string, so there is no JSON subfield access — match the serialized key/value, including the space after the colon:

```bash
evaluations run orb_text --skip-db --filter "metadata LIKE '%\"corpus\": \"orb_text\"%'"
```

The clause applies to both benchmark phases — the retrieval benchmark's searches and every search the capability runs during QA — so the two score the same subset. It is recorded as `document_filter` in the run's experiment metadata, so a filtered run is never mistaken for an unfiltered one when comparing results.

Filtering affects searches only — a run without `--skip-db` still populates the database with the dataset's full corpus.

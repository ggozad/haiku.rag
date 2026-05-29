# Benchmarks

We evaluate `haiku.rag` on a small set of datasets that exercise different parts of the pipeline. Wix and OpenRAG Bench (ORB) are the two we currently track. Retrieval, QA accuracy, and citation retrieval are scored end-to-end through the rag and rag-analysis skills.

## Running Evaluations

You can run evaluations with the `evaluations` CLI:

```bash
evaluations run wix
evaluations run orb_text
```

The evaluation flow is orchestrated with [`pydantic-evals`](https://github.com/pydantic/pydantic-ai/tree/main/libs/pydantic-evals), which we leverage for dataset management, scoring, and report generation.

### Pre-built Databases

Building evaluation databases from scratch can take a long time, especially for large datasets like OpenRAG Bench. Pre-built databases are available on HuggingFace:

```bash
# Download a specific dataset
evaluations download wix

# Download all datasets
evaluations download all

# Force re-download (overwrite existing)
evaluations download wix --force
```

Active datasets:

| Dataset | Size |
|---------|------|
| `wix` | ~511MB |
| `orb_text` — OpenRAG Bench, text embedder (`qwen3-embedding:4b`) with VLM picture descriptions baked into chunk content | ~18 GB |
| `orb_multimodal` — OpenRAG Bench, multimodal embedder (`qwen3-vl-embedding-8b`); picture vectors live in the same space as text for cross-modal retrieval | ~16 GB |

Inactive (kept downloadable, not currently maintained):

| Dataset | Size |
|---------|------|
| `repliqa` | ~30MB |
| `hotpotqa` | ~331MB |

After downloading, run benchmarks with `--skip-db` to use the pre-built database:

```bash
evaluations run wix --skip-db
```

### Configuration

The benchmark script accepts several options:

```bash
evaluations run wix --config /path/to/haiku.rag.yaml --db /path/to/custom.lancedb
```

**Options:**

- `--config PATH` - Specify a custom `haiku.rag.yaml` configuration file
- `--db PATH` - Override the database path (default: platform-specific user data directory)
- `--skip-db` - Skip updating the evaluation database
- `--skip-retrieval` - Skip retrieval benchmark
- `--skip-qa` - Skip QA benchmark
- `--limit N` - Limit number of test cases
- `--name NAME` - Override the evaluation name
- `--target {rag-skill,analysis-skill}` - Choose which [skill](skills/index.md) to benchmark end-to-end (default: `rag-skill`).
- `--skill-model PROVIDER:NAME` - Override the skill model independently from the judge (default: `config.qa.model`, or `config.analysis.model` when set for `--target analysis-skill`).

If no config file is specified, the script searches standard locations: `./haiku.rag.yaml`, user config directory, then falls back to defaults.

To pin the LLM judge in YAML (rather than the default `ollama:qwen3.6`):

```yaml
evaluations:
  judge:
    provider: openai
    name: gpt-4o-mini
    base_url: http://localhost:8000/v1   # optional, for OpenAI-compatible servers (vLLM, LM Studio, etc.)
```

## Methodology

### Retrieval Metrics

**Mean Reciprocal Rank (MRR)** - Used when each query has exactly one relevant document.

- For each query, find the rank (position) of the first relevant document in top-K results
- Reciprocal rank = `1/rank` (e.g., rank 3 → 1/3 ≈ 0.333)
- If not found in top-K, score is 0
- MRR is the mean across all queries
- Range: 0 (never found) to 1 (always at rank 1)

**Mean Average Precision (MAP)** - Used when queries have multiple relevant documents.

- For each relevant document at position k, calculate precision@k = (relevant docs in top k) / k
- Average Precision (AP) = mean of these precision values / total relevant documents
- MAP is the mean of AP scores across all queries
- Range: 0 to 1. Rewards ranking relevant documents higher

### QA Accuracy

`pydantic-evals` coordinates an LLM judge to determine whether the skill's answer is correct. The default judge is `ollama:qwen3.6`, pinned so changes to the skill model don't change the judge underneath. Set `evaluations.judge` in `haiku.rag.yaml` to override (including a custom `base_url` for any OpenAI-compatible endpoint). Accuracy is the fraction of correctly answered questions.

We picked `qwen3.6` over the previously-pinned `gpt-oss` after a 4-cell calibration (gpt-oss / qwen3.6 as both answerer and judge, with Claude Opus 4.7 as a reference). `qwen3.6` had κ ≥ 0.66 vs the reference on both same-family and cross-family answerers (vs ~0.39–0.55 for `gpt-oss`) and showed no measurable self-preference bias, while `gpt-oss` was ~10 pp more lenient on its own outputs.

### Citation Retrieval

Alongside QA accuracy, a second metric scores the URIs the skill registered via the `cite` tool against each dataset's gold `expected_uris`, using the same MRR / MAP math as raw retrieval. The score key is `cited_mrr` for single-doc datasets and `cited_map` for multi-doc. Console output also includes the cite rate (% of cases with at least one citation) and the mean number of citations per case.

This is computed alongside QA accuracy from the same skill run, no extra invocations. The signal complements raw retrieval: where raw retrieval measures whether the retriever surfaced the gold document at any rank, citation retrieval measures whether the skill grounded its answer on it.

## Current results

Numbers measured under the current pinned judge (`ollama:qwen3.6`) on a recent `haiku.rag` version.

### Wix

[WixQA](https://huggingface.co/datasets/Wix/WixQA) is real customer support questions paired with curated answers. 200 cases.

`evaluations run wix --target rag-skill` runs the RAG skill end-to-end and produces both QA accuracy and a citation retrieval metric (`cited_map`) computed from the URIs the skill registered via the `cite` tool against the gold `expected_uris`.

| Skill model                  | Reranker               | QA accuracy | Mean `cited_map` |
|------------------------------|------------------------|-------------|------------------|
| `vllm:Gemma-4-26B-A4B-NVFP4` | `mxbai-rerank-base-v2` | 0.87        | 0.38             |

*Measured on haiku.rag v0.48.0 with `qwen3-embedding:4b` (vLLM, dim 2560), `chunk_size=256`, `search.limit=5`. Judged by `vllm:Qwen3.6-35B-A3B-NVFP4` (qwen3.6 family, NVFP4 quant served via vLLM rather than the default Ollama). 172 / 198 completed cases (2 errored).*

### OpenRAG Bench (ORB)

[OpenRAG Bench](https://huggingface.co/datasets/vectara/open_ragbench) contains ArXiv research papers with multimodal question-answering pairs. Queries include both text-based and image-based questions, testing retrieval and reasoning over visual content like figures, charts, and diagrams. Each query maps to one relevant document.

Two approaches are benchmarked separately:

- **Multimodal embedder** (`Qwen/Qwen3-VL-Embedding-8B`, served via vLLM): picture bytes and text live in a shared vector space, no VLM is run at ingest.
- **Text embedder + VLM picture descriptions** (`qwen3-embedding:4b` + `ollama/ministral-3`): pictures are described at ingest and the descriptions are woven into chunk text. Retrieval runs over text only. See [Picture handling configuration](configuration/processing.md#picture-handling).

#### Multimodal embedder

##### Retrieval (MAP)

| Embedding Model                          | Cases | MAP    |
|------------------------------------------|------:|-------:|
| `Qwen/Qwen3-VL-Embedding-8B`             |  3045 | 0.9774 |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   |  3045 | 0.9709 |

##### QA Accuracy

| Embedding Model              | Skill model                  | Reranker               | Cases | Accuracy |
|------------------------------|------------------------------|------------------------|------:|---------:|
| `Qwen/Qwen3-VL-Embedding-8B` | `vllm:Gemma-4-26B-A4B-NVFP4` | `mxbai-rerank-base-v2` |  1409 |   88.7 % |

#### Text embedder + VLM picture descriptions

##### Retrieval (MAP)

| Embedding Model                          | VLM                  | Reranker               | Cases | MAP    |
|------------------------------------------|----------------------|------------------------|------:|-------:|
| `qwen3-embedding:4b`                     | Ollama / ministral-3 | none                   |  3045 | 0.9722 |
| `qwen3-embedding:4b`                     | Ollama / ministral-3 | `mxbai-rerank-base-v2` |  3045 | 0.9834 |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | Ollama / ministral-3 | `mxbai-rerank-base-v2` |  3045 | 0.9863 |

*qwen3 no-reranker measured on haiku.rag v0.45.0; qwen3 + reranker and nemotron on v0.50.0.*

##### QA accuracy + citation retrieval

| Embedding Model                          | VLM                  | Skill model                  | Cases | QA accuracy | Mean `cited_map` |
|------------------------------------------|----------------------|------------------------------|------:|-------------|------------------|
| `qwen3-embedding:4b`                     | Ollama / ministral-3 | `vllm:Gemma-4-26B-A4B-NVFP4` |  3045 | 0.88        | 0.89             |
| `nvidia/llama-nemotron-embed-vl-1b-v2`   | Ollama / ministral-3 | `vllm:Gemma-4-26B-A4B-NVFP4` |  2836 | 0.96        | 0.81             |

*qwen3 measured on haiku.rag v0.48.0 over all 3045 cases; nemotron on v0.50.0 over 2836 / 3045 cases (run stopped early; numbers stable from ~12% onward). Both with `mxbai-rerank-base-v2`, judged by `vllm:Qwen3.6-35B-A3B-NVFP4`.*

## Inactive datasets

The benchmarks below are not currently maintained. Numbers were measured against earlier `haiku.rag` versions and an older pinned judge (`ollama:gpt-oss`), before the skill workflow became the only path. Retrieval tables don't depend on the judge, but the QA tables aren't reproducible against the current skill-only setup. We may revive them.

### RepliQA

[RepliQA](https://huggingface.co/datasets/ServiceNow/repliqa) contains synthetic news stories with question-answer pairs. We use `News Stories` from `repliqa_3` (1035 documents). Each question has exactly one relevant document, so we use MRR for retrieval evaluation.

#### Retrieval (MRR)

| Embedding Model               | MRR  | Reranker |
|-------------------------------|------|----------|
| Ollama / `qwen3-embedding:8b` | 0.91 | -        |

*Measured on haiku.rag v0.19.6.*

#### QA Accuracy

| Embedding Model              | QA Model                         | Accuracy | Reranker               |
|------------------------------|----------------------------------|----------|------------------------|
| Ollama / `qwen3-embedding:4b`   | Ollama / `gpt-oss` - no thinking | 0.82     | None                   |
| Ollama / `qwen3-embedding:8b`   | Ollama / `gpt-oss` - thinking    | 0.89     | None                   |
| Ollama / `mxbai-embed-large`    | Ollama / `qwen3` - thinking      | 0.85     | None                   |
| Ollama / `mxbai-embed-large`    | Ollama / `qwen3` - thinking      | 0.87     | `mxbai-rerank-base-v2` |
| Ollama / `mxbai-embed-large`    | Ollama / `qwen3:0.6b`            | 0.28     | None                   |

*Measured on haiku.rag v0.19.6, judged by `ollama:gpt-oss`.*

Note the significant degradation when very small models are used such as `qwen3:0.6b`.

### HotpotQA

[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) is a multi-hop question answering dataset requiring reasoning over multiple Wikipedia paragraphs. Each question requires evidence from 2+ documents, making it ideal for testing retrieval and reasoning capabilities. We use MAP for retrieval evaluation since queries have multiple relevant documents.

#### Retrieval (MAP)

| Embedding Model      | MAP  | Reranker |
|----------------------|------|----------|
| `qwen3-embedding:4b` | 0.69 | none     |

*Measured on haiku.rag v0.20.2.*

#### QA Accuracy

| Embedding Model      | QA Model                 | Accuracy |
|----------------------|--------------------------|----------|
| `qwen3-embedding:4b` | `gpt-oss:20b` - thinking | 0.86     |

*Measured on haiku.rag v0.20.2, judged by `ollama:gpt-oss`.*

### Wix (historical, plain text and HTML)

Earlier Wix runs measured under different chunk settings and reranker combinations, against the older `gpt-oss` judge.

#### Retrieval (MAP)

| Embedding Model        | Chunk size | MAP  | Reranker               | Notes                        |
|------------------------|------------|------|------------------------|------------------------------|
| `qwen3-embedding:4b`   | 256        | 0.34 | None                   | html, `chunk-radius=2`       |
| `qwen3-embedding:4b`   | 256        | 0.39 | `mxbai-rerank-base-v2` | html, `chunk-radius=2`       |
| `qwen3-embedding:4b`   | 256        | 0.43 | None                   | plain text, `chunk-radius=0` |
| `qwen3-embedding:4b`   | 512        | 0.45 | None                   | plain text, `chunk-radius=0` |

*Measured on haiku.rag v0.27.2.*

#### QA Accuracy

| Embedding Model      | Chunk size | QA Model                    | Accuracy | Notes                        |
|----------------------|------------|-----------------------------|----------|------------------------------|
| `qwen3-embedding:4b` | 256        | `gpt-oss:20b` - no thinking | 0.80     | html, `chunk-radius=2`       |
| `qwen3-embedding:4b` | 256        | `gpt-oss:20b` - no thinking | 0.83     | html, `chunk-radius=2`, `jinaai/jina-reranker-v3` |

*Measured on haiku.rag v0.27.2, judged by `ollama:gpt-oss`.*

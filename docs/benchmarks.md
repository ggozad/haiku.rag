# Benchmarks

We evaluate `haiku.rag` on several datasets to measure both retrieval quality and question-answering accuracy.

## Running Evaluations

You can run evaluations with the `evaluations` CLI:

```bash
evaluations run repliqa
evaluations run wix
```

The evaluation flow is orchestrated with [`pydantic-evals`](https://github.com/pydantic/pydantic-ai/tree/main/libs/pydantic-evals), which we leverage for dataset management, scoring, and report generation.

### Pre-built Databases

Building evaluation databases from scratch can take a long time, especially for large datasets like OpenRAG Bench. Pre-built databases are available on HuggingFace:

```bash
# Download a specific dataset
evaluations download repliqa

# Download all datasets
evaluations download all

# Force re-download (overwrite existing)
evaluations download repliqa --force
```

Available datasets:

| Dataset | Size |
|---------|------|
| `repliqa` | ~30MB |
| `hotpotqa` | ~331MB |
| `wix` | ~511MB |
| `open_rag_bench` | ~14GB |

After downloading, run benchmarks with `--skip-db` to use the pre-built database:

```bash
evaluations run repliqa --skip-db
```

### Configuration

The benchmark script accepts several options:

```bash
evaluations run repliqa --config /path/to/haiku.rag.yaml --db /path/to/custom.lancedb
```

**Options:**

- `--config PATH` - Specify a custom `haiku.rag.yaml` configuration file
- `--db PATH` - Override the database path (default: platform-specific user data directory)
- `--skip-db` - Skip updating the evaluation database
- `--skip-retrieval` - Skip retrieval benchmark
- `--skip-qa` - Skip QA benchmark
- `--limit N` - Limit number of test cases
- `--name NAME` - Override the evaluation name
- `--judge-model PROVIDER:NAME` - Override the LLM judge model. Defaults to `ollama:qwen3.6` so the judge stays stable when the QA / skill model changes.
- `--target {qa,rag-skill,analysis-skill}` - Choose what to benchmark (default: `qa`). `rag-skill` and `analysis-skill` run the corresponding [skill](skills/index.md) end-to-end against the same datasets and judge as the QA agent.
- `--skill-model PROVIDER:NAME` - Override the skill model independently from the judge (default: `config.qa.model`). Only valid with skill targets.

If no config file is specified, the script searches standard locations: `./haiku.rag.yaml`, user config directory, then falls back to defaults.

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
- Range: 0 to 1; rewards ranking relevant documents higher

### QA Accuracy

For question-answering evaluation, `pydantic-evals` coordinates an LLM judge to determine whether answers are correct. The default judge is `ollama:qwen3.6` — pinned so changes to the QA or skill model don't change the judge underneath. Override per run with `--judge-model provider:name`. Accuracy is the fraction of correctly answered questions.

We picked `qwen3.6` over the previously-pinned `gpt-oss` after a 4-cell calibration (gpt-oss / qwen3.6 as both answerer and judge, with Claude Opus 4.7 as a reference). `qwen3.6` had κ ≥ 0.66 vs the reference on both same-family and cross-family answerers (vs ~0.39–0.55 for `gpt-oss`) and showed no measurable self-preference bias, while `gpt-oss` was ~10 pp more lenient on its own outputs.

### Citation Retrieval

When benchmarking a skill (`--target rag-skill` or `--target analysis-skill`), a second metric scores the URIs the skill registered via the `cite` tool against each dataset's gold `expected_uris`, using the same MRR / MAP math as raw retrieval. The score key is `cited_mrr` for single-doc datasets and `cited_map` for multi-doc. Console output also includes the cite rate (% of cases with at least one citation) and the mean number of citations per case.

This is computed alongside QA accuracy from the same skill run — no extra invocations. The signal complements raw retrieval: where raw retrieval measures whether the retriever surfaced the gold document at any rank, citation retrieval measures whether the skill grounded its answer on it.

## Current results

Numbers measured under the current pinned judge (`ollama:qwen3.6`) on a recent `haiku.rag` version.

### Wix

[WixQA](https://huggingface.co/datasets/Wix/WixQA) — real customer support questions paired with curated answers. 200 cases.

#### Skill QA + citation retrieval

`evaluations run wix --target rag-skill` benchmarks the RAG skill end-to-end and produces both QA accuracy and a citation retrieval metric (`cited_map`) computed from the URIs the skill registered via the `cite` tool against the gold `expected_uris`.

| Skill model      | QA accuracy | Mean `cited_map` |
|------------------|-------------|------------------|
| `ollama:gpt-oss` | 0.85        | 0.40             |

*Measured on haiku.rag v0.43.1, judged by `ollama:qwen3.6` (current default), on 199 of 200 completed cases.* 28 % of cases produce a perfect citation (`cited_map` = 1.0).

## Past results

These were measured under the prior pinned judge (`ollama:gpt-oss`). The pinned default has since switched to `ollama:qwen3.6` (see [Methodology — QA Accuracy](#qa-accuracy)) — under the new judge the QA accuracy numbers below typically shift up by ~5–10 pp.

Retrieval tables don't depend on the judge but are kept here because they were measured on the same older `haiku.rag` versions as their accompanying QA tables.

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

### Wix

[WixQA](https://huggingface.co/datasets/Wix/WixQA) — see description above. We benchmark both the plain text version (HTML stripped, no structure) and HTML version. Since HTML chunks are small (typically a phrase), we use `chunk_radius=2` to expand context.

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
| `qwen3-embedding:4b` | 256        | `gpt-oss:20b` - thinking    | 0.82     | html, `chunk-radius=2`       |
| `qwen3-embedding:4b` | 256        | `gpt-oss:20b` - no thinking | 0.80     | html, `chunk-radius=2`       |
| `qwen3-embedding:4b` | 256        | `gpt-oss:20b` - no thinking | 0.83     | html, `chunk-radius=2`, `jinaai/jina-reranker-v3` |

*Measured on haiku.rag v0.27.2, judged by `ollama:gpt-oss`.*

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

### OpenRAG Bench (ORB)

[OpenRAG Bench](https://huggingface.co/datasets/vectara/open_ragbench) contains ArXiv research papers with multimodal question-answering pairs. Queries include both text-based and image-based questions, testing retrieval over visual content like figures, charts, and diagrams. We use MAP for retrieval evaluation since each query maps to one relevant document.

**Multimodal processing**: Picture descriptions are generated using a Vision Language Model (VLM) during document conversion, making embedded images searchable via text queries. See [Picture Description configuration](configuration/processing.md#picture-description-vlm).

#### Retrieval (MAP)

| Embedding Model      | MAP    | VLM                  |
|----------------------|--------|----------------------|
| `qwen3-embedding:4b` | 0.9626 | Ollama / ministral-3 |

*Measured on haiku.rag v0.26.8.*

#### QA Accuracy

| Embedding Model      | QA Model                    | Accuracy | VLM                  |
|----------------------|-----------------------------|----------|----------------------|
| `qwen3-embedding:4b` | `gpt-oss:20b` - no thinking | 0.912    | Ollama / ministral-3 |

*Measured on haiku.rag v0.26.8, judged by `ollama:gpt-oss`.*

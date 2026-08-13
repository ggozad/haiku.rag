# Benchmarks

We evaluate `haiku.rag` on a small set of datasets that exercise different parts of the pipeline. OpenRAG Bench (ORB), T²-RAGBench, HotpotQA, and Wix are the datasets we currently track. Retrieval, QA accuracy, and citation retrieval are scored end-to-end through the RAG and analysis capabilities.

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
| `orb_multimodal_nemotron` — OpenRAG Bench, multimodal embedder (`nvidia/llama-nemotron-embed-vl-1b-v2`), the embedder behind the published headline results | ~16 GB |
| `t2_finqa` — T²-RAGBench (FinQA) financial QA, text embedder (`qwen3-embedding:4b`); scored by exact numeric match, run with `--target analysis-capability` | ~2 GB |
| `hotpotqa` — HotpotQA multi-hop QA over Wikipedia paragraphs, text embedder (`qwen3-embedding:4b`) | ~1.5 GB |

After downloading, run benchmarks with `--skip-db`. Each database is built with a specific embedder, so pass its reference config from `evaluations/configs/` (a database only opens against a config whose embedder matches):

```bash
evaluations run orb_multimodal_nemotron --skip-db --config configs/orb_multimodal_nemotron.yaml
```

The configs use `vllm` as the model host. Point `base_url` at your own OpenAI-compatible endpoints to reproduce the numbers.

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
- `--target {rag-capability,analysis-capability}` - Choose which [capability](capabilities/index.md) to benchmark end-to-end (default: `rag-capability`). The target names remain stable dataset identifiers.
- `--capability-model PROVIDER:NAME` - Override the capability model independently from the judge (default: `config.qa.model`, or `config.analysis.model` when set for `--target analysis-capability`).

If no config file is specified, the script searches standard locations: `./haiku.rag.yaml`, user config directory, then falls back to defaults.

To pin the LLM judge in YAML (rather than the default `ollama:qwen3.6`). These are the recommended settings:

```yaml
evaluations:
  judge:
    provider: openai
    name: RedHatAI/Qwen3.6-35B-A3B-NVFP4
    base_url: http://localhost:8000/v1   # optional, for OpenAI-compatible servers (vLLM, LM Studio, etc.)
    temperature: 0.6
    max_tokens: 16384
    extra_body:
      top_p: 0.95
      top_k: 20
      min_p: 0
      chat_template_kwargs:
        enable_thinking: true
```

## Methodology

### Retrieval Metrics

**Mean Average Precision (MAP)** scores ranked retrieval results against the gold `expected_uris`.

- For each relevant document at position k, calculate precision@k = (relevant docs in top k) / k
- Average Precision (AP) = sum of these precision values / total relevant documents
- MAP is the mean of AP scores across all queries
- Range: 0 to 1. Rewards ranking relevant documents higher
- For single-doc queries this collapses to `1/rank` (i.e. reciprocal rank)

### QA Accuracy

`pydantic-evals` coordinates an LLM judge to determine whether the capability's answer is correct. The default judge is `ollama:qwen3.6`, pinned so changes to the capability model don't change the judge underneath. Set `evaluations.judge` in `haiku.rag.yaml` to override (including a custom `base_url` for any OpenAI-compatible endpoint). Accuracy is the fraction of correctly answered questions.

A dataset that brings its own deterministic evaluator is scored by that evaluator instead, and no judge runs. T²-RAGBench is the only such dataset today, scored by `NumberMatchEvaluator`.

We picked `qwen3.6` over the previously-pinned `gpt-oss` after a 4-cell calibration (gpt-oss / qwen3.6 as both answerer and judge, with Claude Opus 4.7 as a reference). `qwen3.6` had κ ≥ 0.66 vs the reference on both same-family and cross-family answerers (vs ~0.39–0.55 for `gpt-oss`) and showed no measurable self-preference bias, while `gpt-oss` was ~10 pp more lenient on its own outputs.

### Citation Retrieval

Alongside QA accuracy, a second metric scores the URIs the capability registered via the `cite` tool against each dataset's gold `expected_uris`, using the same MAP math as raw retrieval. The score key is `cited_map`. Console output also includes the cite rate (% of cases with at least one citation) and the mean number of citations per case.

This is computed alongside QA accuracy from the same capability run, no extra invocations. The signal complements raw retrieval: where raw retrieval measures whether the retriever surfaced the gold document at any rank, citation retrieval measures whether the capability grounded its answer on it.

## Current results

Numbers measured under the current pinned judge (`ollama:qwen3.6`) on a recent `haiku.rag` version.

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

### Wix

[WixQA](https://huggingface.co/datasets/Wix/WixQA) is real customer support questions paired with curated answers. 200 cases.

`evaluations run wix --target rag-capability` runs the RAG capability end-to-end and produces both QA accuracy and a citation retrieval metric (`cited_map`) computed from the URIs the capability registered via the `cite` tool against the gold `expected_uris`.

| Capability model                  | Reranker               | QA accuracy | Mean `cited_map` |
|------------------------------|------------------------|-------------|------------------|
| `vllm:Gemma-4-26B-A4B-NVFP4` | `mxbai-rerank-base-v2` | 0.87        | 0.38             |

*Measured on haiku.rag v0.48.0 with `qwen3-embedding:4b` (vLLM, dim 2560), `chunk_size=256`, `search.limit=5`. Judged by `vllm:Qwen3.6-35B-A3B-NVFP4` (qwen3.6 family, NVFP4 quant served via vLLM rather than the default Ollama). 172 / 198 completed cases (2 errored).*

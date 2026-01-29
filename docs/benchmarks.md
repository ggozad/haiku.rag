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
- `--deep` - Use deep QA mode (multi-step reasoning with research graph)

If no config file is specified, the script searches standard locations: `./haiku.rag.yaml`, user config directory, then falls back to defaults.

### Deep QA Mode

The `--deep` flag enables multi-step reasoning using the research graph instead of the simple QA agent:

```bash
evaluations run repliqa --skip-db --deep
```

In deep mode:

- Questions are decomposed into sub-questions by a planning agent
- Each sub-question is answered by searching the knowledge base
- A synthesis agent combines findings into a comprehensive answer
- The graph runs for up to 2 iterations with no early exit (confidence threshold disabled)

This matches the behavior of `haiku-rag ask --deep` in the CLI. Deep mode typically produces more thorough answers but requires more LLM calls per question.

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

For question-answering evaluation, `pydantic-evals` coordinates an LLM judge (Ollama `qwen3`) to determine whether answers are correct. Accuracy is the fraction of correctly answered questions.

## RepliQA

[RepliQA](https://huggingface.co/datasets/ServiceNow/repliqa) contains synthetic news stories with question-answer pairs. We use `News Stories` from `repliqa_3` (1035 documents). Each question has exactly one relevant document, so we use MRR for retrieval evaluation.

*Results from v0.19.6*

### Retrieval (MRR)

| Embedding Model               | MRR  | Reranker |
|-------------------------------|------|----------|
| Ollama / `qwen3-embedding:8b` | 0.91 | -        |

### QA Accuracy

| Embedding Model              | QA Model                         | Accuracy | Reranker               |
|------------------------------|----------------------------------|----------|------------------------|
| Ollama / `qwen3-embedding:4b`   | Ollama / `gpt-oss` - no thinking | 0.82     | None                   |
| Ollama / `qwen3-embedding:8b`   | Ollama / `gpt-oss` - thinking    | 0.89     | None                   |
| Ollama / `mxbai-embed-large`    | Ollama / `qwen3` - thinking      | 0.85     | None                   |
| Ollama / `mxbai-embed-large`    | Ollama / `qwen3` - thinking      | 0.87     | `mxbai-rerank-base-v2` |
| Ollama / `mxbai-embed-large`    | Ollama / `qwen3:0.6b`            | 0.28     | None                   |

Note the significant degradation when very small models are used such as `qwen3:0.6b`.

## Wix

[WixQA](https://huggingface.co/datasets/Wix/WixQA) contains real customer support questions paired with curated answers from Wix. The benchmark follows the evaluation protocol from the [WixQA paper](https://arxiv.org/abs/2505.08643). Each query can have multiple relevant passages, so we use MAP for retrieval evaluation.

We benchmark both the plain text version (HTML stripped, no structure) and HTML version. Since HTML chunks are small (typically a phrase), we use `chunk_radius=2` to expand context.

*Results from v0.20.0*

### Retrieval (MAP)

| Embedding Model        | Chunk size | MAP  | Reranker               | Notes                        |
|------------------------|------------|------|------------------------|------------------------------|
| `qwen3-embedding:4b`   | 256        | 0.34 | None                   | html, `chunk-radius=2`       |
| `qwen3-embedding:4b`   | 256        | 0.39 | `mxbai-rerank-base-v2` | html, `chunk-radius=2`       |
| `qwen3-embedding:4b`   | 256        | 0.43 | None                   | plain text, `chunk-radius=0` |
| `qwen3-embedding:4b`   | 512        | 0.45 | None                   | plain text, `chunk-radius=0` |

### QA Accuracy

| Embedding Model      | Chunk size | QA Model                    | Accuracy | Notes                        |
|----------------------|------------|-----------------------------|----------|------------------------------|
| `qwen3-embedding:4b` | 256        | `gpt-oss:20b` - no thinking | 0.74     | plain text, `chunk-radius=0` |
| `qwen3-embedding:4b` | 256        | `gpt-oss:20b` - thinking    | 0.79     | html, `chunk-radius=2`       |
| `qwen3-embedding:4b` | 256        | `gpt-oss:20b` - thinking    | 0.80     | html, `chunk-radius=2`, reranker=`mxbai-rerank-base-v2` |

## HotpotQA

[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) is a multi-hop question answering dataset requiring reasoning over multiple Wikipedia paragraphs. Each question requires evidence from 2+ documents, making it ideal for testing retrieval and reasoning capabilities. We use MAP for retrieval evaluation since queries have multiple relevant documents.

*Results from v0.20.2*

### Retrieval (MAP)

| Embedding Model      | MAP  | Reranker |
|----------------------|------|----------|
| `qwen3-embedding:4b` | 0.69 | none     |

### QA Accuracy

| Embedding Model      | QA Model                 | Accuracy |
|----------------------|--------------------------|----------|
| `qwen3-embedding:4b` | `gpt-oss:20b` - thinking | 0.86     |

## OpenRAG Bench (ORB)

[OpenRAG Bench](https://huggingface.co/datasets/vectara/open_ragbench) contains ArXiv research papers with multimodal question-answering pairs. Queries include both text-based and image-based questions, testing retrieval over visual content like figures, charts, and diagrams. We use MAP for retrieval evaluation since each query maps to one relevant document.

**Multimodal processing**: Picture descriptions are generated using a Vision Language Model (VLM) during document conversion, making embedded images searchable via text queries. See [Picture Description configuration](configuration/processing.md#picture-description-vlm).

*Results from v0.26.8*

### Retrieval (MAP)

| Embedding Model      | MAP    | VLM                  |
|----------------------|--------|----------------------|
| `qwen3-embedding:4b` | 0.9626 | Ollama / ministral-3 |

### QA Accuracy

| Embedding Model      | QA Model                    | Accuracy | VLM                  |
|----------------------|-----------------------------|----------|----------------------|
| `qwen3-embedding:4b` | `gpt-oss:20b` - no thinking | 0.912    | Ollama / ministral-3 |

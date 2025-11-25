# Benchmarks

We use the [repliqa](https://huggingface.co/datasets/ServiceNow/repliqa) dataset for the evaluation of `haiku.rag`.

You can perform your own evaluations with the `evaluations` CLI command:

```bash
evaluations repliqa
```

The evaluation flow is orchestrated with
[`pydantic-evals`](https://github.com/pydantic/pydantic-ai/tree/main/libs/pydantic-evals),
which we leverage for dataset management, scoring, and report generation.

## Configuration

The benchmark script accepts several options:

```bash
evaluations repliqa --config /path/to/haiku.rag.yaml --db /path/to/custom.lancedb
```

**Configuration options:**
- `--config PATH` - Specify a custom `haiku.rag.yaml` configuration file
- `--db PATH` - Override the database path (default: `~/.local/share/haiku.rag/evaluations/dbs/{dataset}.lancedb` on Linux, `~/Library/Application Support/haiku.rag/evaluations/dbs/{dataset}.lancedb` on macOS)
- `--skip-db` - Skip updating the evaluation database
- `--skip-retrieval` - Skip retrieval benchmark
- `--skip-qa` - Skip QA benchmark
- `--limit N` - Limit number of test cases for both retrieval and QA
- `--name NAME` - Override the evaluation name (defaults to `{dataset}_retrieval_evaluation` or `{dataset}_qa_evaluation`)

If no config file is specified, the script will search for a config file in the standard locations:
1. `./haiku.rag.yaml` (current directory)
2. User config directory
3. Falls back to default configuration

## RepliQA Retrieval

We use the [RepliQA](https://huggingface.co/datasets/ServiceNow/repliqa) dataset to evaluate retrieval performance. We load the `News Stories` from `repliqa_3` (1035 documents) and index them. Subsequently, we run a search over the `question` field for each row of the dataset and check whether we match the document that answers the question. Questions for which the answer cannot be found in the documents are ignored.

For RepliQA, we use **Mean Reciprocal Rank (MRR)** as the primary metric since each query has exactly one relevant document.

**How MRR is calculated:**
- For each query, we retrieve the top-K documents and find the rank (position) of the first relevant document
- The reciprocal rank for that query is `1/rank` (e.g., if the relevant document is at position 3, the score is 1/3 ≈ 0.333)
- If no relevant document is found in the top-K results, the score is 0
- MRR is the mean of these reciprocal ranks across all queries
- Scores range from 0 (never found) to 1 (always found at rank 1)

**Example:** If we run 3 queries and the relevant documents are found at positions 1, 2, and not found:
- Query 1: 1/1 = 1.0
- Query 2: 1/2 = 0.5
- Query 3: 0 (not found)
- MRR = (1.0 + 0.5 + 0) / 3 = 0.5

### MRR Results

| Embedding Model                       | MRR   | Reranker               |
|---------------------------------------|-------|------------------------|
| Ollama / `qwen3-embedding:8b`         | 0.91  | -                      |

## Question/Answer evaluation

Again using the same dataset, we use a QA agent to answer the question.
`pydantic-evals` runs each case and coordinates an LLM judge (Ollama `qwen3`) to
determine whether the answer is correct. The obtained accuracy is as follows:

| Embedding Model                    | QA Model                          | Accuracy  | Reranker               |
|------------------------------------|-----------------------------------|-----------|------------------------|
| Ollama / `qwen3-embedding:8b`      | Ollama / `gpt-oss`                | 0.93      | None                   |
| Ollama / `qwen3-embedding:0.6b`    | Ollama / `gpt-oss`                | 0.89      | None                   |
| Ollama / `mxbai-embed-large`       | Ollama / `qwen3`                  | 0.85      | None                   |
| Ollama / `mxbai-embed-large`       | Ollama / `qwen3`                  | 0.87      | `mxbai-rerank-base-v2` |
| Ollama / `mxbai-embed-large`       | Ollama / `qwen3:0.6b`             | 0.28      | None                   |

Note the significant degradation when very small models are used such as `qwen3:0.6b`.

## Wix Retrieval

We also track retrieval performance on [WixQA](https://huggingface.co/datasets/Wix/WixQA),
a dataset of real customer support questions paired with curated answers from
Wix. The benchmark follows the evaluation protocol described in the
[WixQA paper](https://arxiv.org/abs/2505.08643) and gives us a view into how the
system handles conversational, product-specific support queries.

For retrieval evaluation, we index the reference answer passages shipped with the dataset and
run retrieval against each user question. Each sample supplies one or more
relevant passage URIs.

For Wix, we use **Mean Average Precision (MAP)** as the primary metric since each query has multiple relevant documents. MAP accounts for both the presence and ranking of all relevant documents.

**How MAP is calculated:**
- For each query, we retrieve the top-K documents and identify which ones are relevant
- For each relevant document found at position k, we calculate precision@k = (number of relevant docs in top k) / k
- Average Precision (AP) for that query is the mean of these precision values, divided by the total number of relevant documents
- MAP is the mean of AP scores across all queries
- Scores range from 0 (no relevant documents found) to 1 (all relevant documents ranked at the top)

**Example:** If a query has 2 relevant documents (A and B), and we retrieve 5 documents [A, X, B, Y, Z]:
- A is at position 1: precision@1 = 1/1 = 1.0 (1 relevant out of top 1)
- B is at position 3: precision@3 = 2/3 ≈ 0.667 (2 relevant out of top 3)
- AP = (1.0 + 0.667) / 2 = 0.833
- If we had another query with AP = 0.5, then MAP = (0.833 + 0.5) / 2 = 0.667

MAP rewards systems that rank relevant documents higher, not just finding them.

### MAP Results

| Embedding Model            | MAP   | Reranker               |
|----------------------------|-------|------------------------|
| -                          | -     | -                      |

## QA Accuracy

And for QA accuracy,

| Embedding Model            | QA Model      | Accuracy | Reranker               |
|----------------------------|---------------|----------|------------------------|
| `qwen3-embedding:4b`       | `gpt-oss:20b` | 0.79     | None                   |
| `qwen3-embedding:4b`       | `gpt-oss:20b` | 0.82     | `mxbai-rerank-base-v2` |
| `qwen3-embedding:8b`       | `gpt-oss:20b` | 0.75     | `mxbai-rerank-base-v2` |

# Benchmarks

We use the [repliqa](https://huggingface.co/datasets/ServiceNow/repliqa) dataset for the evaluation of `haiku.rag`.

You can perform your own evaluations with the Typer CLI in
`evaluations/evaluations/benchmark.py`, for example `python -m evaluations.benchmark repliqa`.
The evaluation flow is orchestrated with
[`pydantic-evals`](https://github.com/pydantic/pydantic-ai/tree/main/libs/pydantic-evals),
which we leverage for dataset management, scoring, and report generation.

## Configuration

The benchmark script accepts a `--config` option to specify a custom `haiku.rag.yaml` configuration file:

```bash
python -m evaluations.benchmark repliqa --config /path/to/haiku.rag.yaml
```

If no config file is specified, the script will search for a config file in the standard locations:
1. `./haiku.rag.yaml` (current directory)
2. User config directory
3. Falls back to default configuration

You can also use command-line options:
- `--skip-db` - Skip updating the evaluation database
- `--skip-retrieval` - Skip retrieval benchmark
- `--skip-qa` - Skip QA benchmark
- `--qa-limit N` - Limit number of QA cases to evaluate
- `--name NAME` - Override the evaluation name (defaults to `{dataset}_qa_evaluation`)

## Recall

In order to calculate recall, we load the `News Stories` from `repliqa_3` (1035 documents) and index them. Subsequently, we run a search over the `question` field for each row of the dataset and check whether we match the document that answers the question. Questions for which the answer cannot be found in the documents are ignored.


The recall obtained is ~0.79 for matching in the top result, raising to ~0.91 for the top 3 results with the "bare" default settings (Ollama `qwen3`, `mxbai-embed-large` embeddings, no reranking).

| Embedding Model                       | Document in top 1 | Document in top 3 | Reranker               |
|---------------------------------------|-------------------|-------------------|------------------------|
| Ollama / `qwen3-embedding:8b`         | 0.81              | 0.95              | None                   |
| Ollama / `qwen3-embedding:0.6b`       | 0.77              | 0.97              | None                   |
| Ollama / `qwen3-embedding:8b`         | 0.91              | 0.98              | `mxbai-rerank-base-v2` |
| Ollama / `mxbai-embed-large`          | 0.79              | 0.91              | None                   |
| Ollama / `mxbai-embed-large`          | 0.90              | 0.95              | `mxbai-rerank-base-v2` |
| Ollama / `nomic-embed-text-v1.5`      | 0.74              | 0.90              | None                   |

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

## Wix dataset

We also track retrieval performance on [WixQA](https://huggingface.co/datasets/Wix/WixQA),
a dataset of real customer support questions paired with curated answers from
Wix. The benchmark follows the evaluation protocol described in the
[WixQA paper](https://arxiv.org/abs/2505.08643) and gives us a view into how the
system handles conversational, product-specific support queries.

For retrieval evaluation, we index the reference answer passages shipped with the dataset and
run retrieval against each user question. Each sample supplies one or more
relevant passage URIs. We track two complementary metrics:

- **Recall@K**: Fraction of relevant documents retrieved in top K results. Measures coverage.
- **Success@K**: Fraction of queries with at least one relevant document in top K. Most relevant for RAG, where finding one good document is often sufficient.

### Recall@K Results

| Embedding Model            | Recall@1 | Recall@3 | Recall@5 | Reranker               |
|----------------------------|----------|----------|----------|------------------------|
| `qwen3-embedding:8b`       | 0.31     | 0.48     | 0.54     | None                   |
| `qwen3-embedding:8b`       | 0.36     | 0.57     | 0.68     | `mxbai-rerank-base-v2` |
| `qwen3-embedding:8b`       | 0.36     | 0.58     | 0.67     | `zeroentropy`          |

### Success@K Results

| Embedding Model            | Success@1 | Success@3 | Success@5 | Reranker               |
|----------------------------|-----------|-----------|-----------|------------------------|
| `qwen3-embedding:8b`       | 0.36      | 0.54      | 0.62      | None                   |
| `qwen3-embedding:8b`       | 0.42      | 0.66      | 0.76      | `mxbai-rerank-base-v2` |
| `qwen3-embedding:8b`       | 0.41      | 0.66      | 0.76      | `zeroentropy`          |

## QA Accuracy

And for QA accuracy,

| Embedding Model            | QA Model      | Accuracy | Reranker               |
|----------------------------|---------------|----------|------------------------|
| `qwen3-embedding:4b`       | `gpt-oss:20b` | 0.79     | None                   |
| `qwen3-embedding:8b`       | `gpt-oss:20b` | 0.75     | `mxbai-rerank-base-v2` |

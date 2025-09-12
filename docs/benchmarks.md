# Benchmarks

We use the [repliqa](https://huggingface.co/datasets/ServiceNow/repliqa) dataset for the evaluation of `haiku.rag`.

You can perform your own evaluations using as example the script found at
`tests/generate_benchmark_db.py`.

## Recall

In order to calculate recall, we load the `News Stories` from `repliqa_3` (1035 documents) and index them. Subsequently, we run a search over the `question` field for each row of the dataset and check whether we match the document that answers the question. Questions for which the answer cannot be found in the documents are ignored.


The recall obtained is ~0.79 for matching in the top result, raising to ~0.91 for the top 3 results with the "bare" default settings (Ollama `qwen3`, `mxbai-embed-large` embeddings, no reranking).

| Embedding Model                       | Document in top 1 | Document in top 3 | Reranker               |
|---------------------------------------|-------------------|-------------------|------------------------|
| Ollama / `mxbai-embed-large`          | 0.79              | 0.91              | None                   |
| Ollama / `mxbai-embed-large`          | 0.90              | 0.95              | `mxbai-rerank-base-v2` |
| Ollama / `nomic-embed-text-v1.5`      | 0.74              | 0.90              | None                   |
<!-- | OpenAI / `text-embeddings-3-small`    | 0.75              | 0.88              | None                   |
| OpenAI / `text-embeddings-3-small`    | 0.75              | 0.88              | None                   |
| OpenAI / `text-embeddings-3-small`    | 0.83              | 0.90              | Cohere / `rerank-v3.5` | -->

## Question/Answer evaluation

Again using the same dataset, we use a QA agent to answer the question. In addition we use an LLM judge (using the Ollama `qwen3`) to evaluate whether the answer is correct or not. The obtained accuracy is as follows:

| Embedding Model                    | QA Model                          | Accuracy  | Reranker               |
|------------------------------------|-----------------------------------|-----------|------------------------|
| Ollama / `mxbai-embed-large`       | Ollama / `qwen3`                  | 0.85      | None                   |
| Ollama / `mxbai-embed-large`       | Ollama / `qwen3`                  | 0.87      | `mxbai-rerank-base-v2` |
| Ollama / `mxbai-embed-large`       | Ollama / `qwen3:0.6b`             | 0.28      | None                   |

Note the significant degradation when very small models are used such as `qwen3:0.6b`.
<!-- | Ollama / `mxbai-embed-large`       | Anthropic / `Claude Sonnet 3.7`   | 0.79      | None                   |
| OpenAI / `text-embeddings-3-small` | OpenAI / `gpt-4-turbo`            | 0.62      | None                   | -->

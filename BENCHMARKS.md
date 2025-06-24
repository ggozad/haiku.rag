# `haiku.rag` benchmarks

We use [repliqa](https://huggingface.co/datasets/ServiceNow/repliqa) for the evaluation of `haiku.rag`

* Recall

We load the `News Stories` from `repliqa_3` which is 1035 documents, using `tests/generate_benchmark_db.py`, using the `mxbai-embed-large` Ollama embeddings.

Subsequently, we run a search over the `question` for each row of the dataset and check whether we match the document that answers the question. The recall obtained is ~0.75 for matching in the top result, raising to ~0.75 for the top 3 results.

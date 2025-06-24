from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm

from haiku.rag.client import HaikuRAG


async def populate_db():
    if (Path(__file__).parent / "benchmark.sqlite").exists():
        print("Benchmark database already exists. Skipping creation.")
        return

    ds: Dataset = load_dataset("ServiceNow/repliqa")["repliqa_3"]  # type: ignore
    corpus = ds.filter(lambda doc: doc["document_topic"] == "News Stories")

    async with HaikuRAG(Path(__file__).parent / "benchmark.sqlite") as rag:
        for i, doc in enumerate(tqdm(corpus)):
            await rag.create_document(
                content=doc["document_extracted"],  # type: ignore
                uri=doc["document_id"],  # type: ignore
            )


async def run_match_benchmark():
    ds: Dataset = load_dataset("ServiceNow/repliqa")["repliqa_3"]  # type: ignore
    corpus = ds.filter(lambda doc: doc["document_topic"] == "News Stories")

    correct_at_1 = 0
    correct_at_2 = 0
    correct_at_3 = 0
    total_queries = 0

    async with HaikuRAG(Path(__file__).parent / "benchmark.sqlite") as rag:
        for i, doc in enumerate(tqdm(corpus)):
            doc_id = doc["document_id"]  # type: ignore
            matches = await rag.search(
                query=doc["question"],  # type: ignore
                limit=3,
            )

            total_queries += 1

            # Check position of correct document in results
            for position, (chunk, _) in enumerate(matches):
                retrieved = await rag.get_document_by_id(chunk.document_id)
                if retrieved and retrieved.uri == doc_id:
                    if position == 0:  # First position
                        correct_at_1 += 1
                        correct_at_2 += 1
                        correct_at_3 += 1
                    elif position == 1:  # Second position
                        correct_at_2 += 1
                        correct_at_3 += 1
                    elif position == 2:  # Third position
                        correct_at_3 += 1
                    break

    # Calculate recall metrics
    recall_at_1 = correct_at_1 / total_queries
    recall_at_2 = correct_at_2 / total_queries
    recall_at_3 = correct_at_3 / total_queries

    print(f"Total queries: {total_queries}")
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@2: {recall_at_2:.4f}")
    print(f"Recall@3: {recall_at_3:.4f}")

    return {"recall@1": recall_at_1, "recall@2": recall_at_2, "recall@3": recall_at_3}


async def main():
    await populate_db()
    await run_match_benchmark()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

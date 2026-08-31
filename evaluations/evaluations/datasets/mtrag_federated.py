import argparse
import asyncio
import hashlib
import json
import random
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

from evaluations.config import DatasetSpec
from evaluations.datasets.mtrag import (
    _load_qrels,
    build_mtrag_case,
    load_clapnq_corpus,
    load_clapnq_retrieval,
    map_mtrag_document,
    map_mtrag_retrieval,
)
from evaluations.evaluators import (
    CitationMAPEvaluator,
    MAPEvaluator,
    NDCGEvaluator,
    RecallEvaluator,
)
from haiku.rag.config import load_yaml_config
from haiku.rag.config.models import AppConfig
from haiku.rag.utils import get_default_data_dir

COLLECTION_PREFIX = "clapnq"
# The four corpora MTRAG ships, in upstream order.
DOMAINS = ("clapnq", "cloud", "fiqa", "govt")
DEFAULT_SEED = 20260831
# Whole titles are kept, and the 148 titles holding a gold passage carry 10,723
# passages between them, so that is the floor. A budget near it leaves no
# cross-topic distractors at all and inflates recall; this default leaves about
# 29,000, a gold-title share near a quarter.
GOLD_TITLE_FLOOR = 10_723
DEFAULT_BUDGET = 40_000
INGEST_BATCH_SIZE = 512
FTS_INDEX_NAME = "content_fts_idx"


def _hash_int(payload: str) -> int:
    """sha256 rather than `hash()`, which is salted per process: the partition is
    never stored, and scoring recomputes it in a different process than the one
    that ingested."""
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _unit(payload: str) -> float:
    """A stable value in [0, 1) for probabilistic assignment."""
    return (_hash_int(payload) % 10**9) / 10**9


def domain_files(domain: str, variant: str = "lastturn") -> tuple[str, str, str]:
    """Corpus, qrels and query paths for one MTRAG domain."""
    return (
        f"corpora/passage_level/{domain}.jsonl.zip",
        f"mtrag-human/retrieval_tasks/{domain}/qrels/dev.tsv",
        f"mtrag-human/retrieval_tasks/{domain}/{domain}_{variant}.jsonl",
    )


def _domain_collection(title: str, domain: str, n: int, seed: int) -> int:
    """The collection a title takes from its domain.

    Domains map onto collections proportionally: with more collections than
    domains each domain is subdivided by title, with fewer, domains are grouped.
    """
    index = DOMAINS.index(domain)
    count = len(DOMAINS)
    if n >= count:
        per = n // count
        return index * per + (_hash_int(f"{seed}/sub/{title}") % per)
    return index * n // count


def collection_of(
    title: str,
    n: int,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.0,
    domain: str | None = None,
) -> int:
    """Which of `n` collections holds a title's passages.

    Keyed on the title, so an article's passages never split.

    Without a domain the assignment is a pure hash — a topically arbitrary
    grouping of whole articles, which is all the quota and order-bias arms need.
    With one, `alpha` interpolates between the domain partition (0, each
    collection one topic) and a uniform shard (1, domain ignored). Sharding is
    the endpoint of the knob rather than a rival design.
    """
    if n < 1:
        raise ValueError("a partition needs at least one collection")
    shard = _hash_int(f"{seed}/{title}") % n
    if domain is None or alpha >= 1.0:
        return shard
    if alpha > 0.0 and _unit(f"{seed}/alpha/{title}") < alpha:
        return shard
    return _domain_collection(title, domain, n, seed)


def collection_names(n: int) -> tuple[str, ...]:
    """The collection names in declaration order.

    Order is load-bearing: fusion resolves equal ranks to configured order, so
    permuting these names is an arm rather than a cosmetic change.
    """
    return tuple(f"{COLLECTION_PREFIX}_{index}" for index in range(n))


def database_paths(n: int, seed: int = DEFAULT_SEED) -> dict[str, str]:
    """Where each collection's database lives.

    The partition is in the filename, so a build at one `(n, seed)` can never
    overwrite another's databases or be searched by the wrong config.
    """
    root = get_default_data_dir() / "evaluations" / "dbs"
    return {
        name: str(root / f"mtrag_federated_s{seed}_n{n}_{index}.lancedb")
        for index, name in enumerate(collection_names(n))
    }


def sample_records(
    records: Sequence[Mapping[str, Any]],
    gold_ids: Iterable[str],
    budget: int = DEFAULT_BUDGET,
    seed: int = DEFAULT_SEED,
) -> list[Mapping[str, Any]]:
    """A fixed sub-corpus: every gold passage, plus seeded distractor titles.

    Whole titles are kept or dropped together. Gold is mandatory, so a budget
    below the gold floor yields the gold titles alone rather than an incomplete
    corpus that would score as missing retrievals.
    """
    by_title: dict[str, list[Mapping[str, Any]]] = {}
    title_of: dict[str, str] = {}
    for row in records:
        by_title.setdefault(row["title"], []).append(row)
        title_of[row["_id"]] = row["title"]

    wanted = set(gold_ids)
    missing = sorted(wanted - set(title_of))
    if missing:
        raise ValueError(
            f"{len(missing)} gold passages do not resolve to the corpus, "
            f"first few: {missing[:3]}"
        )

    gold_titles = {title_of[passage_id] for passage_id in wanted}
    kept = {title for title in by_title if title in gold_titles}
    total = sum(len(by_title[title]) for title in kept)

    distractors = [title for title in by_title if title not in gold_titles]
    random.Random(seed).shuffle(distractors)
    for title in distractors:
        size = len(by_title[title])
        if total + size > budget:
            continue
        kept.add(title)
        total += size

    return [row for row in records if row["title"] in kept]


def partition_records(
    records: Sequence[Mapping[str, Any]],
    n: int,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[Mapping[str, Any]]]:
    """Route every record to its collection, naming all `n` even when empty."""
    names = collection_names(n)
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in names}
    for row in records:
        grouped[names[collection_of(row["title"], n, seed)]].append(row)
    return grouped


def pool_composition(
    records: Sequence[Mapping[str, Any]], gold_ids: Iterable[str]
) -> tuple[int, int]:
    """Passages in gold-bearing titles, and passages in distractor titles.

    A pool with no distractors scores as an easy retrieval task and says
    nothing, so the build reports this rather than leaving it to be inferred
    from the budget.
    """
    wanted = set(gold_ids)
    gold_titles = {row["title"] for row in records if row["_id"] in wanted}
    gold_side = sum(1 for row in records if row["title"] in gold_titles)
    return gold_side, len(records) - gold_side


def gold_passage_ids() -> set[str]:
    """Every corpus id the ClapNQ qrels reference."""
    return {passage_id for ids in _load_qrels().values() for passage_id in ids}


def load_pool(
    budget: int = DEFAULT_BUDGET, seed: int = DEFAULT_SEED
) -> list[Mapping[str, Any]]:
    records = [dict(row) for row in load_clapnq_corpus()]
    return sample_records(records, gold_passage_ids(), budget, seed)


POOLED_PREFIX = "dom"


class PassageIdCollision(AssertionError):
    """Two domains claim the same passage id, so uri-keyed gold is ambiguous."""


def pooled_collection_names(n: int) -> tuple[str, ...]:
    """Names for the pooled partition, positional and distinct from the
    single-domain set so the two never share database paths."""
    return tuple(f"{POOLED_PREFIX}_{index}" for index in range(n))


def pooled_database_paths(
    n: int, alpha: float, seed: int = DEFAULT_SEED
) -> dict[str, str]:
    root = get_default_data_dir() / "evaluations" / "dbs"
    tag = f"s{seed}_a{alpha:g}_n{n}"
    return {
        name: str(root / f"mtrag_pooled_{tag}_{index}.lancedb")
        for index, name in enumerate(pooled_collection_names(n))
    }


def load_pooled_records() -> list[Mapping[str, Any]]:
    """Every passage of all four domains, each tagged with the domain it came
    from. Raises when two domains claim one passage id, since gold is uri-keyed.
    """
    from evaluations.datasets.mtrag import _download

    records: list[Mapping[str, Any]] = []
    seen: dict[str, str] = {}
    for domain in DOMAINS:
        corpus_file, _, _ = domain_files(domain)
        path = _download(corpus_file)
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as handle:
                for line in handle:
                    row = json.loads(line)
                    passage_id = row["_id"]
                    if passage_id in seen and seen[passage_id] != domain:
                        raise PassageIdCollision(
                            f"{passage_id} claimed by {seen[passage_id]} and {domain}"
                        )
                    seen[passage_id] = domain
                    records.append(
                        {
                            "_id": passage_id,
                            "title": row["title"],
                            "text": row["text"],
                            "domain": domain,
                        }
                    )
    return records


def load_pooled_queries(variant: str = "lastturn") -> list[dict[str, Any]]:
    """Retrieval queries from every domain, each with its gold passage uris."""
    from evaluations.datasets.mtrag import _download, _parse_qrels

    out: list[dict[str, Any]] = []
    for domain in DOMAINS:
        _, qrels_file, query_file = domain_files(domain, variant)
        qrels = _parse_qrels(_download(qrels_file).read_text().splitlines())
        for line in _download(query_file).read_text().splitlines():
            if not line.strip():
                continue
            query = json.loads(line)
            expected = qrels.get(query["_id"])
            if not expected:
                continue
            out.append(
                {
                    "query_id": f"{domain}/{query['_id']}",
                    "question": query["text"],
                    "expected_uris": expected,
                    "domain": domain,
                }
            )
    return out


def pooled_gold_ids(variant: str = "lastturn") -> set[str]:
    return {
        uri for query in load_pooled_queries(variant) for uri in query["expected_uris"]
    }


def load_pooled(
    budget: int = DEFAULT_BUDGET, seed: int = DEFAULT_SEED
) -> list[Mapping[str, Any]]:
    return sample_pooled_records(load_pooled_records(), pooled_gold_ids(), budget, seed)


def sample_pooled_records(
    records: Sequence[Mapping[str, Any]],
    gold_ids: Iterable[str],
    budget: int = DEFAULT_BUDGET,
    seed: int = DEFAULT_SEED,
) -> list[Mapping[str, Any]]:
    """A fixed sub-corpus at passage level, keeping every gold passage.

    The single-domain dataset keeps whole titles, which cannot work here: `title`
    is the empty string for every cloud and fiqa passage, so two of the four
    domains have exactly one title covering 72,442 and 61,022 passages. Whole
    titles put the gold floor at 146,543 passages, leaving no distractors at any
    budget below the entire corpus.

    Passage level costs nothing this comparison needs: at alpha=0 the domain
    places a collection, so a query's gold is concentrated by construction rather
    than by the atom.
    """
    wanted = set(gold_ids)
    by_id = {row["_id"]: row for row in records}
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError(
            f"{len(missing)} gold passages do not resolve to the pooled corpus, "
            f"first few: {missing[:3]}"
        )
    kept = set(wanted)
    others = [row["_id"] for row in records if row["_id"] not in wanted]
    random.Random(seed).shuffle(others)
    for passage_id in others:
        if len(kept) >= budget:
            break
        kept.add(passage_id)
    return [row for row in records if row["_id"] in kept]


def partition_pooled(
    records: Sequence[Mapping[str, Any]],
    n: int,
    alpha: float,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[Mapping[str, Any]]]:
    """Route pooled records to collections, honouring each record's domain.

    Keyed on the passage id rather than the title, because two of the four
    domains have no titles. See `sample_pooled_records`.
    """
    names = pooled_collection_names(n)
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in names}
    for row in records:
        index = collection_of(row["_id"], n, seed, alpha=alpha, domain=row["domain"])
        grouped[names[index]].append(row)
    return grouped


def _unused_document_loader() -> Dataset:
    raise RuntimeError(
        "the federated corpus is built by build_databases(); run with --skip-db"
    )


def emitted_config(reference: Path, n: int, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """The reference config with this partition's databases placed in it."""
    settings = load_yaml_config(reference)
    lancedb = dict(settings.get("lancedb") or {})
    lancedb.pop("uri", None)
    lancedb["databases"] = database_paths(n, seed)
    settings["lancedb"] = lancedb
    return settings


class FTSIndexNotCoveringRows(AssertionError):
    """The chunks FTS index does not cover every row, so full-text search is
    dead while still returning results."""


async def assert_fts_covers_rows(table: Any, name: str) -> None:
    rows = await table.count_rows()
    indices = {index.name for index in await table.list_indices()}
    if FTS_INDEX_NAME not in indices:
        raise FTSIndexNotCoveringRows(f"{name}: no {FTS_INDEX_NAME} on {rows} rows")
    stats = await table.index_stats(FTS_INDEX_NAME)
    indexed = getattr(stats, "num_indexed_rows", 0) or 0
    if indexed < rows:
        raise FTSIndexNotCoveringRows(
            f"{name}: {FTS_INDEX_NAME} covers {indexed} of {rows} rows; "
            "full-text search would return near-arbitrary rows"
        )


async def build_databases(
    config: AppConfig,
    n: int,
    seed: int = DEFAULT_SEED,
    budget: int = DEFAULT_BUDGET,
) -> dict[str, int]:
    """Ingest the partition into one database per collection.

    Each member is opened by configured name with a scope of one, which is what
    makes `create=True` legal on a client whose config places several.
    """
    from haiku.rag.client import HaikuRAG

    from evaluations.population import _ingest_batched

    names = collection_names(n)
    configured = set(config.lancedb.databases or {})
    missing = sorted(set(names) - configured)
    if missing:
        raise ValueError(
            f"lancedb.databases must place every collection; missing {missing}"
        )

    pool = load_pool(budget, seed)
    gold_side, distractors = pool_composition(pool, gold_passage_ids())
    print(
        f"pool: {len(pool)} passages, {gold_side} in gold-bearing titles, "
        f"{distractors} distractors"
    )
    if not distractors:
        print(
            "  WARNING: no distractor titles, so every passage belongs to an "
            f"answer-bearing article; raise --budget above {GOLD_TITLE_FLOOR}"
        )
    grouped = partition_records(pool, n, seed)
    written: dict[str, int] = {}
    for name in names:
        async with HaikuRAG(config=config, sources=[name], create=True) as client:
            await _ingest_batched(
                client, MTRAG_FEDERATED_SPEC, grouped[name], INGEST_BATCH_SIZE
            )
            # The chunks FTS index is built once when the table is created, over
            # zero rows, and nothing folds later rows into it but an optimize.
            # `auto_vacuum` is false here, as in every reference config, so
            # without this the index covers nothing and full-text search returns
            # near-arbitrary rows while still looking like it works.
            await client.store.vacuum(retention_seconds=0)
            await assert_fts_covers_rows(client.store.chunks_table, name)
        written[name] = len(grouped[name])
    return written


async def build_pooled_databases(
    config: AppConfig,
    n: int,
    alpha: float,
    seed: int = DEFAULT_SEED,
    budget: int = DEFAULT_BUDGET,
) -> dict[str, int]:
    """Ingest the four-domain pooled partition, one database per collection."""
    from haiku.rag.client import HaikuRAG

    from evaluations.population import _ingest_batched

    names = pooled_collection_names(n)
    configured = set(config.lancedb.databases or {})
    missing = sorted(set(names) - configured)
    if missing:
        raise ValueError(
            f"lancedb.databases must place every collection; missing {missing}"
        )

    pool = load_pooled(budget, seed)
    by_domain: dict[str, int] = {}
    for row in pool:
        by_domain[row["domain"]] = by_domain.get(row["domain"], 0) + 1
    # Passage level, not `pool_composition`: that counts gold-bearing titles,
    # which is meaningless here since cloud and fiqa have one empty title each.
    gold = pooled_gold_ids()
    gold_kept = sum(1 for row in pool if row["_id"] in gold)
    print(
        f"pool: {len(pool)} passages, {gold_kept} gold, "
        f"{len(pool) - gold_kept} distractors, by domain {by_domain}"
    )
    grouped = partition_pooled(pool, n, alpha, seed)
    written: dict[str, int] = {}
    for name in names:
        async with HaikuRAG(config=config, sources=[name], create=True) as client:
            await _ingest_batched(
                client, MTRAG_POOLED_SPEC, grouped[name], INGEST_BATCH_SIZE
            )
            await client.store.vacuum(retention_seconds=0)
            await assert_fts_covers_rows(client.store.chunks_table, name)
        written[name] = len(grouped[name])
    return written


MTRAG_FEDERATED_SPEC = DatasetSpec(
    key="mtrag_federated",
    # Never read: the run searches the configured set. Present because the spec
    # requires one, and pointed at the first collection so a stray --db is
    # obviously wrong rather than silently plausible.
    db_filename="mtrag_federated_unused.lancedb",
    document_loader=_unused_document_loader,
    document_mapper=map_mtrag_document,
    # The QA phase is not wired yet, and the loader is what run_qa_benchmark
    # reaches first, so a forgotten --skip-qa fails loudly there. The builder is
    # the one the generation tasks will need when QA arrives.
    qa_loader=_unused_document_loader,
    qa_case_builder=build_mtrag_case,
    retrieval_loader=lambda: load_clapnq_retrieval("lastturn"),
    retrieval_mapper=map_mtrag_retrieval,
    retrieval_evaluators=[
        RecallEvaluator(5),
        RecallEvaluator(10),
        NDCGEvaluator(5),
        MAPEvaluator(),
    ],
    citation_evaluator=CitationMAPEvaluator(),
    # The product default, which is where the fusion depth quota bites.
    retrieval_limit=5,
    ingest_batch_size=INGEST_BATCH_SIZE,
)


MTRAG_POOLED_SPEC = DatasetSpec(
    key="mtrag_pooled",
    db_filename="mtrag_pooled_unused.lancedb",
    document_loader=_unused_document_loader,
    document_mapper=map_mtrag_document,
    qa_loader=_unused_document_loader,
    qa_case_builder=build_mtrag_case,
    retrieval_loader=lambda: Dataset.from_list(load_pooled_queries("lastturn")),
    retrieval_mapper=map_mtrag_retrieval,
    retrieval_evaluators=[
        RecallEvaluator(5),
        RecallEvaluator(10),
        NDCGEvaluator(5),
        MAPEvaluator(),
    ],
    citation_evaluator=CitationMAPEvaluator(),
    retrieval_limit=5,
    ingest_batch_size=INGEST_BATCH_SIZE,
)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the federated ClapNQ partition and emit the config that "
            "searches exactly it."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="reference config")
    parser.add_argument("--n", type=int, required=True, help="collection count")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="build the four-domain pooled corpus instead of clapnq alone",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="pooled only: 0 keeps a collection to one domain, 1 shards across all",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="where to write the emitted config for this partition",
    )
    args = parser.parse_args()

    if args.pooled:
        settings = load_yaml_config(args.config)
        lancedb = dict(settings.get("lancedb") or {})
        lancedb.pop("uri", None)
        lancedb["databases"] = pooled_database_paths(args.n, args.alpha, args.seed)
        settings["lancedb"] = lancedb
    else:
        settings = emitted_config(args.config, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(settings, sort_keys=False))
    print(f"wrote {args.out}")

    # Reload from disk, so the config that builds is the file that will search.
    config = AppConfig.model_validate(load_yaml_config(args.out))
    if args.pooled:
        written = await build_pooled_databases(
            config, args.n, args.alpha, args.seed, args.budget
        )
    else:
        written = await build_databases(config, args.n, args.seed, args.budget)
    for name, count in written.items():
        print(f"{name}: {count} passages")


if __name__ == "__main__":
    asyncio.run(main())

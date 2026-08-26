# The multi-database acceptance dataset

Acceptance gate for `lancedb.databases` (searching, asking and analyzing across
several databases at once), and a permanent regression dataset afterwards. The unit
tests pin mechanism; this dataset exists for three failure modes that are silent end
to end.

## The three failure modes

1. **Wrong attribution** — the right answer, cited to the wrong database. `cited_map`
   scores URIs and would pass it.
2. **Lost evidence under fusion** — a fact in the second database never reaches the
   model because the first filled the limit. Looks like a knowledge gap.
3. **Scope leakage** — a question scoped with `sources` draws on a database it
   excluded.

`cited_sources` is already recorded per case, and `AnalysisState.executions` already
stores the model's own Python, so both attribution and sandbox-surface coverage are
measurable without production changes. What this dataset adds is a corpus where the
expected database is known per question.

## Corpus

Three databases: `northern` and `southern` hold station reports on the same schema,
`equipment` holds instrument spec sheets that share the vocabulary (anemometer,
calibration, elevation) and answer none of the questions. Counts differ — 9 / 8 / 6 —
so a count answer cannot be right by luck while attribution is wrong.

**Every name is invented.** A real station lets the model answer from priors, and the
eval would measure memorisation instead of retrieval. This is load-bearing: do not
replace these with real stations.

Each report has four sections (Overview / Instruments / Measurements / Maintenance)
plus a twelve-row monthly table under `Measurements`. Elevation and commissioning year
appear in **prose under Overview, never in the table**, so B1 and B3 stay chunk-level
retrieval questions instead of becoming table-reading questions. Elevations and years
are unique across the corpus, so a number identifies exactly one station.

Two kinds of deliberate collision:

- **Three near-name pairs** — Kestrel/Kestrel Ridge, Petrel/Petrel Point, Skua/Skua
  Bay. Members of a pair are identical apart from the name and the numbers, sharing
  instrument and technician, because a difference in either would hand the model a free
  discriminator. This is what makes B3 *harder* with a reranker (see Arms).
- **One shared entity** — `Station Auk` in both databases, commissioned 1987 and 2011.
  Unscoped it should surface both; scoped it must yield exactly one.

The corpus is generated from `STATIONS` and `INSTRUMENTS` in `multidb.py`, and the
expected answers are derived from the same definitions, so questions and gold answers
cannot drift apart. Monthly readings are derived with SHA-256 over
`{database}/{name}` — `hash()` is salted per process and would not be stable across
rebuilds, and keying on the name alone would give the two Station Auk reports identical
tables.

## Building

`evaluations run` populates one database and refuses a configured set, so this dataset
builds its own:

```
uv run python -m evaluations.datasets.multidb --config evaluations/configs/multidb.yaml
evaluations run multidb --config evaluations/configs/multidb.yaml --skip-db --skip-retrieval
```

The builder asserts, per station, that **no single chunk holds all twelve monthly
readings**, and fails the build if one does. Without that guarantee S3 collapses into a
search question, and the guarantee has to survive a chunker change — so it is asserted,
not assumed.

It earned its keep on the first real build: at `chunk_size` 256 the chunker keeps the
entire table in one chunk, and so does 128. Only 64 splits it, which is what the config
pins. The cost is a corpus chunked more finely than a real one; the alternative is a
longer table at a realistic chunk size, which would change what S3 asks. Worth
revisiting if the dataset grows.

## Question families

B-family runs against both capabilities, S-family against analysis only.

| | family | proves |
|---|---|---|
| B1 | single-source fact | `cited_sources == ["northern"]` |
| B2 | cross-database join | fusion keeps the second fact; both cited |
| B3 | near-name distractor | precision — the twin must not be cited |
| B4 | B1 scoped with `sources` | scope honoured, zero leakage |
| B5 | shared entity, unscoped | surfaces and attributes both |
| B6 | absent station | grounded refusal |
| B7 | `sources=[]` | empty scope refuses rather than confabulating |
| S1 | documents per database | `list_documents()` |
| S2 | stations per programme mentioning a term | in-code `search()` |
| S3 | total of one station's monthly readings | whole-document surface |
| S4 | item and table counts | `items.jsonl` |
| S5 | section headings in order | `toc.json` |
| S6 | uri and database of a titled document | `metadata.json` |

**B3 and B4 carry 8-10 instances each**, because their gates are pass/fail and "zero
leakage" over three cases is not evidence of absence. **Half the B4 instances put the
better answer in the excluded database**, so that honouring scope costs the model the
better-matching chunk — otherwise the scoped-in database always holds the best content
and no-leakage is satisfied trivially.

**B7 is API-only.** The CLI has no `--sources` (it has `--database` for exactly one),
so an empty scope is unreachable from the command line.

### Why S3 is not a `content.txt` question

`extract_item_text` serialises a table item **to markdown**, and `items.jsonl` emits
`item.text` per item, so a table item's text is the whole twelve-row table.
`items.jsonl` is therefore an information superset of `content.txt`: same text,
itemised, plus structure and chunk ids. There is no natural question answerable only
from `content.txt`, and anything contrived enough to force it would measure the
plumbing rather than the capability.

So S3 asserts that the model reached **a whole-document surface — either `content.txt`
or `items.jsonl`** — and which one it chose is reported as a measurement. If
`content.txt` is never chosen, that is a finding about whether the file earns its place
in the VFS, the same class of question as `HAIKU_RAG_DISABLE_TOC`.

Surface coverage generally is **measured, not forced**: the model picks its route and
the run reports what it touched.

## Arms

- **RAG x {reranker, none}**, and the two arms cover *different* families rather than
  the same family at two strengths. Over-fetch is reranker-gated
  (`client/search.py`, `limit * _RERANK_OVERFETCH if client.reranker else limit`), so:
  - **Fusion loss (B2) is a no-reranker property.** At `limit` 5 across three
    databases, RRF fuses 15 candidates to 5 and the truncation bites.
  - **Attribution under confusion (B3) is harder *with* the reranker**, because a
    reranker scoring the union has no notion of source and is doing semantic matching
    on exactly the near-identical pair built to confuse it. Without one, RRF keeps the
    pair apart because each database contributes its own rank-1.
- **Analysis x reranker** for the S-family.
- Optionally analysis with `HAIKU_RAG_DISABLE_TOC=1`, which answers what that toggle
  was added for.

### RRF ties resolve to configured order

In the RRF branch of `_fuse`, the sort key is the score alone. Python's sort is stable
and `reverse=True` preserves the relative order of equal elements, so equal-scoring
candidates resolve to insertion order, which is `source_names`, which is **the order
the databases appear in the configuration**. At `limit` 5 across three databases the
slot allocation is therefore decided by config order, not relevance.

Consequence for this dataset: **B2 instances randomise database order**, or the family
measures ordering instead of fusion. Consequence for operators: the first database
listed wins ties, and nothing else documents that.

## Gates and rates

**Two hard gates.** Any failure is a bug, not a rate:

- **Scope leakage = 0** — no B4 or B7 case cites a database outside its scope.
- **Attribution errors = 0** on B1 and B3, where exactly one database is correct.

**Rates**: judge pass split behaviour/structural, `cited_map`, and a surface-coverage
table built from the recorded code.

Scoring is **deterministic** everywhere except B6. The answers are known numbers, names
and counts, so exact and numeric matching score them without inheriting judge failure
modes — a red gate has to mean the code is broken, not that the judge spiralled. B1 and
B3 assert **gold present AND distractor absent**, because a hedge ("either 1240 m or
2310 m") passes a presence check while being exactly the failure B3 exists to catch.
Numerics are extracted and compared, never string-matched, since "1,240 metres",
"1240 m" and a full sentence are all legitimate.

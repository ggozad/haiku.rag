import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from pydantic_evals import Case

from evaluations.config import DatasetSpec, ScopedQuestion
from evaluations.datasets.multidb_report import render as render_gate_report
from evaluations.evaluators.multidb import MultiDBScores
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig

NORTHERN = "northern"
SOUTHERN = "southern"
EQUIPMENT = "equipment"

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@dataclass(frozen=True)
class Station:
    """One station report. Names are invented: a real station lets the model
    answer from priors, which would measure memorisation instead of retrieval."""

    name: str
    database: str
    elevation_m: int
    commissioned: int
    programme: str
    instrument: str
    technician: str

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-")

    @property
    def uri(self) -> str:
        return f"station://{self.database}/{self.slug}"

    @property
    def title(self) -> str:
        return f"Station {self.name} Report"

    @property
    def readings(self) -> tuple[int, ...]:
        """Twelve monthly wind-speed readings, derived from database and name so
        they are stable across rebuilds and interpreters, and so the two Station
        Auk reports differ. `hash()` is salted per process, so it cannot be used."""
        digest = hashlib.sha256(f"{self.database}/{self.name}".encode()).digest()
        return tuple(37 + digest[i] % 53 for i in range(12))

    @property
    def readings_total(self) -> int:
        return sum(self.readings)


@dataclass(frozen=True)
class Instrument:
    """An equipment spec sheet: shares the stations' vocabulary (anemometer,
    calibration, elevation) and answers none of the questions."""

    model: str
    kind: str
    calibration_days: int
    operating_ceiling_m: int

    @property
    def slug(self) -> str:
        return self.model.lower().replace(" ", "-").replace("/", "-")

    @property
    def uri(self) -> str:
        return f"equipment://{EQUIPMENT}/{self.slug}"

    @property
    def title(self) -> str:
        return f"{self.model} Specification"


NORTHERN_PROGRAMME = "Northern Uplands Programme"
SOUTHERN_PROGRAMME = "Southern Ranges Programme"

# Three near-name pairs, each identical apart from the name and the numbers: the
# wrong database yields a wrong elevation, so B3 fails visibly and cited_sources
# names the culprit. Pair members share instrument and technician so only the name
# and the numbers distinguish them, which is what makes the reranker arm the harder
# one for that family.
#
# Station Auk exists in both databases with commissioning years 24 years apart:
# unscoped it should surface both, scoped it must yield exactly one.
#
# Elevations and years are unique across the corpus, so a number identifies one
# station. Document counts differ per database (9 / 8 / 6) so a count answer cannot
# be right by luck while attribution is wrong.
STATIONS: tuple[Station, ...] = (
    Station(
        "Kestrel",
        NORTHERN,
        1240,
        1998,
        NORTHERN_PROGRAMME,
        "Vaisala WXT536",
        "R. Aldiss",
    ),
    Station(
        "Petrel",
        NORTHERN,
        860,
        1991,
        NORTHERN_PROGRAMME,
        "Gill WindSonic 4",
        "M. Torrance",
    ),
    Station(
        "Skua", NORTHERN, 2145, 1989, NORTHERN_PROGRAMME, "Young 81000V", "S. Okonkwo"
    ),
    Station(
        "Auk", NORTHERN, 1105, 1987, NORTHERN_PROGRAMME, "Thies 4.3350", "R. Aldiss"
    ),
    Station(
        "Gannet",
        NORTHERN,
        640,
        2006,
        NORTHERN_PROGRAMME,
        "Vaisala WXT536",
        "M. Torrance",
    ),
    Station(
        "Tern",
        NORTHERN,
        1420,
        2001,
        NORTHERN_PROGRAMME,
        "Gill WindSonic 4",
        "S. Okonkwo",
    ),
    Station(
        "Guillemot",
        NORTHERN,
        1780,
        1996,
        NORTHERN_PROGRAMME,
        "Young 81000V",
        "R. Aldiss",
    ),
    Station(
        "Shearwater",
        NORTHERN,
        505,
        2013,
        NORTHERN_PROGRAMME,
        "Thies 4.3350",
        "M. Torrance",
    ),
    Station(
        "Fulmar",
        NORTHERN,
        1550,
        1984,
        NORTHERN_PROGRAMME,
        "Campbell CSAT3B",
        "S. Okonkwo",
    ),
    Station(
        "Kestrel Ridge",
        SOUTHERN,
        2310,
        2004,
        SOUTHERN_PROGRAMME,
        "Vaisala WXT536",
        "R. Aldiss",
    ),
    Station(
        "Petrel Point",
        SOUTHERN,
        1975,
        1993,
        SOUTHERN_PROGRAMME,
        "Gill WindSonic 4",
        "M. Torrance",
    ),
    Station(
        "Skua Bay",
        SOUTHERN,
        415,
        1979,
        SOUTHERN_PROGRAMME,
        "Young 81000V",
        "S. Okonkwo",
    ),
    Station(
        "Auk", SOUTHERN, 1770, 2011, SOUTHERN_PROGRAMME, "Thies 4.3350", "L. Feodorov"
    ),
    Station(
        "Albatross",
        SOUTHERN,
        2540,
        1999,
        SOUTHERN_PROGRAMME,
        "Vaisala WXT536",
        "P. Nakamura",
    ),
    Station(
        "Prion",
        SOUTHERN,
        2185,
        1995,
        SOUTHERN_PROGRAMME,
        "Gill WindSonic 4",
        "L. Feodorov",
    ),
    Station(
        "Sheathbill",
        SOUTHERN,
        930,
        2016,
        SOUTHERN_PROGRAMME,
        "Young 81000V",
        "P. Nakamura",
    ),
    Station(
        "Snowcap",
        SOUTHERN,
        2760,
        2009,
        SOUTHERN_PROGRAMME,
        "Metek uSonic-3",
        "L. Feodorov",
    ),
)

# The near-name pairs, northern member first. B3 draws its instances from these.
NEAR_NAME_PAIRS: tuple[tuple[str, str], ...] = (
    ("Kestrel", "Kestrel Ridge"),
    ("Petrel", "Petrel Point"),
    ("Skua", "Skua Bay"),
)

SHARED_STATION = "Auk"

INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument("Vaisala WXT536", "ultrasonic anemometer", 365, 3000),
    Instrument("Gill WindSonic 4", "ultrasonic anemometer", 730, 2800),
    Instrument("Young 81000V", "ultrasonic anemometer", 400, 3200),
    Instrument("Thies 4.3350", "cup anemometer", 545, 2400),
    Instrument("Campbell CSAT3B", "sonic anemometer", 300, 3500),
    Instrument("Metek uSonic-3", "sonic anemometer", 450, 3100),
)

DATABASE_NAMES = (NORTHERN, SOUTHERN, EQUIPMENT)


def stations_in(database: str) -> tuple[Station, ...]:
    return tuple(s for s in STATIONS if s.database == database)


def station(name: str, database: str) -> Station:
    for candidate in STATIONS:
        if candidate.name == name and candidate.database == database:
            return candidate
    raise KeyError(f"no station {name!r} in {database!r}")


def render_station_report(s: Station) -> str:
    """Four sections plus a twelve-row table, so `toc.json` has real structure,
    `items.jsonl` reports a table, and the readings outlast one chunk."""
    rows = "\n".join(
        f"| {month} | {value} |"
        for month, value in zip(MONTHS, s.readings, strict=True)
    )
    return f"""# {s.title}

## Overview

Station {s.name} sits at {s.elevation_m} metres and was commissioned in
{s.commissioned}. It belongs to the {s.programme} and reports hourly.

## Instruments

The primary sensor is a {s.instrument}. Calibration is verified against the
programme reference before each seasonal changeover.

## Measurements

Mean monthly wind speed, in tenths of a metre per second, for the reporting year:

| Month | Mean wind speed |
| --- | --- |
{rows}

## Maintenance

Maintenance is carried out by {s.technician}. The mast was last inspected during
the autumn visit; no corrosion was recorded and the guy tensions were within
tolerance.
"""


def render_spec_sheet(i: Instrument) -> str:
    return f"""# {i.title}

## Overview

The {i.model} is a {i.kind} used across the station network. It is rated for
installation up to an elevation of {i.operating_ceiling_m} metres.

## Calibration

The recommended calibration interval is {i.calibration_days} days. Calibration
is performed against a reference anemometer under laboratory conditions.

## Notes

This sheet describes the instrument only. It records no station, no programme
and no measurement history.
"""


@dataclass(frozen=True)
class CorpusDocument:
    database: str
    uri: str
    title: str
    content: str


def corpus() -> tuple[CorpusDocument, ...]:
    docs = [
        CorpusDocument(s.database, s.uri, s.title, render_station_report(s))
        for s in STATIONS
    ]
    docs += [
        CorpusDocument(EQUIPMENT, i.uri, i.title, render_spec_sheet(i))
        for i in INSTRUMENTS
    ]
    return tuple(docs)


def documents_for(database: str) -> tuple[CorpusDocument, ...]:
    return tuple(d for d in corpus() if d.database == database)


class TableSplitError(AssertionError):
    """A single chunk holds every monthly reading, so `content.txt` is no longer
    the only route to their total and the S3 case would test search instead."""


async def _assert_readings_outlast_one_chunk(client: HaikuRAG, s: Station) -> None:
    doc = await client.get_document_by_uri(s.uri)
    if doc is None or doc.id is None:  # pragma: no cover - the builder just wrote it
        raise TableSplitError(f"{s.uri} missing after import")
    values = [str(v) for v in s.readings]
    for chunk in await client.chunk_repository.get_by_document_id(doc.id):
        if all(v in chunk.content for v in values):
            raise TableSplitError(
                f"one chunk of {s.uri} holds all {len(values)} monthly readings; "
                "lower processing.chunk_size so the table splits"
            )


async def build_databases(config: AppConfig) -> dict[str, int]:
    """Write each configured database from the generated corpus.

    `evaluations run` populates one database and refuses a configured set, so
    this dataset builds its own and the run is `--skip-db`.
    """
    configured = set(config.lancedb.databases or {})
    missing = set(DATABASE_NAMES) - configured
    if missing:
        raise ValueError(
            f"lancedb.databases must place {sorted(DATABASE_NAMES)}; missing {sorted(missing)}"
        )

    written: dict[str, int] = {}
    for name in DATABASE_NAMES:
        async with HaikuRAG(config=config, sources=[name], create=True) as client:
            for doc in documents_for(name):
                await client.create_document(doc.content, uri=doc.uri, title=doc.title)
            for s in stations_in(name):
                await _assert_readings_outlast_one_chunk(client, s)
            written[name] = len(documents_for(name))
    return written


def iter_expected_totals() -> Iterator[tuple[Station, int]]:
    for s in STATIONS:
        yield s, s.readings_total


async def main() -> None:  # pragma: no cover - operator entry point
    import argparse

    from haiku.rag.config import AppConfig, load_yaml_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = AppConfig.model_validate(load_yaml_config(args.config))
    written = await build_databases(config)
    for name, count in written.items():
        print(f"{name}: {count} documents")


if __name__ == "__main__":  # pragma: no cover - operator entry point
    import asyncio

    asyncio.run(main())


# --- question families -------------------------------------------------------
#
# B-family runs against both capabilities, S-family against analysis only, so
# they are registered as two dataset keys: a surface question cannot pass under
# the RAG target, and scoring it there would be noise rather than a finding.

UNIQUE_STATIONS = tuple(
    s
    for s in STATIONS
    if s.name != SHARED_STATION and not any(s.name in pair for pair in NEAR_NAME_PAIRS)
)


def _row(
    question_id: str,
    family: str,
    question: str,
    *,
    sources: list[str] | None = None,
    scope: list[str] | None = None,
    expected_value: float | None = None,
    expected_values: list[float] | None = None,
    distractor_value: float | None = None,
    expected_ordered: list[str] | None = None,
    forbidden_text: list[str] | None = None,
    expected_sources: list[str] | None = None,
    answerability: str = "ANSWERABLE",
) -> dict:
    return {
        "id": question_id,
        "family": family,
        "question": question,
        "sources": sources,
        "scope": scope,
        "expected_value": expected_value,
        "expected_values": expected_values,
        "distractor_value": distractor_value,
        "expected_ordered": expected_ordered,
        "forbidden_text": forbidden_text,
        "expected_sources": expected_sources,
        "answerability": answerability,
    }


def _twin_of(name: str) -> str:
    for north, south in NEAR_NAME_PAIRS:
        if name == north:
            return south
        if name == south:
            return north
    raise KeyError(name)


def behaviour_rows() -> list[dict]:
    rows: list[dict] = []

    # B1 — a fact held by exactly one database.
    for i, s in enumerate(UNIQUE_STATIONS[:3]):
        rows.append(
            _row(
                f"b1-{i}",
                "B1",
                f"At what elevation in metres does Station {s.name} sit?",
                expected_value=s.elevation_m,
                expected_sources=[s.database],
            )
        )

    # B2 — a fact from each database in one answer. RRF ties resolve to the
    # order the databases are listed, so the order is rotated across instances:
    # otherwise the family measures ordering rather than fusion.
    order = list(DATABASE_NAMES)
    for i, (north_name, south_name) in enumerate(NEAR_NAME_PAIRS):
        north, south = station(north_name, NORTHERN), station(south_name, SOUTHERN)
        higher = north if north.elevation_m > south.elevation_m else south
        rotated = order[i % len(order) :] + order[: i % len(order)]
        rows.append(
            _row(
                f"b2-{i}",
                "B2",
                f"Which sits higher, Station {north.name} or Station {south.name}, "
                "and at what elevation in metres?",
                sources=rotated,
                expected_value=higher.elevation_m,
                expected_sources=[NORTHERN, SOUTHERN],
            )
        )

    # B3 — the near-name distractor. Gold present and the twin's number absent,
    # because a hedge naming both is the failure this family exists to catch.
    for i, (north_name, south_name) in enumerate(NEAR_NAME_PAIRS):
        for name, db in ((north_name, NORTHERN), (south_name, SOUTHERN)):
            s = station(name, db)
            twin = station(_twin_of(name), SOUTHERN if db == NORTHERN else NORTHERN)
            rows.append(
                _row(
                    f"b3-elev-{i}-{db}",
                    "B3",
                    f"At what elevation in metres does Station {s.name} sit?",
                    expected_value=s.elevation_m,
                    distractor_value=twin.elevation_m,
                    expected_sources=[db],
                )
            )
        north = station(north_name, NORTHERN)
        twin = station(south_name, SOUTHERN)
        rows.append(
            _row(
                f"b3-year-{i}",
                "B3",
                f"In what year was Station {north.name} commissioned?",
                expected_value=north.commissioned,
                distractor_value=twin.commissioned,
                expected_sources=[NORTHERN],
            )
        )

    # B4 — scoped. Half the instances ask about a near-name pair member with the
    # twin excluded, so honouring scope costs the model the other strong match.
    for i, (north_name, south_name) in enumerate(NEAR_NAME_PAIRS):
        for name, db in ((north_name, NORTHERN), (south_name, SOUTHERN)):
            s = station(name, db)
            rows.append(
                _row(
                    f"b4-pair-{i}-{db}",
                    "B4",
                    f"At what elevation in metres does Station {s.name} sit?",
                    sources=[db],
                    scope=[db],
                    expected_value=s.elevation_m,
                    expected_sources=[db],
                )
            )
    for i, s in enumerate(UNIQUE_STATIONS[:4]):
        rows.append(
            _row(
                f"b4-unique-{i}",
                "B4",
                f"In what year was Station {s.name} commissioned?",
                sources=[s.database],
                scope=[s.database],
                expected_value=s.commissioned,
                expected_sources=[s.database],
            )
        )

    # B5 — one entity in both databases, unscoped: both must surface and both
    # must be attributed.
    north, south = station(SHARED_STATION, NORTHERN), station(SHARED_STATION, SOUTHERN)
    rows.append(
        _row(
            "b5-years",
            "B5",
            f"In what years was Station {SHARED_STATION} commissioned? There is a "
            "station of that name in more than one programme.",
            expected_values=[north.commissioned, south.commissioned],
            expected_sources=[NORTHERN, SOUTHERN],
        )
    )
    rows.append(
        _row(
            "b5-elevations",
            "B5",
            f"At what elevations do the stations named {SHARED_STATION} sit?",
            expected_values=[north.elevation_m, south.elevation_m],
            expected_sources=[NORTHERN, SOUTHERN],
        )
    )
    rows.append(
        _row(
            "b5-programmes",
            "B5",
            f"Which programmes operate a Station {SHARED_STATION}?",
            expected_ordered=[NORTHERN_PROGRAMME],
            expected_sources=[NORTHERN, SOUTHERN],
        )
    )

    # B6 — absent stations. Judge-scored, because a deterministic refusal matcher
    # keys on phrasing the model may never use.
    for i, absent in enumerate(("Cormorant", "Razorbill", "Kittiwake")):
        rows.append(
            _row(
                f"b6-{i}",
                "B6",
                f"At what elevation in metres does Station {absent} sit?",
                answerability="UNANSWERABLE",
            )
        )

    # B7 — empty scope covers no database, so there is no evidence to answer
    # from. API-only: the CLI has no --sources.
    for i, s in enumerate(UNIQUE_STATIONS[:3]):
        rows.append(
            _row(
                f"b7-{i}",
                "B7",
                f"At what elevation in metres does Station {s.name} sit?",
                sources=[],
                scope=[],
                answerability="UNANSWERABLE",
            )
        )
    return rows


def surface_rows() -> list[dict]:
    rows: list[dict] = []

    # S1 — inventory per database. Counts differ, so a count cannot be right by
    # luck while attribution is wrong.
    for name in DATABASE_NAMES:
        rows.append(
            _row(
                f"s1-{name}",
                "S1",
                f"How many documents does the {name} database hold?",
                expected_value=len(documents_for(name)),
            )
        )

    # S2 — a scoped search per programme, run in code.
    for programme, db in (
        (NORTHERN_PROGRAMME, NORTHERN),
        (SOUTHERN_PROGRAMME, SOUTHERN),
    ):
        for model in ("Vaisala WXT536", "Young 81000V"):
            count = sum(1 for s in stations_in(db) if s.instrument == model)
            rows.append(
                _row(
                    f"s2-{db}-{model.split()[0].lower()}",
                    "S2",
                    f"How many stations in the {programme} use a {model}?",
                    expected_value=count,
                )
            )

    # S3 — a whole-document surface. No single chunk holds all twelve readings
    # (asserted at build time), so the total cannot come from search alone.
    for name, db in (("Kestrel", NORTHERN), ("Auk", NORTHERN), ("Snowcap", SOUTHERN)):
        s = station(name, db)
        rows.append(
            _row(
                f"s3-{db}-{s.slug}",
                "S3",
                f"What is the total of the twelve monthly mean wind speeds in the "
                f"report for Station {s.name} in the {db} database?",
                expected_value=s.readings_total,
            )
        )

    # S4 — structure: the table's row count, which needs the itemised document.
    for name, db in (("Kestrel", NORTHERN), ("Albatross", SOUTHERN)):
        s = station(name, db)
        rows.append(
            _row(
                f"s4-{db}-{s.slug}",
                "S4",
                f"How many data rows does the monthly readings table in the report "
                f"for Station {s.name} have?",
                expected_value=len(MONTHS),
            )
        )

    # S5 — the outline, in document order.
    for name, db in (("Kestrel", NORTHERN), ("Snowcap", SOUTHERN)):
        s = station(name, db)
        rows.append(
            _row(
                f"s5-{db}-{s.slug}",
                "S5",
                f"List the section headings of the report for Station {s.name}, in "
                "the order they appear.",
                expected_ordered=[
                    "Overview",
                    "Instruments",
                    "Measurements",
                    "Maintenance",
                ],
            )
        )

    # S6 — per-document metadata, with the twin's uri forbidden.
    for north_name, south_name in NEAR_NAME_PAIRS[:2]:
        s = station(north_name, NORTHERN)
        twin = station(south_name, SOUTHERN)
        rows.append(
            _row(
                f"s6-{s.slug}",
                "S6",
                f"What is the uri of the document titled {s.title!r}, and which "
                "database holds it?",
                expected_ordered=[s.uri],
                forbidden_text=[twin.uri],
            )
        )
    return rows


def load_behaviour_questions() -> Dataset:
    return Dataset.from_list(behaviour_rows())


def load_surface_questions() -> Dataset:
    return Dataset.from_list(surface_rows())


def build_multidb_case(index: int, row: Mapping[str, Any]) -> Case[Any, Any, dict]:
    """One case. A scope travels in the inputs, since the task function receives
    a case's inputs and never its metadata."""
    question = str(row["question"])
    sources = row["sources"]
    inputs: str | ScopedQuestion = (
        question
        if sources is None
        else ScopedQuestion(question=question, sources=list(sources))
    )
    metadata = {
        "question_id": str(row["id"]),
        "family": str(row["family"]),
        "case_index": str(index),
        "answerability": row["answerability"],
    }
    for key in (
        "expected_value",
        "expected_values",
        "distractor_value",
        "expected_ordered",
        "forbidden_text",
        "expected_sources",
        "scope",
    ):
        if row[key] is not None:
            metadata[key] = row[key]
    return Case(
        name=f"{index}_{row['id']}",
        inputs=inputs,
        expected_output=None,
        metadata=metadata,
    )


def _unused_document_loader() -> Dataset:  # pragma: no cover - never called
    raise RuntimeError(
        "the multi-database corpus is built by build_databases(); run with --skip-db"
    )


def print_gate_report(cases: list[Any]) -> None:
    """Print the hard gates, per-family rates and surface coverage.

    Emits a greppable `GATES: PASSED` / `GATES: FAILED` line, since this dataset
    is an acceptance gate rather than a rate to watch drift on.
    """
    text, _passed = render_gate_report(cases)
    print(text)


MULTIDB_SPEC = DatasetSpec(
    key="multidb",
    db_filename="multidb_northern.lancedb",
    document_loader=_unused_document_loader,
    document_mapper=lambda _row: None,
    qa_loader=load_behaviour_questions,
    qa_case_builder=build_multidb_case,
    qa_evaluator=MultiDBScores(),
    report_hook=print_gate_report,
)

MULTIDB_SURFACES_SPEC = DatasetSpec(
    key="multidb_surfaces",
    db_filename="multidb_northern.lancedb",
    document_loader=_unused_document_loader,
    document_mapper=lambda _row: None,
    qa_loader=load_surface_questions,
    qa_case_builder=build_multidb_case,
    qa_evaluator=MultiDBScores(),
    report_hook=print_gate_report,
)

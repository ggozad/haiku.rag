import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

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

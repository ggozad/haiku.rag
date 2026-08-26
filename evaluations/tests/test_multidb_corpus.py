import pytest

from evaluations.datasets.multidb import (
    DATABASE_NAMES,
    EQUIPMENT,
    MONTHS,
    NORTHERN,
    SOUTHERN,
    build_databases,
    corpus,
    documents_for,
    render_station_report,
    station,
    stations_in,
)
from haiku.rag.config.models import AppConfig, LanceDBConfig


def test_document_counts_differ_per_database():
    """S1 asks how many documents a database holds. Equal counts would let a
    wrong-attribution answer pass, so the three differ."""
    counts = {name: len(documents_for(name)) for name in DATABASE_NAMES}
    assert counts == {NORTHERN: 9, SOUTHERN: 8, EQUIPMENT: 6}
    assert len(set(counts.values())) == len(counts)
    assert len(corpus()) == sum(counts.values())


def test_elevations_and_years_identify_one_station():
    """Gold answers are numbers, so a number must not be ambiguous across the
    corpus."""
    from evaluations.datasets.multidb import STATIONS

    assert len({s.elevation_m for s in STATIONS}) == len(STATIONS)
    assert len({s.commissioned for s in STATIONS}) == len(STATIONS)


def test_near_name_pairs_differ_only_in_name_and_numbers():
    """B3's value is that the pair is otherwise identical; a difference in
    instrument or technician would give the model a free discriminator."""
    from evaluations.datasets.multidb import NEAR_NAME_PAIRS

    for north_name, south_name in NEAR_NAME_PAIRS:
        north, south = station(north_name, NORTHERN), station(south_name, SOUTHERN)
        assert north.instrument == south.instrument
        assert north.technician == south.technician
        assert north.elevation_m != south.elevation_m
        assert north.commissioned != south.commissioned


def test_near_name_stations_disagree_on_elevation():
    """B3 depends on the wrong database yielding a wrong number, so the pair must
    never share an elevation."""
    assert station("Kestrel", NORTHERN).elevation_m == 1240
    assert station("Kestrel Ridge", SOUTHERN).elevation_m == 2310


def test_shared_entity_differs_between_databases():
    """Station Auk exists in both. B5 needs the two to be distinguishable, and S3
    needs their readings to differ or the expected total is ambiguous."""
    northern, southern = station("Auk", NORTHERN), station("Auk", SOUTHERN)
    assert northern.commissioned != southern.commissioned
    assert northern.readings != southern.readings
    assert northern.uri != southern.uri


def test_readings_are_stable():
    """Gold answers are derived from these, so a change to the derivation silently
    rewrites every expected total. Pinned deliberately."""
    assert station("Kestrel", NORTHERN).readings_total == 689
    assert station("Auk", NORTHERN).readings_total == 756
    assert station("Auk", SOUTHERN).readings_total == 653


def test_station_report_has_four_sections_and_twelve_rows():
    report = render_station_report(station("Kestrel", NORTHERN))
    for heading in (
        "## Overview",
        "## Instruments",
        "## Measurements",
        "## Maintenance",
    ):
        assert heading in report
    for month in MONTHS:
        assert f"| {month} |" in report


def test_equipment_shares_vocabulary_but_answers_nothing():
    """The dilution probe only works if the sheets compete on wording while
    holding no station facts."""
    sheets = documents_for(EQUIPMENT)
    assert all("anemometer" in d.content for d in sheets)
    assert all("calibration" in d.content.lower() for d in sheets)
    station_names = {s.name for s in stations_in(NORTHERN)} | {
        s.name for s in stations_in(SOUTHERN)
    }
    for sheet in sheets:
        assert not any(f"Station {name}" in sheet.content for name in station_names)
        assert "Programme" not in sheet.content


def test_every_station_is_reachable_by_name_and_database():
    for s in (*stations_in(NORTHERN), *stations_in(SOUTHERN)):
        assert station(s.name, s.database) is s
    with pytest.raises(KeyError):
        station("Kestrel", SOUTHERN)


async def test_build_refuses_a_config_missing_a_database():
    """Building into a config that does not place all three would write a corpus
    the run cannot read."""
    config = AppConfig(
        lancedb=LanceDBConfig(databases={NORTHERN: "/tmp/n", SOUTHERN: "/tmp/s"})
    )
    with pytest.raises(ValueError, match="equipment"):
        await build_databases(config)

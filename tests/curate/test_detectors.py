import numpy as np
import pytest

from haiku.rag.config.models import CurateConfig, CurateThresholdsConfig
from haiku.rag.curate.detectors import (
    detect_across,
    detect_database,
    isolation_scores,
)
from haiku.rag.curate.store.models import (
    DatabaseView,
    FlagKind,
    RepeatedText,
    Revision,
)


def _centroid(*axes: float) -> bytes:
    vector = np.zeros(8, dtype=np.float32)
    vector[: len(axes)] = axes
    return (vector / np.linalg.norm(vector)).astype("<f4").tobytes()


SAME = _centroid(1.0)


def _revision(id: int, **overrides) -> Revision:
    values = {
        "id": id,
        "database": "wiki",
        "document_id": f"d{id}",
        "subject": f"file:///d{id}",
        "md5": None,
        "embedder": "ollama:test:8",
        "centroid": _centroid(*([0.0] * (id % 8)), 1.0),
        "chars": 1000,
        "chunks": 10,
        "embedded_chunks": 10,
        "replacement_chars": 0,
        "metadata_keys": [],
        "became_current_at": "2026-09-29T10:00:00+00:00",
        "ended_at": None,
        "deleted": False,
    }
    values.update(overrides)
    return Revision(**values)


def _view(current, previous=None, **overrides) -> DatabaseView:
    values = {
        "database": "wiki",
        "current": current,
        "previous": previous or {},
        "deletions": [],
        "watched": {},
        "repeated": [],
    }
    values.update(overrides)
    return DatabaseView(**values)


def _kinds(detections) -> list[FlagKind]:
    return sorted(d.kind for d in detections)


def _config(**thresholds) -> CurateConfig:
    return CurateConfig(thresholds=CurateThresholdsConfig(**thresholds))


def test_clean_documents_raise_nothing():
    assert detect_database(_view([_revision(1), _revision(2)]), _config()) == []


class TestBadUpdate:
    def _update(self, config=None, **new):
        previous = _revision(1, ended_at="2026-09-29T11:00:00+00:00", centroid=SAME)
        new.setdefault("centroid", SAME)
        current = _revision(2, document_id="d1", subject="file:///d1", **new)
        detections = detect_database(
            _view([current], {2: previous}), config or _config()
        )
        return [d for d in detections if d.kind is FlagKind.BAD_UPDATE]

    def test_unchanged_update_raises_nothing(self):
        assert self._update() == []

    def test_centroid_moving_away_is_flagged(self):
        [flag] = self._update(centroid=_centroid(0.5, 1.0))
        assert flag.fingerprint_id == 2 and flag.previous_fingerprint_id == 1
        assert flag.subject == "file:///d1"
        [reason] = flag.reasons
        assert reason["metric"] == "centroid_cosine"
        assert reason["value"] == pytest.approx(0.5 / np.hypot(0.5, 1.0), rel=1e-5)
        assert reason["threshold"] == 0.85

    def test_centroid_within_threshold_is_not_flagged(self):
        assert self._update(centroid=_centroid(1.0, 0.5)) == []

    def test_cosine_is_skipped_across_an_embedder_change(self):
        assert self._update(centroid=_centroid(0.0, 1.0), embedder="other:x:8") == []

    def test_missing_centroid_skips_the_cosine(self):
        assert self._update(centroid=None) == []

    @pytest.mark.parametrize(
        "chars,flagged", [(3000, False), (3001, True), (333, True)]
    )
    def test_size_change(self, chars, flagged):
        flags = self._update(chars=chars)
        assert bool(flags) is flagged
        if flagged:
            assert flags[0].reasons[0]["metric"] == "chars_ratio"

    def test_chunks_dropping_to_zero_is_flagged(self):
        [flag] = self._update(chunks=0, embedded_chunks=0, centroid=None)
        assert flag.reasons[0]["metric"] == "chunks_ratio"
        assert flag.reasons[0]["value"] is None

    def test_update_flag_replaces_the_document_flag(self):
        previous = _revision(1, ended_at="x", centroid=SAME)
        current = _revision(
            2,
            subject="file:///d1",
            centroid=_centroid(0.0, 1.0),
            replacement_chars=900,
        )
        detections = detect_database(_view([current], {2: previous}), _config())
        assert _kinds(detections) == [FlagKind.BAD_UPDATE]


class TestBadDocument:
    @pytest.mark.parametrize("replacements,flagged", [(200, False), (201, True)])
    def test_replacement_character_rate(self, replacements, flagged):
        view = _view([_revision(1, replacement_chars=replacements)])
        flags = detect_database(view, _config())
        assert bool(flags) is flagged
        if flagged:
            assert flags[0].reasons == [
                {
                    "metric": "replacement_chars_per_1k",
                    "value": 201.0,
                    "previous": None,
                    "threshold": 200.0,
                }
            ]

    def test_empty_text_has_no_replacement_rate(self):
        assert detect_database(_view([_revision(1, chars=0)]), _config()) == []

    def test_no_embedded_chunks(self):
        [flag] = detect_database(
            _view([_revision(1, embedded_chunks=0, centroid=None)]), _config()
        )
        assert flag.kind is FlagKind.BAD_DOCUMENT
        assert flag.reasons == [
            {"metric": "embedded_chunks", "value": 0, "previous": None, "threshold": 1}
        ]


class TestWatchedChange:
    def test_revision_after_the_watch_is_flagged(self):
        view = _view(
            [_revision(1, became_current_at="2026-09-29T12:00:00+00:00")],
            watched={"file:///d1": "2026-09-29T11:00:00+00:00"},
        )
        [flag] = detect_database(view, _config())
        assert flag.kind is FlagKind.WATCHED_CHANGE
        assert flag.fingerprint_id == 1

    def test_revision_before_the_watch_is_not_flagged(self):
        view = _view(
            [_revision(1, became_current_at="2026-09-29T10:00:00+00:00")],
            watched={"file:///d1": "2026-09-29T11:00:00+00:00"},
        )
        assert detect_database(view, _config()) == []

    def test_deletion_after_the_watch_is_flagged(self):
        gone = _revision(1, ended_at="2026-09-29T12:00:00+00:00", deleted=True)
        view = _view(
            [],
            deletions=[gone],
            watched={"file:///d1": "2026-09-29T11:00:00+00:00"},
        )
        [flag] = detect_database(view, _config())
        assert flag.kind is FlagKind.WATCHED_DELETION
        assert flag.fingerprint_id == 1

    def test_watched_change_raises_alongside_document_flags(self):
        view = _view(
            [
                _revision(
                    1,
                    became_current_at="2026-09-29T12:00:00+00:00",
                    replacement_chars=900,
                )
            ],
            watched={"file:///d1": "2026-09-29T11:00:00+00:00"},
        )
        kinds = _kinds(detect_database(view, _config()))
        assert kinds == sorted([FlagKind.BAD_DOCUMENT, FlagKind.WATCHED_CHANGE])


def test_near_identical_documents_form_a_duplicate_group():
    view = _view(
        [_revision(1, centroid=SAME), _revision(2, centroid=SAME), _revision(3)]
    )
    [flag] = detect_database(view, _config())
    assert flag.kind is FlagKind.DUPLICATE_GROUP
    assert flag.members == [
        {"database": "wiki", "document_id": "d1"},
        {"database": "wiki", "document_id": "d2"},
    ]
    assert [r["document_id"] for r in flag.reasons] == ["d1", "d2"]


def test_duplicate_group_identity_follows_membership():
    pair = detect_database(
        _view([_revision(n, centroid=SAME) for n in (1, 2)]), _config()
    )
    trio = detect_database(
        _view([_revision(n, centroid=SAME) for n in (1, 2, 4)]), _config()
    )
    assert pair[0].identity != trio[0].identity


def test_repeated_chunk_text():
    view = _view(
        [_revision(1, centroid=None, embedded_chunks=1)],
        repeated=[RepeatedText("h1", 40, ["d1", "d2", "d3", "d4", "d5"])],
    )
    flags = [
        d for d in detect_database(view, _config()) if d.kind is FlagKind.REPEATED_CHUNK
    ]
    [flag] = flags
    assert flag.subject == "h1"
    assert flag.members is not None and len(flag.members) == 5
    assert flag.reasons == [
        {"metric": "documents", "value": 5, "previous": None, "threshold": 5},
        {"metric": "chars", "value": 40, "previous": None, "threshold": 20},
    ]


def test_repeated_chunk_identity_ignores_membership():
    config = _config()
    one = detect_database(
        _view([], repeated=[RepeatedText("h1", 40, ["d1", "d2"])]), config
    )
    other = detect_database(
        _view([], repeated=[RepeatedText("h1", 40, ["d1", "d2", "d3"])]), config
    )
    assert one[0].identity == other[0].identity


def test_missing_required_metadata():
    config = CurateConfig(required_metadata=["department", "classification"])
    view = _view(
        [
            _revision(1, metadata_keys=["department"]),
            _revision(2, metadata_keys=["classification", "department"]),
        ]
    )
    [flag] = detect_database(view, config)
    assert flag.kind is FlagKind.MISSING_METADATA
    assert flag.fingerprint_id == 1
    assert flag.reasons == [
        {
            "metric": "missing_key",
            "value": "classification",
            "previous": None,
            "threshold": None,
        }
    ]


def test_exact_copies_across_databases():
    revisions = [
        _revision(1, md5="m1"),
        _revision(2, database="papers", md5="m1"),
        _revision(3, database="wiki", md5="m2"),
        _revision(4, database="wiki", md5="m2"),
        _revision(5, md5=None),
    ]
    [flag] = detect_across(revisions)
    assert flag.kind is FlagKind.DUPLICATE_GROUP
    assert flag.database is None
    assert flag.members == [
        {"database": "papers", "document_id": "d2"},
        {"database": "wiki", "document_id": "d1"},
    ]
    assert flag.reasons == [
        {"metric": "md5", "value": "m1", "previous": None, "threshold": None}
    ]


def test_isolation_scores():
    revisions = [
        _revision(1, centroid=SAME),
        _revision(2, centroid=SAME),
        _revision(3, centroid=_centroid(0.0, 1.0)),
        _revision(4, centroid=None),
    ]
    scores = isolation_scores(revisions, neighbours=1)
    assert scores["d1"] == pytest.approx(0.0, abs=1e-6)
    assert scores["d3"] == pytest.approx(1.0, abs=1e-6)
    assert scores["d4"] is None


def test_isolation_of_a_single_document_is_undefined():
    assert isolation_scores([_revision(1)], neighbours=5) == {"d1": None}

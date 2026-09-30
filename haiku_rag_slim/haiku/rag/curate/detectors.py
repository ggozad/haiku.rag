from collections import defaultdict

import numpy as np

from haiku.rag.config.models import CurateConfig, CurateThresholdsConfig
from haiku.rag.curate.store.models import (
    DatabaseView,
    Detection,
    FlagKind,
    Revision,
)
from haiku.rag.similarity import duplicate_families


def _reason(metric: str, value, previous=None, threshold=None) -> dict[str, object]:
    return {
        "metric": metric,
        "value": value,
        "previous": previous,
        "threshold": threshold,
    }


def _vector(revision: Revision) -> np.ndarray | None:
    if revision.centroid is None:
        return None
    return np.frombuffer(revision.centroid, dtype="<f4")


def _embedded(revisions: list[Revision]) -> list[tuple[Revision, np.ndarray]]:
    return [(r, v) for r in revisions if (v := _vector(r)) is not None]


def _replacement_rate(revision: Revision) -> float:
    if revision.chars == 0:
        return 0.0
    return revision.replacement_chars / revision.chars * 1000


def _ratio(new: int, old: int) -> float | None:
    """How many times larger the larger of the two is; None when only one is zero."""
    if new == old:
        return 1.0
    if new == 0 or old == 0:
        return None
    return max(new, old) / min(new, old)


def _update_reasons(
    current: Revision, previous: Revision, thresholds: CurateThresholdsConfig
) -> list[dict]:
    reasons = []
    new_vector, old_vector = _vector(current), _vector(previous)
    if (
        new_vector is not None
        and old_vector is not None
        and current.embedder == previous.embedder
    ):
        cosine = float(new_vector @ old_vector)
        if cosine < thresholds.update_cosine:
            reasons.append(
                _reason("centroid_cosine", cosine, 1.0, thresholds.update_cosine)
            )
    for metric, new, old in (
        ("chars_ratio", current.chars, previous.chars),
        ("chunks_ratio", current.chunks, previous.chunks),
    ):
        ratio = _ratio(new, old)
        if ratio is None or ratio > thresholds.size_factor:
            reasons.append(_reason(metric, ratio, old, thresholds.size_factor))
    return reasons


def _document_reasons(
    revision: Revision, thresholds: CurateThresholdsConfig
) -> list[dict]:
    reasons = []
    rate = _replacement_rate(revision)
    if rate > thresholds.replacement_chars_per_1k:
        reasons.append(
            _reason(
                "replacement_chars_per_1k",
                rate,
                threshold=thresholds.replacement_chars_per_1k,
            )
        )
    if revision.embedded_chunks == 0:
        reasons.append(_reason("embedded_chunks", 0, threshold=1))
    return reasons


def _document_detection(
    kind: FlagKind,
    revision: Revision,
    reasons: list[dict],
    previous: Revision | None = None,
) -> Detection:
    return Detection(
        kind=kind,
        database=revision.database,
        subject=revision.subject,
        fingerprint_id=revision.id,
        previous_fingerprint_id=previous.id if previous is not None else None,
        reasons=reasons,
    )


def detect_database(view: DatabaseView, config: CurateConfig) -> list[Detection]:
    """Every condition the store shows in one database."""
    thresholds = config.thresholds
    detections = []
    for revision in view.current:
        previous = view.previous.get(revision.id)
        update = (
            _update_reasons(revision, previous, thresholds)
            if previous is not None
            else []
        )
        if update:
            detections.append(
                _document_detection(FlagKind.BAD_UPDATE, revision, update, previous)
            )
        elif reasons := _document_reasons(revision, thresholds):
            detections.append(
                _document_detection(FlagKind.BAD_DOCUMENT, revision, reasons)
            )
        watched_at = view.watched.get(revision.subject)
        if watched_at is not None and revision.became_current_at > watched_at:
            detections.append(
                _document_detection(FlagKind.WATCHED_CHANGE, revision, [], previous)
            )
        missing = [
            key for key in config.required_metadata if key not in revision.metadata_keys
        ]
        if missing:
            detections.append(
                _document_detection(
                    FlagKind.MISSING_METADATA,
                    revision,
                    [_reason("missing_key", key) for key in missing],
                )
            )
    for gone in view.deletions:
        watched_at = view.watched.get(gone.subject)
        if watched_at is not None and gone.ended_at and gone.ended_at > watched_at:
            detections.append(_document_detection(FlagKind.WATCHED_DELETION, gone, []))
    detections += _duplicate_groups(view, config)
    for repeated in view.repeated:
        detections.append(
            Detection(
                kind=FlagKind.REPEATED_CHUNK,
                database=view.database,
                subject=repeated.text_hash,
                members=[
                    {"database": view.database, "document_id": document_id}
                    for document_id in sorted(repeated.document_ids)
                ],
                reasons=[
                    _reason(
                        "documents",
                        len(repeated.document_ids),
                        threshold=config.repeated_chunks.min_documents,
                    ),
                    _reason(
                        "chars",
                        repeated.chars,
                        threshold=config.repeated_chunks.min_chars,
                    ),
                ],
            )
        )
    return detections


def _duplicate_groups(view: DatabaseView, config: CurateConfig) -> list[Detection]:
    embedded = _embedded(view.current)
    if len(embedded) < 2:
        return []
    families = duplicate_families(
        [r.document_id for r, _ in embedded],
        np.stack([v for _, v in embedded]),
        np.array([r.embedded_chunks for r, _ in embedded]),
        config.duplicates,
    )
    return [
        Detection(
            kind=FlagKind.DUPLICATE_GROUP,
            database=view.database,
            members=[
                {"database": view.database, "document_id": member}
                for member in family.members
            ],
            reasons=[
                {
                    "metric": "similarity",
                    "document_id": member,
                    "value": family.similarity[member],
                    "threshold": config.duplicates.similarity_threshold,
                }
                for member in family.members
            ],
        )
        for family in families
    ]


def detect_across(current: list[Revision]) -> list[Detection]:
    """Exact copies, by md5, of one file in more than one database."""
    by_md5: dict[str, list[Revision]] = defaultdict(list)
    for revision in current:
        if revision.md5 is not None:
            by_md5[revision.md5].append(revision)
    detections = []
    for md5, revisions in sorted(by_md5.items()):
        if len({r.database for r in revisions}) < 2:
            continue
        detections.append(
            Detection(
                kind=FlagKind.DUPLICATE_GROUP,
                database=None,
                members=[
                    {"database": database, "document_id": document_id}
                    for database, document_id in sorted(
                        (r.database, r.document_id) for r in revisions
                    )
                ],
                reasons=[_reason("md5", md5)],
            )
        )
    return detections


def isolation_scores(
    current: list[Revision], neighbours: int
) -> dict[str, float | None]:
    """One minus the mean centroid cosine to each document's nearest neighbours."""
    scores: dict[str, float | None] = {r.document_id: None for r in current}
    embedded = _embedded(current)
    k = min(neighbours, len(embedded) - 1)
    if k < 1:
        return scores
    unit = np.stack([v for _, v in embedded])
    block = 512
    for start in range(0, len(embedded), block):
        sims = unit[start : start + block] @ unit.T
        for row in range(sims.shape[0]):
            sims[row, start + row] = -np.inf
        nearest = np.partition(sims, -k, axis=1)[:, -k:]
        for offset, value in enumerate(1 - nearest.mean(axis=1)):
            scores[embedded[start + offset][0].document_id] = float(value)
    return scores

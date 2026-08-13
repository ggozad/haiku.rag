import pytest

from haiku.rag.capabilities.ledger import (
    CapabilityEvidenceRecord,
    CitationDeclaration,
    EvidenceRef,
    citation_status,
)


def rag_ref(chunk_id: str = "c1") -> EvidenceRef:
    return EvidenceRef(capability="rag", chunk_id=chunk_id)


def test_a_record_survives_the_state_round_trip():
    """Capability state is persisted as JSON, so the schema must survive it.

    A dict keyed by ``(capability, chunk_id)`` does not: the key serialises to
    ``"rag,c1"`` and fails revalidation as a tuple.
    """
    record = CapabilityEvidenceRecord(question=4)
    record.note_evidence(5)
    record.declare([rag_ref()], epoch=7, retrieved_now={"c1"})

    restored = CapabilityEvidenceRecord.model_validate(record.model_dump(mode="json"))

    assert restored == record
    assert citation_status([restored], question=4) == "grounded"
    assert restored.occurrences["c1"].cited_in_questions == [4]
    assert restored.occurrences["c1"].retrieved_in_questions == [4]


def test_no_declaration_reads_as_missing():
    assert citation_status([CapabilityEvidenceRecord()], question=0) == "missing"
    assert citation_status([], question=0) == "missing"


def test_refs_make_it_grounded_and_no_refs_make_it_ungrounded():
    grounded = CapabilityEvidenceRecord(question=0)
    grounded.declare([rag_ref()], epoch=1)

    ungrounded = CapabilityEvidenceRecord(question=0)
    ungrounded.declare([], epoch=1)

    assert citation_status([grounded], question=0) == "grounded"
    assert citation_status([ungrounded], question=0) == "ungrounded"


def test_an_earlier_questions_declaration_is_never_current():
    """A later question inherits nothing: it has declared nothing yet."""
    record = CapabilityEvidenceRecord(question=2)
    record.declare([rag_ref()], epoch=3)
    assert citation_status([record], question=2) == "grounded"

    record.begin_question(8)

    assert citation_status([record], question=8) == "missing"


def test_a_citation_in_the_same_request_as_the_evidence_is_not_current():
    """Citing must follow seeing: equal epochs mean one request."""
    record = CapabilityEvidenceRecord(question=0)
    record.note_evidence(5)
    record.declare([rag_ref()], epoch=5)

    assert citation_status([record], question=0) == "missing"

    record.declare([rag_ref()], epoch=7)

    assert citation_status([record], question=0) == "grounded"


def test_evidence_from_another_capability_after_citing_makes_it_uncited():
    """Currency spans capabilities, which only works because epochs are global."""
    cited = CapabilityEvidenceRecord(question=0)
    cited.note_evidence(3)
    cited.declare([rag_ref()], epoch=5)
    searched_after = CapabilityEvidenceRecord(question=0)
    searched_after.note_evidence(7)

    assert citation_status([cited], question=0) == "grounded"
    assert citation_status([cited, searched_after], question=0) == "missing"


def test_declarations_at_the_same_epoch_merge():
    record = CapabilityEvidenceRecord(question=0)
    record.declare([rag_ref("c1")], epoch=3)
    record.declare([rag_ref("c2")], epoch=3)

    assert record.declaration is not None
    assert [ref.chunk_id for ref in record.declaration.refs] == ["c1", "c2"]


def test_repeating_a_ref_at_the_same_epoch_does_not_duplicate_it():
    record = CapabilityEvidenceRecord(question=0)
    record.declare([rag_ref()], epoch=3)
    record.declare([rag_ref()], epoch=3)

    assert record.declaration is not None
    assert len(record.declaration.refs) == 1


def test_neither_cite_order_downgrades_a_grounded_declaration():
    grounded_then_empty = CapabilityEvidenceRecord(question=0)
    grounded_then_empty.declare([rag_ref()], epoch=3)
    grounded_then_empty.declare([], epoch=3)

    empty_then_grounded = CapabilityEvidenceRecord(question=0)
    empty_then_grounded.declare([], epoch=3)
    empty_then_grounded.declare([rag_ref()], epoch=3)

    assert citation_status([grounded_then_empty], question=0) == "grounded"
    assert citation_status([empty_then_grounded], question=0) == "grounded"


def test_the_same_chunk_id_under_two_capabilities_stays_separate():
    rag = CapabilityEvidenceRecord(question=0)
    rag.declare([EvidenceRef(capability="rag", chunk_id="shared")], epoch=3)
    analysis = CapabilityEvidenceRecord(question=0)
    analysis.declare([EvidenceRef(capability="analysis", chunk_id="shared")], epoch=3)

    assert rag.occurrences["shared"].capability == "rag"
    assert analysis.occurrences["shared"].capability == "analysis"


def test_citing_the_same_chunk_in_two_questions_records_both():
    record = CapabilityEvidenceRecord(question=2)
    record.declare([rag_ref()], epoch=3, retrieved_now={"c1"})
    record.begin_question(8)
    record.declare([rag_ref()], epoch=9)

    occurrence = record.occurrences["c1"]
    assert occurrence.cited_in_questions == [2, 8]
    assert occurrence.retrieved_in_questions == [2]


def test_a_declaration_records_the_question_and_epoch_it_was_made_at():
    record = CapabilityEvidenceRecord(question=6)
    record.declare([rag_ref()], epoch=11)

    assert record.declaration == CitationDeclaration(
        question=6, epoch=11, refs=[rag_ref()]
    )


def test_a_fresh_record_has_no_question_identity():
    """The identity is established by the run, and its absence must be detectable.

    A default record is truthy, so its mere presence cannot stand in for having
    been through ``for_run``: a host that seeds one would otherwise pass the
    resumption check with a fabricated identity of zero.
    """
    assert CapabilityEvidenceRecord().question is None


def test_evidence_cannot_move_backwards_in_the_conversation():
    """Epochs are message counts, and currency depends on them only growing.

    Silently keeping the newer value would leave every later declaration stale
    for the rest of the conversation, permanently and invisibly.
    """
    record = CapabilityEvidenceRecord(question=0)
    record.note_evidence(9)

    with pytest.raises(ValueError, match="append-only"):
        record.note_evidence(4)


def test_citing_before_a_run_establishes_the_question_is_refused():
    with pytest.raises(ValueError, match="question identity"):
        CapabilityEvidenceRecord().declare([rag_ref()], epoch=3)


def test_a_declaration_cannot_move_backwards():
    """Otherwise a stale citation replaces a newer one and revives the answer."""
    record = CapabilityEvidenceRecord(question=0)
    record.declare([rag_ref("newer")], epoch=5)

    with pytest.raises(ValueError, match="append-only"):
        record.declare([rag_ref("older")], epoch=3)

    assert record.declaration is not None
    assert [ref.chunk_id for ref in record.declaration.refs] == ["newer"]


def test_evidence_cannot_predate_a_recorded_declaration():
    """The declaration's epoch is a recorded message count like any other."""
    record = CapabilityEvidenceRecord(question=0)
    record.declare([rag_ref()], epoch=5)

    with pytest.raises(ValueError, match="append-only"):
        record.note_evidence(3)


def test_a_question_starts_behind_the_epochs_of_the_one_before_it():
    """A question's identity is the history it arrives on, not a continuation.

    Epochs are compared only within the question that recorded them, so a host
    whose stored history shifted between two questions is answered rather than
    refused.
    """
    record = CapabilityEvidenceRecord(question=0)
    record.note_evidence(9)

    record.begin_question(4)

    assert record.question == 4


def test_a_question_cannot_reuse_the_identity_of_the_one_before_it():
    """Occurrences outlive their question and are ordered by identity.

    Two questions sharing one identity merge into a single capsule group, and a
    lower one is rendered as though its evidence were cited earlier.
    """
    record = CapabilityEvidenceRecord(question=4)

    with pytest.raises(ValueError, match="already answered"):
        record.begin_question(4)

    with pytest.raises(ValueError, match="already answered"):
        record.begin_question(3)

    assert record.question == 4


def test_a_question_starts_clear_of_the_one_before_it():
    """Evidence and declarations describe a single question and end with it.

    Carrying either into the next question makes it answerable by the last
    question's citations, and freezes its own declarations behind an epoch no
    message in it can reach.
    """
    record = CapabilityEvidenceRecord(question=4)
    record.note_evidence(6)
    record.declare([rag_ref()], epoch=7)

    record.begin_question(9)

    assert record.latest_evidence_epoch == 0
    assert record.declaration is None
    assert citation_status([record], question=9) == "missing"

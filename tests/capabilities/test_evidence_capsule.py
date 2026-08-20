from dataclasses import dataclass, field, replace
from typing import Any
from unittest.mock import patch

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from haiku.rag.capabilities.analysis import create_capability as create_analysis
from haiku.rag.capabilities.compaction import (
    CAPSULE_HEADER,
    EvidenceCompactionCapability,
    build_capsule,
    group_label,
)
from haiku.rag.capabilities.compaction import create_capability as create_compaction
from haiku.rag.capabilities.evidence import DiscoveredEvidence, discover_evidence
from haiku.rag.capabilities.ledger import (
    CapabilityEvidenceRecord,
    EvidenceOccurrence,
)
from haiku.rag.capabilities.rag import create_capability as create_rag
from haiku.rag.config.models import AppConfig
from haiku.rag.store.models.citation import Citation


@dataclass
class Deps:
    state: dict[str, Any] = field(default_factory=dict)


def citation(chunk_id: str, content: str = "evidence body", pictures=()) -> Citation:
    return Citation(
        document_id=f"doc-of-{chunk_id}",
        chunk_id=chunk_id,
        document_uri=f"test://{chunk_id}",
        document_title=f"Title {chunk_id}",
        content=content,
        picture_refs=list(pictures),
    )


def replace_citation(cited: Citation, **changes: Any) -> Citation:
    return cited.model_copy(update=changes)


def discovered(
    capability: str = "rag",
    *,
    cited: dict[str, list[int]] | None = None,
    contents: dict[str, str] | None = None,
    pictures: dict[str, list[str]] | None = None,
) -> DiscoveredEvidence:
    """One capability's records, as the compactor would find them."""
    cited = cited or {}
    contents = contents or {}
    pictures = pictures or {}
    record = CapabilityEvidenceRecord(question=max(max(cited.values(), default=[0])))
    for chunk_id, questions in cited.items():
        record.occurrences[chunk_id] = EvidenceOccurrence(
            capability=capability,
            chunk_id=chunk_id,
            retrieved_in_questions=list(questions),
            cited_in_questions=list(questions),
        )
    return DiscoveredEvidence(
        capability=capability,
        record=record,
        state_carried=True,
        citations={
            chunk_id: citation(
                chunk_id,
                contents.get(chunk_id, "evidence body"),
                pictures.get(chunk_id, ()),
            )
            for chunk_id in cited
        },
        tool_names=frozenset({f"{capability}_search"}),
        cite_available=True,
    )


def test_a_retained_picture_carries_the_source_it_came_from():
    """Compaction re-fetches cited pictures later, so the capsule has to remember
    which database each came from."""
    found = discovered(cited={"c1": [2]}, pictures={"c1": ["#/pictures/0"]})
    found = replace(
        found,
        citations={"c1": replace_citation(found.citations["c1"], source="beta")},
    )

    capsule = build_capsule([found])

    [picture] = capsule.pictures
    assert picture.source == "beta"


def test_nothing_cited_produces_no_capsule():
    capsule = build_capsule([discovered()])

    assert capsule.text == ""
    assert capsule.pictures == ()


def test_cited_evidence_is_grouped_newest_question_first():
    capsule = build_capsule([discovered(cited={"old": [2], "new": [8]})])

    assert capsule.text.index(group_label(1)) < capsule.text.index(group_label(2))
    assert capsule.text.index("[new]") < capsule.text.index("[old]")
    assert CAPSULE_HEADER in capsule.text


def test_an_entry_is_rendered_once_in_its_most_recent_citing_group():
    capsule = build_capsule([discovered(cited={"reused": [2, 8], "only-old": [2]})])

    assert capsule.text.count("[reused]") == 1
    assert capsule.text.index("[reused]") < capsule.text.index("[only-old]")


def test_evidence_cited_in_one_question_forms_one_group():
    capsule = build_capsule([discovered(cited={"a": [4], "b": [4]})])

    assert group_label(1) in capsule.text
    assert group_label(2) not in capsule.text


def test_every_cited_entry_is_kept_whole():
    """No budget: a long citation is retained in full rather than truncated."""
    body = "L" * 20_000
    capsule = build_capsule([discovered(cited={"long": [4]}, contents={"long": body})])

    assert body in capsule.text


def test_both_capabilities_share_one_capsule():
    capsule = build_capsule(
        [
            discovered("rag", cited={"from-rag": [4]}),
            discovered("analysis", cited={"from-analysis": [6]}),
        ]
    )

    assert capsule.text.count(CAPSULE_HEADER) == 1
    assert "[from-rag]" in capsule.text
    assert "[from-analysis]" in capsule.text


def test_the_same_chunk_id_under_two_capabilities_is_kept_apart():
    capsule = build_capsule(
        [
            discovered("rag", cited={"shared": [4]}, contents={"shared": "rag body"}),
            discovered(
                "analysis", cited={"shared": [4]}, contents={"shared": "analysis body"}
            ),
        ]
    )

    assert "rag body" in capsule.text
    assert "analysis body" in capsule.text


def test_cited_evidence_with_no_canonical_citation_is_an_error():
    """Both are written by the same call, so divergence is not a valid state.

    Rendering the rest would quietly drop evidence an answer rested on, against
    the one guarantee this capsule makes.
    """
    evidence = discovered(cited={"present": [4]})
    evidence.record.occurrences["absent"] = EvidenceOccurrence(
        capability="rag", chunk_id="absent", cited_in_questions=[4]
    )

    with pytest.raises(ValueError, match="absent"):
        build_capsule([evidence])


def test_retrieved_but_uncited_evidence_is_not_kept():
    evidence = discovered(cited={"cited": [4]})
    evidence.record.occurrences["seen-only"] = EvidenceOccurrence(
        capability="rag", chunk_id="seen-only", retrieved_in_questions=[4]
    )

    capsule = build_capsule([evidence])

    assert "[cited]" in capsule.text
    assert "seen-only" not in capsule.text


def test_a_source_is_named_once_when_the_title_is_the_uri():
    """Real corpora set both to the document id, which reads as a stutter."""
    evidence = discovered(cited={"a": [4]})
    evidence.citations["a"].document_title = "2410.11843v5"
    evidence.citations["a"].document_uri = "2410.11843v5"

    capsule = build_capsule([evidence])

    assert 'Source: "2410.11843v5"' in capsule.text
    assert "2410.11843v5)" not in capsule.text


def test_pictures_of_cited_evidence_are_all_retained_newest_first():
    capsule = build_capsule(
        [
            discovered(
                cited={"a": [2], "c": [6]},
                pictures={"a": ["#/pictures/0"], "c": ["#/pictures/1", "#/pictures/2"]},
            )
        ]
    )

    assert [picture.self_ref for picture in capsule.pictures] == [
        "#/pictures/1",
        "#/pictures/2",
        "#/pictures/0",
    ]
    assert capsule.pictures[0].document_id == "doc-of-c"
    assert capsule.pictures[0].capability == "rag"


def test_a_picture_of_uncited_evidence_is_not_retained():
    found = discovered(cited={"cited": [4]}, pictures={"cited": ["#/pictures/0"]})
    found.record.occurrences["seen-only"] = EvidenceOccurrence(
        capability="rag", chunk_id="seen-only", retrieved_in_questions=[4]
    )
    evidence = replace(
        found,
        citations={
            **found.citations,
            "seen-only": citation("seen-only", pictures=["#/pictures/9"]),
        },
    )

    capsule = build_capsule([evidence])

    assert [picture.self_ref for picture in capsule.pictures] == ["#/pictures/0"]


def test_one_picture_cited_through_two_chunks_is_attached_once():
    """Overlapping chunks of one document share a figure, counted twice by a provider."""
    found = discovered(
        cited={"first": [4], "second": [4]},
        pictures={"first": ["#/pictures/1"], "second": ["#/pictures/1"]},
    )
    shared = {
        chunk_id: replace_citation(cited, document_id="doc-shared")
        for chunk_id, cited in found.citations.items()
    }

    capsule = build_capsule([replace(found, citations=shared)])

    assert len(capsule.pictures) == 1
    assert capsule.pictures[0].chunk_id == "first"


def test_the_same_reference_in_two_documents_is_kept_twice():
    """``#/pictures/1`` means a different figure in a different document.

    One capability throughout, so only the document differs: dropping the document
    from the identity would have to fail this.
    """
    capsule = build_capsule(
        [
            discovered(
                "rag",
                cited={"a": [4], "b": [4]},
                pictures={"a": ["#/pictures/1"], "b": ["#/pictures/1"]},
            )
        ]
    )

    assert len(capsule.pictures) == 2
    assert {picture.capability for picture in capsule.pictures} == {"rag"}
    assert {picture.document_id for picture in capsule.pictures} == {
        "doc-of-a",
        "doc-of-b",
    }


def test_a_picture_label_names_the_chunk_it_belongs_to():
    capsule = build_capsule(
        [discovered(cited={"a": [4]}, pictures={"a": ["#/pictures/0"]})]
    )

    label = capsule.pictures[0].label
    assert "[a]" in label
    assert "#/pictures/0" in label
    assert "knowledge base" in label
    assert "Not provided by the user" in label


def _spy_discovery(found: list[list[DiscoveredEvidence]]):
    """Discover from ``before_run``, the earliest point the registry is reliable."""
    original = EvidenceCompactionCapability.before_run

    async def spy(self, ctx):
        await original(self, ctx)
        found.append(discover_evidence(ctx))

    return patch.object(EvidenceCompactionCapability, "before_run", spy)


async def _answer(_messages, _info):
    return ModelResponse(parts=[TextPart("answer")])


@pytest.mark.asyncio
async def test_the_compactor_discovers_both_evidence_capabilities(temp_db_path):
    """Discovery runs one way through the registry, so nothing needs wiring."""
    compactor = create_compaction()
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    analysis = create_analysis(
        db_path=temp_db_path, config=AppConfig(), defer_loading=False
    )
    found: list[list[DiscoveredEvidence]] = []

    with _spy_discovery(found):
        agent = Agent(
            FunctionModel(_answer),
            deps_type=Deps,
            capabilities=[rag, analysis, compactor],
        )
        await agent.run("a question", deps=Deps())

    assert {evidence.capability: set(evidence.tool_names) for evidence in found[0]} == {
        "rag": {"rag_search"},
        "analysis": {"analysis_search", "analysis_execute_code"},
    }


@pytest.mark.asyncio
async def test_discovery_sees_the_run_instances_not_the_registered_ones(temp_db_path):
    """A registered capability holds no state; only its per-run copy does."""
    compactor = create_compaction()
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)
    found: list[list[DiscoveredEvidence]] = []

    with _spy_discovery(found):
        agent = Agent(
            FunctionModel(_answer), deps_type=Deps, capabilities=[rag, compactor]
        )
        await agent.run("a question", deps=Deps())

    assert rag.state is None
    assert found[0][0].record.question == 0


def test_two_compactors_fail_fast(temp_db_path):
    """Each would rewrite the same history and each would build its own capsule.

    They share this capability's id, so pydantic-ai refuses at construction and
    nothing here has to police it.
    """
    rag = create_rag(db_path=temp_db_path, config=AppConfig(), defer_loading=False)

    with pytest.raises(UserError, match="unique within a run"):
        Agent(
            FunctionModel(_answer),
            deps_type=Deps,
            capabilities=[rag, create_compaction(), create_compaction()],
        )


@pytest.mark.asyncio
async def test_a_compactor_alone_discovers_nothing_and_still_runs():
    found: list[list[DiscoveredEvidence]] = []

    with _spy_discovery(found):
        agent = Agent(
            FunctionModel(_answer), deps_type=Deps, capabilities=[create_compaction()]
        )
        result = await agent.run("a question", deps=Deps())

    assert found == [[]]
    assert result.output == "answer"


@pytest.mark.asyncio
async def test_a_deferred_capability_the_model_never_loaded_has_an_empty_record(
    temp_db_path,
):
    """It is still discovered, because every registered capability gets a run copy.

    Nothing was retrieved under it, so its record contributes no entries and the
    compactor needs no special case for it.
    """
    deferred = create_rag(db_path=temp_db_path, config=AppConfig())
    found: list[list[DiscoveredEvidence]] = []

    with _spy_discovery(found):
        agent = Agent(
            FunctionModel(_answer),
            deps_type=Deps,
            capabilities=[deferred, create_compaction()],
        )
        await agent.run("a question", deps=Deps())

    assert deferred.defer_loading is True
    assert deferred.state is None
    assert [evidence.capability for evidence in found[0]] == ["rag"]
    assert found[0][0].record.occurrences == {}
    assert build_capsule(found[0]).text == ""

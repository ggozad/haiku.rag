import json
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent import AgentSpec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError

from haiku.rag.capabilities.analysis import AnalysisCapability, AnalysisState
from haiku.rag.capabilities.compaction import CAPABILITY_ID as COMPACTION_ID
from haiku.rag.capabilities.compaction import EvidenceCompactionCapability
from haiku.rag.capabilities.policy import CAPABILITY_ID as POLICY_ID
from haiku.rag.capabilities.policy import CitationPolicyCapability
from haiku.rag.capabilities.rag import RAGCapability, RAGState

ALL_CAPABILITIES = [
    RAGCapability,
    AnalysisCapability,
    EvidenceCompactionCapability,
    CitationPolicyCapability,
]


def _from_spec(
    spec: dict, types: Sequence[type[AbstractCapability[Any]]]
) -> list[AbstractCapability[Any]]:
    """Build an agent from a spec and return the capabilities it declared.

    A spec needs a model, and pydantic-ai injects capabilities of its own
    alongside ours.
    """
    agent = Agent.from_spec({"model": "test", **spec}, custom_capability_types=types)
    return [
        capability
        for capability in agent.root_capability.capabilities
        if type(capability) in types
    ]


def test_rag_capability_is_built_from_a_spec(temp_db_path):
    (capability,) = _from_spec(
        {"capabilities": [{"RAGCapability": {"db_path": str(temp_db_path)}}]},
        [RAGCapability],
    )

    assert isinstance(capability, RAGCapability)
    assert capability.db_path == temp_db_path
    assert capability.id == "haiku-rag"
    assert capability.state_type is RAGState
    assert capability.tool_names == {"rag_search", "rag_cite"}
    assert capability.request_limit == 20


def test_analysis_capability_is_built_from_a_spec(temp_db_path):
    (capability,) = _from_spec(
        {"capabilities": [{"AnalysisCapability": {"db_path": str(temp_db_path)}}]},
        [AnalysisCapability],
    )

    assert isinstance(capability, AnalysisCapability)
    assert capability.db_path == temp_db_path
    assert capability.id == "haiku-rag-analysis"
    assert capability.state_type is AnalysisState
    assert capability.request_limit == 30


def test_a_config_mapping_in_a_spec_is_validated(temp_db_path, temp_yaml_config):
    """A `config:` block is validated into AppConfig rather than reaching
    get_config()."""
    (capability,) = _from_spec(
        {
            "capabilities": [
                {
                    "RAGCapability": {
                        "db_path": str(temp_db_path),
                        "config": {"qa": {"max_searches": 9}},
                    }
                }
            ]
        },
        [RAGCapability],
    )

    assert isinstance(capability, RAGCapability)
    assert capability.config.qa.max_searches == 9


@pytest.mark.parametrize("form", ["bare", "empty-mapping"])
def test_the_zero_argument_capabilities_are_built_from_a_spec(form):
    """Their ids must be stamped: pydantic-ai rejects a duplicate id, which is
    what keeps a single decision-maker per run."""
    names = ["EvidenceCompactionCapability", "CitationPolicyCapability"]
    entries: list[Any] = (
        list(names) if form == "bare" else [{name: {}} for name in names]
    )

    compaction, policy = _from_spec(
        {"capabilities": entries},
        [EvidenceCompactionCapability, CitationPolicyCapability],
    )

    assert compaction.id == COMPACTION_ID
    assert policy.id == POLICY_ID


def test_a_spec_cannot_register_two_citation_policies():
    with pytest.raises(UserError, match=POLICY_ID):
        _from_spec(
            {"capabilities": ["CitationPolicyCapability"] * 2},
            [CitationPolicyCapability],
        )


def test_the_generated_spec_schema_describes_every_capability():
    schema = AgentSpec.model_json_schema_with_capabilities(ALL_CAPABILITIES)
    serialized = json.dumps(schema)

    for capability in ALL_CAPABILITIES:
        assert capability.__name__ in serialized

    params = schema["$defs"]["spec_params_RAGCapability"]["properties"]
    assert set(params) == {
        "config",
        "db_path",
        "defer_loading",
        "request_limit",
        "vision",
    }
    assert params["config"] == {
        "anyOf": [{"$ref": "#/$defs/AppConfig"}, {"type": "null"}]
    }
    assert "AppConfig" in schema["$defs"]
    assert {"format": "path", "type": "string"} in params["db_path"]["anyOf"]

    # Internal constructor wiring must not become a spec surface.
    for internal in (
        "state_type",
        "instruction_text",
        "tool_names",
        "state_namespace",
        "borrowed_rag",
        "rag_lock",
    ):
        assert internal not in serialized

    # A zero-argument from_spec leaves no params object at all, so the per-run
    # build caches cannot be set from a spec.
    assert "spec_params_EvidenceCompactionCapability" not in schema["$defs"]

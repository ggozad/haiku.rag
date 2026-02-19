from dataclasses import dataclass
from unittest.mock import MagicMock

from haiku.rag.tools.context import RAGDeps


def test_ragdeps_protocol_satisfied_by_dataclass():
    """A dataclass with a client attribute satisfies RAGDeps."""

    @dataclass
    class MyDeps:
        client: MagicMock

    deps = MyDeps(client=MagicMock())
    assert isinstance(deps, RAGDeps)


def test_ragdeps_protocol_not_satisfied_without_client():
    """An object without client does not satisfy RAGDeps."""

    @dataclass
    class NoDeps:
        other: str

    deps = NoDeps(other="x")
    assert not isinstance(deps, RAGDeps)

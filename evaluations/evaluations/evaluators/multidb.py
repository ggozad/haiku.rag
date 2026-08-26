import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    """Every number in the text, comma separators removed.

    Answers are scored by extraction rather than string matching: "1,240 metres",
    "1240 m" and a sentence around either are all legitimate.
    """
    found: set[float] = set()
    for match in _NUMBER.finditer(text or ""):
        try:
            found.add(float(match.group().replace(",", "")))
        except ValueError:  # pragma: no cover - the pattern only matches numbers
            continue
    return found


def _as_floats(
    single: float | int | None, many: Sequence[float | int] | None
) -> set[float]:
    values: list[float | int] = [] if single is None else [single]
    values.extend(many or ())
    return {float(v) for v in values}


def cited_sources(ctx: EvaluatorContext) -> list[str]:
    return [s for s in (ctx.attributes.get("cited_sources") or []) if s]


@dataclass
class NumericAnswer(Evaluator):
    """The gold number is present and the distractor's is absent.

    Presence alone is the wrong assertion: "Station Kestrel sits at either 1240 m
    or 2310 m" contains the gold value while demonstrating exactly the confusion
    the near-name pair exists to provoke. Reads `expected_value` and optional
    `distractor_value` from case metadata.
    """

    def get_default_evaluation_name(self) -> str:
        return "answer_correct"

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float | bool]:
        meta = ctx.metadata or {}
        expected = _as_floats(meta.get("expected_value"), meta.get("expected_values"))
        if not expected:
            return {}
        found = numbers_in(str(ctx.output))
        forbidden = _as_floats(
            meta.get("distractor_value"), meta.get("distractor_values")
        )
        correct = expected <= found and not (forbidden & found)
        return {"answer_correct": 1.0 if correct else 0.0}


@dataclass
class AttributionGate(Evaluator):
    """The databases cited are exactly the databases that hold the answer.

    A hard gate: `cited_map` scores URIs and would pass an answer attributed to
    the wrong database, which is the failure mode this dataset exists for.
    """

    def get_default_evaluation_name(self) -> str:
        return "attribution_correct"

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float | bool]:
        meta = ctx.metadata or {}
        expected = meta.get("expected_sources")
        if expected is None:
            return {}
        return {"attribution_correct": set(cited_sources(ctx)) == set(expected)}


@dataclass
class ScopeGate(Evaluator):
    """Nothing outside a scoped question's `sources` is cited.

    A hard gate, and separate from attribution: a case can cite the right
    database and still have reached outside its scope to get there.
    """

    def get_default_evaluation_name(self) -> str:
        return "scope_honoured"

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float | bool]:
        meta = ctx.metadata or {}
        scope = meta.get("scope")
        if scope is None:
            return {}
        return {"scope_honoured": set(cited_sources(ctx)) <= set(scope)}


@dataclass
class TextAnswer(Evaluator):
    """Required strings appear, in order, and forbidden strings do not.

    Order matters for the headings case, where the outline is only right if the
    sections come back in document order.
    """

    def get_default_evaluation_name(self) -> str:
        return "answer_correct"

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float | bool]:
        meta = ctx.metadata or {}
        required = meta.get("expected_ordered")
        if required is None:
            return {}
        haystack = str(ctx.output).lower()
        cursor = 0
        for needle in required:
            found = haystack.find(str(needle).lower(), cursor)
            if found < 0:
                return {"answer_correct": 0.0}
            cursor = found + len(str(needle))
        for forbidden in meta.get("forbidden_text") or []:
            if str(forbidden).lower() in haystack:
                return {"answer_correct": 0.0}
        return {"answer_correct": 1.0}


@dataclass
class MultiDBScores(Evaluator):
    """Every deterministic score for a case, in one evaluator.

    `DatasetSpec.qa_evaluator` takes a single evaluator and replaces the judge
    when set, so the scorers are composed here rather than listed. Each abstains
    on cases whose metadata does not ask for it.
    """

    def get_default_evaluation_name(self) -> str:
        return "answer_correct"

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float | bool]:
        meta = ctx.metadata or {}
        numeric = bool(meta.get("expected_value") or meta.get("expected_values"))
        textual = meta.get("expected_ordered") is not None
        if numeric and textual:
            raise ValueError(
                f"case {meta.get('question_id')!r} asks for both a numeric and an "
                "ordered-text answer; they share the answer_correct key"
            )
        scores: dict[str, float | bool] = {}
        for evaluator in (
            NumericAnswer(),
            TextAnswer(),
            AttributionGate(),
            ScopeGate(),
        ):
            scores.update(evaluator.evaluate(ctx))
        return scores

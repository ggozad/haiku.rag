from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria, TypeSafeError

StateField = Literal["input", "expected_output", "output"]
Judgement = dict[str, EvaluationReason | float | str]


@dataclass(repr=False)
class SystemOneJudge(Evaluator[object, object, object]):
    """Judges an output with one yes/no question to a `/v1/systemone` endpoint.

    The material goes in the state and the rubric in the question. The judge
    decides alone at p >= `pass_at` or p < `fail_below`; `fallback` decides
    the band between and any case the endpoint fails. Without a fallback the
    band splits at 0.5 and endpoint errors raise.
    """

    client: AsyncTypeSafeClient
    instructions: Any
    criteria: NoulCriteria | None = None
    model: str | None = None
    include_input: bool = False
    include_expected_output: bool = False
    state_names: Mapping[StateField, str] = field(default_factory=dict)
    pass_at: float = 0.8
    fail_below: float = 0.2
    fallback: Evaluator | None = None
    evaluation_name: str = "verdict"

    def _state(self, ctx: EvaluatorContext) -> dict[str, Any]:
        values: dict[StateField, Any] = {}
        if self.include_input:
            values["input"] = ctx.inputs
        if self.include_expected_output:
            values["expected_output"] = ctx.expected_output
        values["output"] = ctx.output
        return {self.state_names.get(key, key): value for key, value in values.items()}

    def _question(self) -> Noul:
        if self.criteria is None:
            return Noul(instructions=self.instructions)
        return Noul(instructions=self.instructions, criteria=self.criteria)

    async def _fallback_verdict(self, ctx: EvaluatorContext) -> EvaluationReason:
        assert self.fallback is not None
        output = await self.fallback.evaluate_async(ctx)
        if isinstance(output, Mapping):
            output = output[self.evaluation_name]
        if isinstance(output, EvaluationReason):
            return output
        return EvaluationReason(value=output)

    async def evaluate(self, ctx: EvaluatorContext) -> Judgement:
        name = self.evaluation_name
        try:
            response = await self.client.system_one(
                self._state(ctx), {name: self._question()}, model=self.model
            )
        except TypeSafeError:
            if self.fallback is None:
                raise
            return {
                name: await self._fallback_verdict(ctx),
                f"{name}_decided_by": "fallback_on_error",
            }

        p = response.nouls[name].noul
        if self.fallback is not None and self.fail_below <= p < self.pass_at:
            verdict = await self._fallback_verdict(ctx)
            decided_by = "fallback"
        else:
            threshold = self.pass_at if self.fallback is not None else 0.5
            verdict = EvaluationReason(
                value=p >= threshold, reason=f"system_one p={p:.3f}"
            )
            decided_by = "system_one"
        return {
            name: verdict,
            f"{name}_probability": p,
            f"{name}_decided_by": decided_by,
        }

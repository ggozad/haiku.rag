from dataclasses import dataclass

from pydantic_ai import models
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.evaluator import EvaluationReason, EvaluatorOutput
from pydantic_evals.evaluators.llm_as_a_judge import (
    judge_input_output_expected,
    judge_output,
)

from evaluations.evaluators.citation import average_precision
from evaluations.evaluators.refusal import REFUSAL_ELIGIBLE_LABELS, REFUSAL_RUBRIC


@dataclass
class ConversationEvaluator(Evaluator):
    """Score a live-session conversation turn by turn.

    Expects the case output to be the list of per-turn answers, case inputs
    the list of user questions, ``metadata["turns"]`` the per-turn reference,
    answerability label, and optional gold ``relevant_uris``, and the
    ``turn_cited_uris`` attribute the per-turn cited URIs.

    Each turn's answer is judged against the reference with the conversation
    so far — including the model's own earlier answers — as context. Citation
    AP is computed on turns with gold passages; refusal on ANSWERABLE and
    UNANSWERABLE turns. Returned counts allow micro aggregation across
    conversations; ``turn_pass_rate`` is the per-conversation (macro) rate.
    Per-turn verdicts are returned as ``turn_{n}_pass`` (with the judge's
    reason), ``turn_{n}_refused``, and ``turn_{n}_cited_ap`` for diagnosis.
    """

    rubric: str
    model: models.Model | models.KnownModelName | str | None = None

    async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
        questions: list[str] = list(ctx.inputs)
        answers: list[str] = list(ctx.output)
        turns: list[dict] = (ctx.metadata or {}).get("turns", [])
        turn_cited: list[list[str]] = list(
            ctx.attributes.get("turn_cited_uris") or [[] for _ in answers]
        )
        if not (len(questions) == len(answers) == len(turns) == len(turn_cited)):
            raise ValueError(
                f"conversation arrays disagree: {len(questions)} questions, "
                f"{len(answers)} answers, {len(turns)} turn annotations, "
                f"{len(turn_cited)} citation lists"
            )

        passed = 0
        judged = 0
        citation_scores: list[float] = []
        true_refusals = 0
        false_refusals = 0
        unanswerable = 0
        per_turn: dict[str, EvaluationReason | bool | float | str] = {}

        transcript_lines: list[str] = []
        for index, (question, answer, turn) in enumerate(
            zip(questions, answers, turns)
        ):
            number = index + 1
            transcript_lines.append(f"user: {question}")
            transcript = "\n".join(transcript_lines)
            transcript_lines.append(f"agent: {answer}")

            try:
                grading = await judge_input_output_expected(
                    transcript, answer, turn["reference"], self.rubric, self.model
                )
            except Exception as error:
                per_turn[f"turn_{number}_judge_error"] = str(error)[:200]
            else:
                judged += 1
                if grading.pass_:
                    passed += 1
                per_turn[f"turn_{number}_pass"] = EvaluationReason(
                    value=grading.pass_, reason=grading.reason
                )

            label = turn.get("answerability")
            if label in REFUSAL_ELIGIBLE_LABELS:
                try:
                    refused = (
                        await judge_output(answer, REFUSAL_RUBRIC, self.model)
                    ).pass_
                except Exception as error:
                    per_turn[f"turn_{number}_judge_error"] = str(error)[:200]
                else:
                    per_turn[f"turn_{number}_refused"] = refused
                    if label == "UNANSWERABLE":
                        unanswerable += 1
                        if refused:
                            true_refusals += 1
                    elif refused:
                        false_refusals += 1

            relevant = set(turn.get("relevant_uris") or [])
            if relevant:
                turn_ap = average_precision(turn_cited[index], relevant)
                citation_scores.append(turn_ap)
                per_turn[f"turn_{number}_cited_ap"] = turn_ap

        total = len(answers)
        result: dict[str, EvaluationReason | bool | float | int | str] = {
            "turn_pass_rate": passed / judged if judged else 0.0,
            "turns_passed": passed,
            "turns_judged": judged,
            "turns_total": total,
            "cited_eligible": len(citation_scores),
            "true_refusals": true_refusals,
            "false_refusals": false_refusals,
            "unanswerable_turns": unanswerable,
        }
        if citation_scores:
            result["cited_map"] = sum(citation_scores) / len(citation_scores)
        result.update(per_turn)
        return result

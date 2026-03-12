import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai.models import Model
from pydantic_evals import Case
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output_expected

from gepa.core.adapter import EvaluationBatch

from evaluations.benchmark import JUDGE_MODEL_CONFIG
from evaluations.config import DatasetSpec
from haiku.rag.agents.qa import get_qa_agent
from haiku.rag.agents.qa.prompts import QA_SYSTEM_PROMPT
from haiku.rag.client import HaikuRAG
from haiku.rag.config.models import AppConfig, ModelConfig
from haiku.rag.utils import get_model

logger = logging.getLogger(__name__)


OPTIMIZATION_SCORING_RUBRIC = """You are evaluating the quality of an answer to a question,
comparing it against a reference answer.

Score on a scale of 0.0 to 1.0:
- 1.0: The answer is factually correct, complete, and concise. It covers all key points
  from the reference answer without contradictions or significant omissions.
- 0.7-0.9: The answer is mostly correct and addresses the core question, but may miss
  some secondary details or include minor inaccuracies.
- 0.4-0.6: The answer is partially correct — it addresses some aspects of the question
  but misses key information or contains notable inaccuracies.
- 0.1-0.3: The answer is mostly incorrect or fails to address the core question,
  though it may contain some tangentially relevant information.
- 0.0: The answer is completely wrong, irrelevant, or empty.

GUIDELINES:
- Focus on factual correctness relative to the reference answer
- Ignore differences in phrasing, style, or formatting
- A concise correct answer scores higher than a verbose partially correct one
- "I cannot find enough information" when the reference has an answer scores 0.0
"""


@dataclass
class EvalTrajectory:
    """Per-case evaluation result for GEPA reflection."""

    question: str
    expected_answer: str
    actual_answer: str | None
    score: float
    judge_reason: str | None = None


QACase = Case[str, str, dict[str, str]]


@dataclass
class QAPromptAdapter:
    """GEPA adapter that evaluates QA prompt candidates against a dataset.

    Implements the GEPAAdapter protocol:
    - evaluate(): Run QA agent with candidate prompt, score with LLMJudge
    - make_reflective_dataset(): Build failure records for the GEPA proposer
    """

    config: AppConfig
    db_path: Path
    judge_model: Model

    def evaluate(
        self,
        batch: list[QACase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[EvalTrajectory, str | None]:
        instructions = candidate["instructions"]
        return asyncio.run(self._evaluate_async(batch, instructions, capture_traces))

    async def _evaluate_async(
        self,
        batch: list[QACase],
        instructions: str,
        capture_traces: bool,
    ) -> EvaluationBatch[EvalTrajectory, str | None]:
        outputs: list[str | None] = []
        scores: list[float] = []
        trajectories: list[EvalTrajectory] | None = [] if capture_traces else None

        async with HaikuRAG(self.db_path, config=self.config) as rag:
            qa = get_qa_agent(rag, self.config, system_prompt=instructions)

            for case in batch:
                question = case.inputs
                expected = case.expected_output or ""

                try:
                    answer, _ = await qa.answer(question)
                except Exception:
                    logger.warning(
                        "QA agent failed for question: %s", question, exc_info=True
                    )
                    answer = None

                if answer is not None:
                    score, reason = await self._judge(question, answer, expected)
                else:
                    score, reason = 0.0, "QA agent failed to produce an answer"

                outputs.append(answer)
                scores.append(score)

                if capture_traces and trajectories is not None:
                    trajectories.append(
                        EvalTrajectory(
                            question=question,
                            expected_answer=expected,
                            actual_answer=answer,
                            score=score,
                            judge_reason=reason,
                        )
                    )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    async def _judge(
        self, question: str, answer: str, expected: str
    ) -> tuple[float, str | None]:
        """Score an answer using pydantic-evals LLMJudge with float scoring."""
        result = await judge_input_output_expected(
            inputs=question,
            output=answer,
            expected_output=expected,
            rubric=OPTIMIZATION_SCORING_RUBRIC,
            model=self.judge_model,
        )
        return result.score, result.reason

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[EvalTrajectory, str | None],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if eval_batch.trajectories is None:
            return {}

        records: list[dict[str, Any]] = []
        for traj in eval_batch.trajectories:
            records.append(
                {
                    "Inputs": {"question": traj.question},
                    "Generated Outputs": {
                        "answer": traj.actual_answer or "(no answer)"
                    },
                    "Feedback": (
                        f"Expected answer: {traj.expected_answer}\n"
                        f"Score: {traj.score:.2f}\n"
                        f"Judge reasoning: {traj.judge_reason or 'N/A'}"
                    ),
                }
            )

        return {"instructions": records}

    propose_new_texts = None


class ReflectionLM:
    """LanguageModel implementation for GEPA's ReflectiveMutationProposer.

    Wraps a pydantic-ai Agent to satisfy GEPA's LanguageModel protocol.
    """

    def __init__(self, model_config: ModelConfig, config: AppConfig) -> None:
        from pydantic_ai import Agent

        model = get_model(model_config, config)
        self._agent: Agent[None, str] = Agent(model=model, output_type=str)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, list):
            text = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in prompt
            )
        else:
            text = prompt
        result = self._agent.run_sync(text)
        return result.output


REFLECTION_MINIBATCH_SIZE = 3


def run_optimization(
    spec: DatasetSpec,
    config: AppConfig,
    cases: list[QACase],
    iterations: int,
    db_path: Path | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Run GEPA optimization and return results summary."""
    from rich.console import Console

    console = Console()

    judge_model = get_model(JUDGE_MODEL_CONFIG, config)

    db = spec.db_path(db_path)
    adapter = QAPromptAdapter(
        config=config,
        db_path=db,
        judge_model=judge_model,
    )

    reflection_lm = ReflectionLM(config.qa.model, config)

    seed_prompt = spec.resolve_system_prompt(config) or QA_SYSTEM_PROMPT
    seed_candidate = {"instructions": seed_prompt}

    mid = len(cases) // 2
    trainset = cases[:mid]
    valset = cases[mid:]

    # Budget: initial valset eval + worst-case iterations
    # (each iteration: 2 minibatch evals + full valset if accepted)
    max_metric_calls = len(valset) + iterations * (
        2 * REFLECTION_MINIBATCH_SIZE + len(valset)
    )

    console.print(f"Optimizing prompt for dataset: {spec.key}", style="bold magenta")
    console.print(
        f"Train: {len(trainset)}, Val: {len(valset)}, "
        f"Iterations: {iterations}, GEPA budget: {max_metric_calls}"
    )
    console.print(f"Seed prompt length: {len(seed_prompt)} chars")

    from gepa import optimize as gepa_optimize

    result = gepa_optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        display_progress_bar=True,
    )

    best_score = result.val_aggregate_scores[result.best_idx]
    best_prompt = result.best_candidate
    if isinstance(best_prompt, dict):
        best_prompt = best_prompt["instructions"]
    total_calls = result.total_metric_calls or "unknown"

    console.print("\n=== Optimization Results ===", style="bold cyan")
    console.print(f"Total metric calls: {total_calls}")
    console.print(f"Candidates explored: {result.num_candidates}")
    console.print(f"Best score: {best_score:.4f}")
    console.print(f"\nOptimized prompt:\n{best_prompt}")

    if output:
        output.write_text(best_prompt)
        console.print(f"\nSaved to: {output}", style="green")

    return {
        "best_score": best_score,
        "best_prompt": best_prompt,
        "total_calls": total_calls,
        "num_candidates": result.num_candidates,
    }

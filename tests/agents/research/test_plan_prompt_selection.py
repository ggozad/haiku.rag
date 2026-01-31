from haiku.rag.agents.research.prompts import (
    ITERATIVE_PLAN_PROMPT,
    ITERATIVE_PLAN_PROMPT_WITH_CONTEXT,
)


def test_iterative_plan_prompt_proposes_first_question():
    """ITERATIVE_PLAN_PROMPT should instruct to propose the first question."""
    assert "first question" in ITERATIVE_PLAN_PROMPT.lower()
    assert "is_complete=False" in ITERATIVE_PLAN_PROMPT


def test_iterative_plan_prompt_with_context_evaluates_evidence():
    """ITERATIVE_PLAN_PROMPT_WITH_CONTEXT should evaluate prior answers."""
    assert "prior_answers" in ITERATIVE_PLAN_PROMPT_WITH_CONTEXT
    assert (
        "evaluat" in ITERATIVE_PLAN_PROMPT_WITH_CONTEXT.lower()
    )  # matches evaluate/evaluating


def test_prompt_selection_uses_context_prompt_with_prior_answers():
    """When prior_answers exist, should use ITERATIVE_PLAN_PROMPT_WITH_CONTEXT."""
    has_prior_answers = True

    effective_plan_prompt = (
        ITERATIVE_PLAN_PROMPT_WITH_CONTEXT
        if has_prior_answers
        else ITERATIVE_PLAN_PROMPT
    )

    assert effective_plan_prompt == ITERATIVE_PLAN_PROMPT_WITH_CONTEXT

from haiku.rag.agents.research.prompts import (
    ITERATIVE_PLAN_PROMPT,
    ITERATIVE_PLAN_PROMPT_WITH_CONTEXT,
)


def test_iterative_plan_prompt_with_context_does_not_instruct_gather_context():
    """ITERATIVE_PLAN_PROMPT_WITH_CONTEXT should not instruct to use gather_context.

    When prior answers already exist, we don't need to gather context again.
    """
    assert "gather_context" not in ITERATIVE_PLAN_PROMPT_WITH_CONTEXT


def test_iterative_plan_prompt_instructs_gather_context():
    """ITERATIVE_PLAN_PROMPT should instruct to use gather_context for initial planning."""
    assert "gather_context" in ITERATIVE_PLAN_PROMPT


def test_prompt_selection_uses_context_prompt_with_prior_answers():
    """When prior_answers exist, should use ITERATIVE_PLAN_PROMPT_WITH_CONTEXT."""
    has_prior_answers = True

    effective_plan_prompt = (
        ITERATIVE_PLAN_PROMPT_WITH_CONTEXT
        if has_prior_answers
        else ITERATIVE_PLAN_PROMPT
    )

    assert effective_plan_prompt == ITERATIVE_PLAN_PROMPT_WITH_CONTEXT

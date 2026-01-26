from haiku.rag.agents.research.prompts import PLAN_PROMPT, PLAN_PROMPT_WITH_CONTEXT


def test_plan_prompt_with_context_does_not_instruct_gather_context():
    """PLAN_PROMPT_WITH_CONTEXT should not instruct to use gather_context.

    When session context already exists, we don't need to gather context again.
    """
    assert "gather_context" not in PLAN_PROMPT_WITH_CONTEXT


def test_plan_prompt_instructs_gather_context():
    """PLAN_PROMPT should instruct to use gather_context for initial planning."""
    assert "gather_context" in PLAN_PROMPT


def test_prompt_selection_uses_context_prompt_with_session_context():
    """When session_context exists, should use PLAN_PROMPT_WITH_CONTEXT."""
    has_prior_answers = False
    has_session_context = True

    effective_plan_prompt = (
        PLAN_PROMPT_WITH_CONTEXT
        if has_prior_answers or has_session_context
        else PLAN_PROMPT
    )

    assert effective_plan_prompt == PLAN_PROMPT_WITH_CONTEXT

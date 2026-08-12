from review_agent.intent_clarification import (
    ConsoleIntentClarifier,
    NonInteractiveIntentClarifier,
)
from review_agent.models import (
    ClarificationQuestion,
    IntentDecisionAction,
    IntentField,
)


def question(*, proposed: bool = True) -> ClarificationQuestion:
    return ClarificationQuestion(
        field=IntentField.GOAL,
        question="Is this the intended goal?",
        rationale="The answer changes correctness conclusions.",
        proposed_values=["Preserve addition"] if proposed else [],
        claim_ids=["claim_test"] if proposed else [],
    )


def test_non_interactive_never_reads_input_and_skips():
    decision = NonInteractiveIntentClarifier().decide(question())

    assert decision.action is IntentDecisionAction.SKIPPED_NON_INTERACTIVE
    assert decision.continuation_basis == "non_interactive_policy"


def test_eval_no_user_token_continues_without_promoting_inferred_intent():
    decision = ConsoleIntentClarifier(
        input_fn=lambda _prompt: "continue-with-uncertainty:benchmark-no-user",
        output_fn=lambda _message: None,
    ).decide(question())

    assert decision is not None
    assert decision.action is IntentDecisionAction.SKIPPED_NON_INTERACTIVE
    assert decision.continuation_basis == "benchmark_no_user"


def test_console_confirm_uses_proposed_value():
    decision = ConsoleIntentClarifier(
        input_fn=lambda _prompt: "confirm",
        output_fn=lambda _message: None,
    ).decide(question())

    assert decision is not None
    assert decision.action is IntentDecisionAction.CONFIRMED
    assert decision.continuation_basis is None


def test_console_benchmark_auto_accept_uses_non_human_basis():
    decision = ConsoleIntentClarifier(
        input_fn=lambda _prompt: "confirm:benchmark-auto-accept",
        output_fn=lambda _message: None,
    ).decide(question())

    assert decision is not None
    assert decision.action is IntentDecisionAction.CONFIRMED
    assert decision.continuation_basis == "benchmark_auto_accept"


def test_console_correction_parses_semicolon_values():
    answers = iter(["correct", "Preserve addition; Keep integer return type"])
    decision = ConsoleIntentClarifier(
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _message: None,
    ).decide(question())

    assert decision is not None
    assert decision.action is IntentDecisionAction.CORRECTED
    assert decision.corrected_values == [
        "Preserve addition",
        "Keep integer return type",
    ]


def test_console_defer_returns_none_for_resumable_wait():
    decision = ConsoleIntentClarifier(
        input_fn=lambda _prompt: "defer",
        output_fn=lambda _message: None,
    ).decide(question())

    assert decision is None


def test_console_requires_correction_for_multiple_conflicting_values():
    conflicting = ClarificationQuestion(
        field=IntentField.GOAL,
        question="Which goal is intended?",
        rationale="Conflicting goals change the review conclusion.",
        proposed_values=["Preserve retries", "Remove retries"],
        claim_ids=["claim_preserve", "claim_remove"],
    )
    answers = iter(["confirm", "correct", "Preserve retries"])

    decision = ConsoleIntentClarifier(
        input_fn=lambda _prompt: next(answers),
        output_fn=lambda _message: None,
    ).decide(conflicting)

    assert decision is not None
    assert decision.action is IntentDecisionAction.CORRECTED
    assert decision.corrected_values == ["Preserve retries"]

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from review_agent.models import (
    BENCHMARK_AUTO_ACCEPT_BASIS,
    ClarificationQuestion,
    IntentDecision,
    IntentDecisionAction,
    IntentField,
)


class IntentClarifier(Protocol):
    def decide(self, question: ClarificationQuestion) -> IntentDecision | None:
        """Return a decision, or None to persist an awaiting-user state."""


@dataclass(frozen=True)
class NonInteractiveIntentClarifier:
    continuation_basis: str = "non_interactive_policy"

    def decide(self, question: ClarificationQuestion) -> IntentDecision:
        return IntentDecision(
            question_id=question.question_id,
            action=IntentDecisionAction.SKIPPED_NON_INTERACTIVE,
            continuation_basis=self.continuation_basis,
        )


class ConsoleIntentClarifier:
    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._output = output_fn

    def decide(self, question: ClarificationQuestion) -> IntentDecision | None:
        self._output(f"Intent clarification [{question.field.value}]")
        self._output(question.question)
        if question.proposed_values:
            self._output("Proposed value(s):")
            for value in question.proposed_values:
                self._output(f"  - {value}")
        self._output(f"Why it matters: {question.rationale}")

        options = "confirm / correct / reject / skip / defer"
        while True:
            try:
                raw = self._input(f"Choose {options}: ").strip().casefold()
            except (EOFError, KeyboardInterrupt, OSError):
                return None
            if raw in {"defer", "d"}:
                return None
            if raw in {"skip", "s"}:
                return IntentDecision(
                    question_id=question.question_id,
                    action=IntentDecisionAction.SKIPPED,
                    continuation_basis="user_skip",
                )
            if raw == "continue-with-uncertainty:benchmark-no-user":
                return IntentDecision(
                    question_id=question.question_id,
                    action=IntentDecisionAction.SKIPPED_NON_INTERACTIVE,
                    continuation_basis="benchmark_no_user",
                )
            if raw == "confirm:benchmark-auto-accept":
                if not question.proposed_values:
                    self._output("There is no proposed value to auto-accept.")
                    continue
                if (
                    question.field is IntentField.GOAL
                    and len(question.proposed_values) > 1
                ):
                    self._output(
                        "Conflicting goal candidates cannot be auto-accepted."
                    )
                    continue
                return IntentDecision(
                    question_id=question.question_id,
                    action=IntentDecisionAction.CONFIRMED,
                    continuation_basis=BENCHMARK_AUTO_ACCEPT_BASIS,
                )
            if raw in {"confirm", "c", "yes", "y"}:
                if not question.proposed_values:
                    self._output("There is no proposed value to confirm; choose correct or skip.")
                    continue
                if (
                    question.field is IntentField.GOAL
                    and len(question.proposed_values) > 1
                ):
                    self._output(
                        "Multiple conflicting values cannot be confirmed together; "
                        "choose correct or reject."
                    )
                    continue
                return IntentDecision(
                    question_id=question.question_id,
                    action=IntentDecisionAction.CONFIRMED,
                )
            if raw in {"reject", "r", "no", "n"}:
                if not question.claim_ids:
                    self._output("There is no inferred candidate to reject; choose correct or skip.")
                    continue
                return IntentDecision(
                    question_id=question.question_id,
                    action=IntentDecisionAction.REJECTED,
                )
            if raw in {"correct", "edit", "e"}:
                try:
                    answer = self._input(
                        "Enter corrected value(s), separated with semicolons: "
                    ).strip()
                except (EOFError, KeyboardInterrupt, OSError):
                    return None
                values = [item.strip() for item in answer.split(";") if item.strip()]
                if not values:
                    self._output("At least one corrected value is required.")
                    continue
                return IntentDecision(
                    question_id=question.question_id,
                    action=IntentDecisionAction.CORRECTED,
                    corrected_values=values,
                    user_response=answer,
                )
            self._output(f"Unknown choice. Use {options}.")

from __future__ import annotations

from dataclasses import replace

import pytest

from review_agent_eval.clarification import (
    BENCHMARK_AUTO_ACCEPT_POLICY_VERSION,
    ClarificationChannel,
    ClarificationProtocolError,
    ClarificationSession,
    MaterialClaimMatcher,
    MaterialClaimMatcherFactory,
    MaterialClaimMatchOutcome,
    IntentContinuationMode,
    UNANSWERED_CLARIFICATION_CONTINUE,
    canonical_material_claim_matcher_snapshot,
    unanswered_clarification_action,
)
from review_agent_eval.adapters.base import AgentRunConfig
from review_agent_eval.adapters.current_agent import current_agent_capabilities
from review_agent_eval.cases import REPOSITORY_MATERIALIZER_PROTOCOL, WireContractV2
from review_agent_eval.config import (
    AgentConfigSnapshot,
    ClarificationMatcherSnapshot,
    ResourceBudgets,
    derive_trial_id,
)
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_INPUT_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    MAX_CLARIFICATION_QUESTIONS,
    MAX_QUESTION_CHARS,
    ClarificationAction,
    ClarificationAnswer,
    ClarificationScript,
    IntentDimension,
    ReviewTargetKind,
    SchemaError,
    SubmissionClarificationExchange,
    stable_id,
)


def answer(
    answer_id: str,
    *,
    dimension: IntentDimension = IntentDimension.GOAL,
    material_claim: str = "The change must support dry-run mode",
    action: ClarificationAction = ClarificationAction.CONFIRM,
    response: str | None = "Yes",
    corrected_values: tuple[str, ...] = (),
) -> ClarificationAnswer:
    return ClarificationAnswer(
        answer_id=answer_id,
        dimension=dimension,
        material_claim=material_claim,
        action=action,
        response=response,
        corrected_values=corrected_values,
    )


def question(
    question_id: str,
    *,
    dimension: IntentDimension = IntentDimension.GOAL,
    material_claim: str = "The change must support dry-run mode",
    question_text: str = "Should dry-run mode be part of the requested behavior?",
    proposed_values: tuple[str, ...] | list[str] = ("dry-run",),
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "dimension": dimension,
        "question": question_text,
        "material_claim": material_claim,
        "proposed_values": proposed_values,
    }


def session_with(
    *answers: ClarificationAnswer,
    max_rounds: int = 16,
    matcher: MaterialClaimMatcher | None = None,
    unanswered_action: str | None = None,
    continuation_mode: IntentContinuationMode = IntentContinuationMode.SCRIPTED,
) -> ClarificationSession:
    snapshot = (
        canonical_material_claim_matcher_snapshot()
        if matcher is None
        else semantic_matcher_snapshot()
    )
    factory = None
    if matcher is not None:
        matcher.binding_digest = snapshot.digest()  # type: ignore[attr-defined]
        factory = StaticMatcherFactory(matcher)
    return ClarificationSession(
        ClarificationScript(max_rounds=max_rounds, answers=answers),
        run_binding=run_binding(
            snapshot,
            unanswered_action=unanswered_action,
        ),
        matcher_factory=factory,
        continuation_mode=continuation_mode,
    )


def semantic_matcher_snapshot() -> ClarificationMatcherSnapshot:
    return replace(
        canonical_material_claim_matcher_snapshot(),
        matcher_id="test-semantic-material-claim",
        matcher_version="1.0.0-test",
        implementation_digest="1" * 64,
        model_artifact_digest="2" * 64,
        rubric_digest="3" * 64,
        normalization_version="test-semantic-normalization-v1",
        threshold=0.75,
        parameters={"fixture": "semantic-equivalence-table-v1"},
    )


def run_binding(
    snapshot: ClarificationMatcherSnapshot,
    *,
    matcher_digest: str | None = None,
    unanswered_action: str | None = None,
) -> AgentRunConfig:
    task_id = "task-clarification-session"
    parameters = {}
    if unanswered_action is not None:
        parameters["clarification"] = {
            "unanswered_action": unanswered_action,
            "intent_continuation_policy_version": (
                BENCHMARK_AUTO_ACCEPT_POLICY_VERSION
            ),
        }
    agent = AgentConfigSnapshot(
        agent_id="agent-clarification-test",
        agent_name="Clarification test agent",
        agent_version="1",
        commit="a" * 40,
        model="none",
        provider="test",
        parameters=parameters,
        prompt_config_digest="b" * 64,
    )
    run_id = stable_id("run", "clarification-session-tests", snapshot.digest())
    capabilities = current_agent_capabilities()
    return AgentRunConfig._from_verified_binding(
        run_id=run_id,
        task_id=task_id,
        eval_input_digest="c" * 64,
        wire_contract=WireContractV2(
            case_schema_version=EVAL_CASE_SCHEMA_VERSION,
            input_schema_version=EVAL_INPUT_SCHEMA_VERSION,
            submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
            review_target_kind=ReviewTargetKind.REPOSITORY,
            materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
        ),
        adapter_capabilities=capabilities,
        adapter_capabilities_digest=capabilities.digest(),
        clarification_matcher=snapshot,
        clarification_matcher_config_digest=(
            snapshot.digest() if matcher_digest is None else matcher_digest
        ),
        trial_index=1,
        trial_id=derive_trial_id(run_id, task_id, 1),
        agent=agent,
        budgets=ResourceBudgets(
            agent_timeout_seconds=10,
            evaluator_timeout_seconds=10,
            max_agent_output_bytes=64 * 1024,
            max_trace_bytes=64 * 1024,
            max_execution_artifact_file_bytes=64 * 1024,
            max_execution_artifact_total_bytes=256 * 1024,
            max_parallel_trials=1,
        ),
    )


class StaticMatcherFactory:
    def __init__(self, matcher: MaterialClaimMatcher) -> None:
        self.matcher = matcher

    def build(
        self,
        snapshot: ClarificationMatcherSnapshot,
    ) -> MaterialClaimMatcher:
        del snapshot
        return self.matcher


def test_intent_continuation_policy_version_is_required_when_policy_is_present() -> None:
    with pytest.raises(ClarificationProtocolError, match="exactly"):
        unanswered_clarification_action(
            {"clarification": {"unanswered_action": "defer"}}
        )
    with pytest.raises(ClarificationProtocolError, match="version"):
        unanswered_clarification_action(
            {
                "clarification": {
                    "unanswered_action": "defer",
                    "intent_continuation_policy_version": "forged-v1",
                }
            }
        )


class FakeSemanticMatcher:
    def __init__(
        self,
        equivalent_pairs: set[tuple[IntentDimension, str, str]],
    ) -> None:
        self.equivalent_pairs = equivalent_pairs
        self.calls: list[tuple[IntentDimension, str, str]] = []
        self.binding_digest = "e" * 64

    def equivalent(
        self,
        dimension: IntentDimension,
        actual_claim: str,
        scripted_claim: str,
    ) -> bool:
        candidate = (dimension, actual_claim, scripted_claim)
        self.calls.append(candidate)
        return candidate in self.equivalent_pairs


def assert_unanswered(exchange: SubmissionClarificationExchange) -> None:
    assert exchange.matched_answer_id is None
    assert exchange.action is None
    assert exchange.response is None
    assert exchange.resolved_values == ()


def test_ask_uses_canonical_types_and_limits() -> None:
    session = session_with(answer("answer-1"))
    exchange = session.channel.ask(
        **question("question-1", proposed_values=["dry-run"])
    )

    assert exchange.dimension is IntentDimension.GOAL
    assert exchange.resolved_values == ("dry-run",)

    with pytest.raises(SchemaError, match="IntentDimension"):
        session.channel.ask(
            **question(
                "question-2",
                dimension="goal",  # type: ignore[arg-type]
            )
        )
    with pytest.raises(SchemaError, match="opaque identifier"):
        session.channel.ask(**question("question with spaces"))
    with pytest.raises(SchemaError, match="character limit"):
        session.channel.ask(
            **question(
                "question-3",
                question_text="q" * (MAX_QUESTION_CHARS + 1),
            )
        )
    assert session.transcript == (exchange,)


def test_channel_capability_exposes_only_question_operations_not_truth_or_transcript() -> None:
    session = session_with(answer("answer-1"))
    channel = session.channel

    protocol_methods = {
        name
        for name, member in ClarificationChannel.__dict__.items()
        if not name.startswith("_") and callable(member)
    }
    assert protocol_methods == {"ask", "skip_unresolved"}
    assert isinstance(channel, ClarificationChannel)
    assert "__dir__" not in type(channel).__dict__
    assert [name for name in dir(channel) if not name.startswith("_")] == [
        "ask",
        "skip_unresolved",
    ]
    for forbidden in (
        "script",
        "answers",
        "remaining_answers",
        "max_rounds",
        "policy",
        "truth",
        "transcript",
        "match_receipts",
        "consumed_ids",
        "consumed_answer_ids",
    ):
        assert not hasattr(channel, forbidden)

    assert session.transcript == ()
    assert session.match_receipts == ()
    assert session.consumed_answer_ids == frozenset()
    with pytest.raises(AttributeError):
        session.transcript = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        session.consumed_answer_ids = frozenset()  # type: ignore[misc]


def test_matching_is_dimension_aware_and_independent_of_question_order() -> None:
    goal = answer(
        "answer-goal",
        dimension=IntentDimension.GOAL,
        material_claim="Support dry-run mode",
    )
    scope = answer(
        "answer-scope",
        dimension=IntentDimension.SCOPE,
        material_claim="Only change the command package",
        action=ClarificationAction.CORRECT,
        response="Include the config package too",
        corrected_values=("command", "config"),
    )
    session = session_with(goal, scope)

    scope_exchange = session.channel.ask(
        **question(
            "question-scope",
            dimension=IntentDimension.SCOPE,
            material_claim="Only change the command package",
        )
    )
    goal_exchange = session.channel.ask(
        **question(
            "question-goal",
            dimension=IntentDimension.GOAL,
            material_claim="Support dry-run mode",
        )
    )

    assert scope_exchange.turn_index == 1
    assert scope_exchange.matched_answer_id == "answer-scope"
    assert goal_exchange.turn_index == 2
    assert goal_exchange.matched_answer_id == "answer-goal"
    assert session.transcript == (scope_exchange, goal_exchange)
    assert session.consumed_answer_ids == frozenset(
        {"answer-goal", "answer-scope"}
    )


def test_run_bound_canonical_matcher_normalizes_whitespace_and_casefold() -> None:
    scripted_claim = "The Change\tMUST support DRY-RUN mode"
    asked_claim = "  the change must   SUPPORT dry-run MODE\n"
    session = session_with(
        answer("answer-normalized", material_claim=scripted_claim)
    )

    exchange = session.channel.ask(
        **question("question-normalized", material_claim=asked_claim)
    )

    assert exchange.matched_answer_id == "answer-normalized"
    assert exchange.material_claim == asked_claim
    assert session.consumed_answer_ids == frozenset({"answer-normalized"})


def test_injected_semantic_matcher_can_match_a_paraphrased_material_claim() -> None:
    scripted_claim = "Support dry-run mode"
    paraphrase = "Let users preview changes without applying them"
    matcher = FakeSemanticMatcher(
        {(IntentDimension.GOAL, paraphrase, scripted_claim)}
    )
    matcher.binding_digest = semantic_matcher_snapshot().digest()
    assert isinstance(matcher, MaterialClaimMatcher)
    assert isinstance(StaticMatcherFactory(matcher), MaterialClaimMatcherFactory)
    session = session_with(
        answer("answer-semantic", material_claim=scripted_claim),
        matcher=matcher,
    )

    exchange = session.channel.ask(
        **question("question-semantic", material_claim=paraphrase)
    )

    assert exchange.matched_answer_id == "answer-semantic"
    receipt = session.match_receipts[0]
    assert receipt.outcome is MaterialClaimMatchOutcome.MATCHED
    assert receipt.matched_answer_id == "answer-semantic"
    assert receipt.matcher_digest == semantic_matcher_snapshot().digest()
    assert tuple(item.answer_id for item in receipt.candidates) == (
        "answer-semantic",
    )
    assert matcher.calls == [
        (IntentDimension.GOAL, paraphrase, scripted_claim)
    ]


def test_builtin_factory_never_treats_unknown_semantic_snapshot_as_exact() -> None:
    snapshot = semantic_matcher_snapshot()

    with pytest.raises(
        ClarificationProtocolError,
        match="unsupported by this Harness",
    ):
        ClarificationSession(
            ClarificationScript(max_rounds=1, answers=(answer("answer-1"),)),
            run_binding=run_binding(snapshot),
        )


def test_agent_run_config_cannot_bind_a_different_matcher_digest() -> None:
    snapshot = semantic_matcher_snapshot()

    with pytest.raises(
        SchemaError,
        match="matcher digest does not match its snapshot",
    ):
        run_binding(snapshot, matcher_digest="f" * 64)


def test_factory_matcher_must_match_the_run_bound_identity() -> None:
    matcher = FakeSemanticMatcher(set())
    snapshot = semantic_matcher_snapshot()

    with pytest.raises(
        ClarificationProtocolError,
        match="does not match the run binding",
    ):
        ClarificationSession(
            ClarificationScript(max_rounds=1, answers=(answer("answer-1"),)),
            run_binding=run_binding(snapshot),
            matcher_factory=StaticMatcherFactory(matcher),
        )


def test_multiple_semantic_candidates_fail_closed_without_consuming_answers() -> None:
    first_claim = "Support dry-run mode"
    second_claim = "Preview changes before applying them"
    ambiguous_paraphrase = "Allow safe change previews"
    unique_paraphrase = "Provide the dry-run switch"
    matcher = FakeSemanticMatcher(
        {
            (IntentDimension.GOAL, ambiguous_paraphrase, first_claim),
            (IntentDimension.GOAL, ambiguous_paraphrase, second_claim),
            (IntentDimension.GOAL, unique_paraphrase, first_claim),
        }
    )
    session = session_with(
        answer("answer-first", material_claim=first_claim),
        answer("answer-second", material_claim=second_claim),
        matcher=matcher,
    )

    ambiguous = session.channel.ask(
        **question("question-ambiguous", material_claim=ambiguous_paraphrase)
    )

    assert_unanswered(ambiguous)
    assert session.match_receipts[0].outcome is MaterialClaimMatchOutcome.AMBIGUOUS
    assert session.consumed_answer_ids == frozenset()

    unique = session.channel.ask(
        **question("question-unique", material_claim=unique_paraphrase)
    )
    assert unique.matched_answer_id == "answer-first"
    assert session.match_receipts[1].outcome is MaterialClaimMatchOutcome.MATCHED
    assert session.consumed_answer_ids == frozenset({"answer-first"})
    assert session.transcript == (ambiguous, unique)


@pytest.mark.parametrize("result", [None, 1, "yes"])
def test_matcher_must_return_a_real_boolean_without_mutating_transcript(
    result: object,
) -> None:
    class InvalidMatcher:
        binding_digest = "e" * 64

        def equivalent(self, *args: object) -> object:
            return result

    session = session_with(
        answer("answer-invalid-matcher"),
        matcher=InvalidMatcher(),  # type: ignore[arg-type]
    )
    with pytest.raises(ClarificationProtocolError, match="must return bool"):
        session.channel.ask(**question("question-invalid-matcher"))
    assert session.transcript == ()
    assert session.match_receipts == ()
    assert session.consumed_answer_ids == frozenset()


@pytest.mark.parametrize(
    ("action", "response", "corrected_values", "expected_resolved"),
    [
        (ClarificationAction.CONFIRM, "Yes", (), ("proposed-a", "proposed-b")),
        (
            ClarificationAction.CORRECT,
            "Use corrected values",
            ("corrected-a", "corrected-b"),
            ("corrected-a", "corrected-b"),
        ),
        (ClarificationAction.REJECT, "No", (), ()),
        (ClarificationAction.SKIP, None, (), ()),
        (ClarificationAction.DEFER, "Decide later", (), ()),
    ],
)
def test_all_canonical_actions_produce_the_required_resolved_values_and_round_trip(
    action: ClarificationAction,
    response: str | None,
    corrected_values: tuple[str, ...],
    expected_resolved: tuple[str, ...],
) -> None:
    scripted_answer = answer(
        "answer-1",
        action=action,
        response=response,
        corrected_values=corrected_values,
    )
    session = session_with(scripted_answer)

    exchange = session.channel.ask(
        **question(
            "question-1",
            proposed_values=("proposed-a", "proposed-b"),
        )
    )

    assert exchange.action is action
    assert exchange.response == response
    assert exchange.resolved_values == expected_resolved
    assert SubmissionClarificationExchange.from_dict(exchange.to_dict()) == exchange


def test_confirm_requires_proposed_values_and_does_not_consume_on_failed_match() -> None:
    session = session_with(answer("answer-confirm"))

    without_proposal = session.channel.ask(
        **question("question-empty", proposed_values=())
    )
    assert_unanswered(without_proposal)
    assert session.consumed_answer_ids == frozenset()

    with_proposal = session.channel.ask(**question("question-proposed"))
    assert with_proposal.matched_answer_id == "answer-confirm"
    assert with_proposal.resolved_values == ("dry-run",)
    assert session.consumed_answer_ids == frozenset({"answer-confirm"})


def test_wrong_dimension_and_material_claim_do_not_match_or_consume() -> None:
    session = session_with(
        answer(
            "answer-exact",
            dimension=IntentDimension.CONSTRAINT,
            material_claim="No network access",
            action=ClarificationAction.REJECT,
            response="Network access is allowed",
        )
    )

    wrong_dimension = session.channel.ask(
        **question(
            "question-wrong-dimension",
            dimension=IntentDimension.SCOPE,
            material_claim="No network access",
        )
    )
    wrong_claim = session.channel.ask(
        **question(
            "question-wrong-claim",
            dimension=IntentDimension.CONSTRAINT,
            material_claim="Network requests are forbidden",
        )
    )
    exact = session.channel.ask(
        **question(
            "question-exact",
            dimension=IntentDimension.CONSTRAINT,
            material_claim="No network access",
        )
    )

    assert_unanswered(wrong_dimension)
    assert_unanswered(wrong_claim)
    assert exact.matched_answer_id == "answer-exact"
    assert exact.turn_index == 3
    assert session.consumed_answer_ids == frozenset({"answer-exact"})


def test_unanswered_policy_continues_without_fabricating_a_case_answer() -> None:
    session = session_with(
        max_rounds=4,
        unanswered_action=UNANSWERED_CLARIFICATION_CONTINUE,
    )

    exchange = session.channel.ask(**question("question-policy-skip"))

    assert exchange.action is ClarificationAction.SKIP
    assert exchange.matched_answer_id is None
    assert exchange.response is None
    assert exchange.resolved_values == ()
    assert session.consumed_answer_ids == frozenset()
    assert session.match_receipts[0].outcome is MaterialClaimMatchOutcome.UNMATCHED
    assert session.match_receipts[0].matched_answer_id is None


def test_benchmark_auto_accept_confirms_without_fabricating_a_case_answer() -> None:
    session = session_with(
        max_rounds=4,
        unanswered_action=UNANSWERED_CLARIFICATION_CONTINUE,
        continuation_mode=IntentContinuationMode.BENCHMARK_AUTO_ACCEPT,
    )

    exchange = session.channel.ask(**question("question-auto-accept"))

    assert exchange.action is ClarificationAction.CONFIRM
    assert exchange.matched_answer_id is None
    assert exchange.response is None
    assert exchange.resolved_values == ("dry-run",)
    assert session.consumed_answer_ids == frozenset()
    assert (
        session.match_receipts[0].outcome
        is MaterialClaimMatchOutcome.BENCHMARK_AUTO_ACCEPTED
    )
    assert session.match_receipts[0].candidates == ()


def test_unanswered_policy_skips_when_canonical_material_claim_is_unavailable() -> None:
    session = session_with(
        answer("answer-must-not-match"),
        max_rounds=4,
        unanswered_action=UNANSWERED_CLARIFICATION_CONTINUE,
    )

    exchange = session.channel.skip_unresolved(
        question_id="question-unresolved",
        dimension=IntentDimension.GOAL,
        question="What outcome should this change achieve?",
        proposed_values=(),
    )

    assert exchange.action is ClarificationAction.SKIP
    assert exchange.matched_answer_id is None
    assert exchange.material_claim == exchange.question
    assert session.consumed_answer_ids == frozenset()
    assert session.match_receipts[0].candidates == ()
    assert session.match_receipts[0].outcome is MaterialClaimMatchOutcome.UNMATCHED


def test_each_answer_is_consumed_at_most_once_and_exhaustion_is_recorded() -> None:
    session = session_with(answer("answer-once"))

    answered = session.channel.ask(**question("question-1"))
    exhausted = session.channel.ask(**question("question-2"))

    assert answered.matched_answer_id == "answer-once"
    assert_unanswered(exhausted)
    assert exhausted.turn_index == 2
    assert session.transcript == (answered, exhausted)


def test_questions_beyond_script_max_rounds_remain_unanswered_and_unconsumed() -> None:
    session = session_with(
        answer("answer-late"),
        max_rounds=1,
    )

    first = session.channel.ask(
        **question("question-wrong", material_claim="A different exact claim")
    )
    over_budget = session.channel.ask(**question("question-too-late"))

    assert_unanswered(first)
    assert_unanswered(over_budget)
    assert over_budget.turn_index == 2
    assert session.match_receipts[0].outcome is MaterialClaimMatchOutcome.UNMATCHED
    assert session.match_receipts[1].outcome is MaterialClaimMatchOutcome.ROUND_LIMIT
    assert session.transcript == (first, over_budget)
    assert session.consumed_answer_ids == frozenset()


def test_duplicate_question_id_raises_without_corrupting_contiguous_transcript() -> None:
    session = session_with()
    first = session.channel.ask(**question("question-1"))

    with pytest.raises(
        ClarificationProtocolError,
        match=r"duplicate clarification question_id: 'question-1'",
    ):
        session.channel.ask(**question("question-1"))

    second = session.channel.ask(**question("question-2"))
    assert first.turn_index == 1
    assert second.turn_index == 2
    assert session.transcript == (first, second)


def test_canonical_global_question_limit_raises_without_an_illegal_exchange() -> None:
    session = session_with()

    for index in range(1, MAX_CLARIFICATION_QUESTIONS + 1):
        exchange = session.channel.ask(**question("question-%d" % index))
        assert exchange.turn_index == index

    with pytest.raises(
        ClarificationProtocolError,
        match="canonical clarification question limit of %d"
        % MAX_CLARIFICATION_QUESTIONS,
    ):
        session.channel.ask(**question("question-over-limit"))

    assert len(session.transcript) == MAX_CLARIFICATION_QUESTIONS
    assert tuple(item.turn_index for item in session.transcript) == tuple(
        range(1, MAX_CLARIFICATION_QUESTIONS + 1)
    )

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

import review_agent_eval.intent_evaluator as intent_evaluator_module
from review_agent_eval.clarification import (
    MaterialClaimCandidateDecision,
    MaterialClaimMatchOutcome,
    MaterialClaimMatchReceipt,
)
from review_agent_eval.intent_evaluator import (
    EXACT_INTENT_WEIGHT,
    INTENT_EVALUATION_SCHEMA_VERSION,
    INTENT_NORMALIZATION_POLICY_VERSION,
    NORMALIZED_INTENT_WEIGHT,
    SEMANTIC_FULL_INTENT_WEIGHT_BASE,
    SEMANTIC_PARTIAL_INTENT_WEIGHT_BASE,
    GeneratedIntentClaim,
    IntentClaimOrigin,
    IntentEvaluationError,
    IntentEvaluationResult,
    IntentEvaluationStatus,
    IntentEvaluator,
    IntentJudgeRelation,
    IntentMatchKind,
    IntentReasonCode,
    IntentSemanticJudgeDecision,
    IntentSemanticJudgeFailure,
    IntentSemanticJudgeRequest,
    IntentSemanticJudgeUngraded,
    normalize_intent_text,
)
from review_agent_eval.models import (
    ClarificationAction,
    ClarificationAnswer,
    ClarificationPolicy,
    ClarificationScript,
    ExpectedIntentClaim,
    ForbiddenIntentClaim,
    IntentAuthority,
    IntentClaimJudgement,
    IntentClaimSource,
    IntentDimension,
    IntentResult,
    IntentTruth,
    SubmissionClarificationExchange,
    SubmissionIntent,
    SubmissionIntentClaim,
    canonical_json,
    canonical_sha256,
)


MATCHER_DIGEST = "a" * 64
EVALUATOR_REVISION = "task-8-intent-tests-v1"


def submission_claim(
    claim_id: str,
    text: str,
    *,
    dimension: IntentDimension = IntentDimension.GOAL,
    source: IntentClaimSource = IntentClaimSource.EXPLICIT,
) -> SubmissionIntentClaim:
    return SubmissionIntentClaim(
        claim_id=claim_id,
        dimension=dimension,
        text=text,
        source=source,
    )


def submission_intent(
    *,
    status: IntentResult = IntentResult.SUFFICIENT,
    goal: str | None = None,
    acceptance_criteria: tuple[str, ...] = (),
    scope: tuple[str, ...] = (),
    constraints: tuple[str, ...] = (),
    claims: tuple[SubmissionIntentClaim, ...] = (),
    transcript: tuple[SubmissionClarificationExchange, ...] = (),
) -> SubmissionIntent:
    return SubmissionIntent(
        status=status,
        goal=goal,
        acceptance_criteria=acceptance_criteria,
        scope=scope,
        constraints=constraints,
        claims=claims,
        clarification_questions=transcript,
        uncertainties=(),
    )


def expected_claim(
    truth_id: str,
    text: str,
    *,
    dimension: IntentDimension = IntentDimension.GOAL,
    required: bool = True,
) -> ExpectedIntentClaim:
    return ExpectedIntentClaim(
        truth_id=truth_id,
        dimension=dimension,
        text=text,
        required=required,
    )


def forbidden_claim(
    truth_id: str,
    text: str,
    *,
    dimension: IntentDimension = IntentDimension.GOAL,
) -> ForbiddenIntentClaim:
    return ForbiddenIntentClaim(
        truth_id=truth_id,
        dimension=dimension,
        text=text,
        rationale="This interpretation is explicitly excluded.",
    )


def intent_truth(
    *expected: ExpectedIntentClaim,
    forbidden: tuple[ForbiddenIntentClaim, ...] = (),
    policy: ClarificationPolicy = ClarificationPolicy.NOT_REQUIRED,
    scorable: bool = True,
) -> IntentTruth:
    if not scorable:
        assert not expected and not forbidden
        return IntentTruth(
            scorable=False,
            authority=None,
            expected_claims=(),
            forbidden_claims=(),
            clarification_policy=None,
        )
    return IntentTruth(
        scorable=True,
        authority=IntentAuthority.SYNTHETIC,
        expected_claims=tuple(expected),
        forbidden_claims=forbidden,
        clarification_policy=policy,
    )


def clarification_answer(
    answer_id: str = "answer-1",
    *,
    dimension: IntentDimension = IntentDimension.GOAL,
    material_claim: str = "Support dry-run mode",
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


def clarification_script(
    *answers: ClarificationAnswer,
    max_rounds: int = 4,
) -> ClarificationScript:
    return ClarificationScript(max_rounds=max_rounds, answers=tuple(answers))


def clarification_exchange(
    *,
    turn_index: int = 1,
    question_id: str = "question-1",
    dimension: IntentDimension = IntentDimension.GOAL,
    material_claim: str = "Support dry-run mode",
    matched_answer_id: str | None = "answer-1",
    action: ClarificationAction | None = ClarificationAction.CONFIRM,
    response: str | None = "Yes",
    resolved_values: tuple[str, ...] | None = None,
) -> SubmissionClarificationExchange:
    if action is None:
        matched_answer_id = None
        response = None
        resolved = ()
    elif resolved_values is None:
        resolved = (
            (material_claim,)
            if action is ClarificationAction.CONFIRM
            else ()
        )
    else:
        resolved = resolved_values
    return SubmissionClarificationExchange(
        turn_index=turn_index,
        question_id=question_id,
        dimension=dimension,
        question="Does this material claim belong in the final Intent?",
        material_claim=material_claim,
        matched_answer_id=matched_answer_id,
        action=action,
        response=response,
        resolved_values=resolved,
    )


def match_receipt(
    exchange: SubmissionClarificationExchange,
    script: ClarificationScript,
    *,
    equivalent_answer_ids: set[str] | None = None,
    action_ineligible_answer_ids: set[str] | None = None,
    consumed_answer_ids: set[str] | None = None,
    matcher_digest: str = MATCHER_DIGEST,
) -> MaterialClaimMatchReceipt:
    equivalent_ids = (
        ({exchange.matched_answer_id} if exchange.matched_answer_id else set())
        if equivalent_answer_ids is None
        else set(equivalent_answer_ids)
    )
    ineligible_ids = set(action_ineligible_answer_ids or ())
    consumed_ids = set(consumed_answer_ids or ())

    if exchange.turn_index > script.max_rounds:
        candidates: tuple[MaterialClaimCandidateDecision, ...] = ()
        outcome = MaterialClaimMatchOutcome.ROUND_LIMIT
        matched_answer_id = None
    else:
        candidates = tuple(
            MaterialClaimCandidateDecision(
                answer_id=answer.answer_id,
                request_digest=canonical_sha256(
                    {
                        "matcher_digest": matcher_digest,
                        "dimension": exchange.dimension.value,
                        "actual_claim": exchange.material_claim,
                        "scripted_claim": answer.material_claim,
                        "answer_id": answer.answer_id,
                    }
                ),
                equivalent=answer.answer_id in equivalent_ids,
                action_eligible=answer.answer_id not in ineligible_ids,
            )
            for answer in script.answers
            if answer.dimension is exchange.dimension
            and answer.answer_id not in consumed_ids
        )
        eligible = tuple(
            item
            for item in candidates
            if item.equivalent and item.action_eligible
        )
        if len(eligible) == 1:
            outcome = MaterialClaimMatchOutcome.MATCHED
            matched_answer_id = eligible[0].answer_id
        elif len(eligible) > 1:
            outcome = MaterialClaimMatchOutcome.AMBIGUOUS
            matched_answer_id = None
        else:
            outcome = MaterialClaimMatchOutcome.UNMATCHED
            matched_answer_id = None

    assert matched_answer_id == exchange.matched_answer_id
    return MaterialClaimMatchReceipt(
        turn_index=exchange.turn_index,
        question_id=exchange.question_id,
        dimension=exchange.dimension,
        actual_claim_digest=canonical_sha256(
            {
                "dimension": exchange.dimension.value,
                "material_claim": exchange.material_claim,
            }
        ),
        matcher_digest=matcher_digest,
        candidates=candidates,
        outcome=outcome,
        matched_answer_id=matched_answer_id,
    )


def benchmark_auto_accept_receipt(
    exchange: SubmissionClarificationExchange,
    *,
    matcher_digest: str = MATCHER_DIGEST,
) -> MaterialClaimMatchReceipt:
    return MaterialClaimMatchReceipt(
        turn_index=exchange.turn_index,
        question_id=exchange.question_id,
        dimension=exchange.dimension,
        actual_claim_digest=canonical_sha256(
            {
                "dimension": exchange.dimension.value,
                "material_claim": exchange.material_claim,
            }
        ),
        matcher_digest=matcher_digest,
        candidates=(),
        outcome=MaterialClaimMatchOutcome.BENCHMARK_AUTO_ACCEPTED,
        matched_answer_id=None,
    )


def judge_decision(
    request: IntentSemanticJudgeRequest,
    relation: IntentJudgeRelation,
    *,
    score_ppm: int = 700_000,
) -> IntentSemanticJudgeDecision:
    return IntentSemanticJudgeDecision(
        request_id=request.request_id,
        relation=relation,
        score_ppm=score_ppm,
        reason_refs=(request.generated_id,),
    )


def judge_failure(
    request: IntentSemanticJudgeRequest,
    *,
    failure_code: str = "attempts_exhausted",
    evaluator_execution_digest: str = "e" * 64,
    judge_result_digest: str | None = None,
) -> IntentSemanticJudgeFailure:
    return IntentSemanticJudgeFailure(
        request_id=request.request_id,
        failure_code=failure_code,
        evaluator_execution_digest=evaluator_execution_digest,
        judge_result_digest=(
            canonical_sha256(
                {
                    "kind": "judge_failed",
                    "request_id": request.request_id,
                    "failure_code": failure_code,
                }
            )
            if judge_result_digest is None
            else judge_result_digest
        ),
    )


def judge_ungraded(
    request: IntentSemanticJudgeRequest,
    *,
    ungraded_reason: str = "policy_skipped",
    evaluator_execution_digest: str = "e" * 64,
    judge_result_digest: str | None = None,
) -> IntentSemanticJudgeUngraded:
    return IntentSemanticJudgeUngraded(
        request_id=request.request_id,
        ungraded_reason=ungraded_reason,
        evaluator_execution_digest=evaluator_execution_digest,
        judge_result_digest=(
            canonical_sha256(
                {
                    "kind": "ungraded",
                    "request_id": request.request_id,
                    "ungraded_reason": ungraded_reason,
                }
            )
            if judge_result_digest is None
            else judge_result_digest
        ),
    )


def evaluate(
    intent: SubmissionIntent | None,
    truth: IntentTruth,
    *,
    script: ClarificationScript | None = None,
    receipts: tuple[MaterialClaimMatchReceipt, ...] = (),
    decisions: tuple[IntentSemanticJudgeDecision, ...] = (),
    failures: tuple[IntentSemanticJudgeFailure, ...] = (),
    ungraded: tuple[IntentSemanticJudgeUngraded, ...] = (),
) -> IntentEvaluationResult:
    return _intent_evaluator().evaluate(
        intent,
        truth,
        clarification_script() if script is None else script,
        receipts=receipts,
        semantic_decisions=decisions,
        semantic_failures=failures,
        semantic_ungraded=ungraded,
    )


def _intent_evaluator() -> IntentEvaluator:
    return IntentEvaluator(evaluator_revision=EVALUATOR_REVISION)


def hydrate_dict(
    value: object,
    intent: SubmissionIntent | None,
    truth: IntentTruth,
    *,
    script: ClarificationScript | None = None,
    receipts: tuple[MaterialClaimMatchReceipt, ...] = (),
    decisions: tuple[IntentSemanticJudgeDecision, ...] = (),
    failures: tuple[IntentSemanticJudgeFailure, ...] = (),
    ungraded: tuple[IntentSemanticJudgeUngraded, ...] = (),
    evaluator: IntentEvaluator | None = None,
) -> IntentEvaluationResult:
    return IntentEvaluationResult.from_dict(
        value,
        evaluator=_intent_evaluator() if evaluator is None else evaluator,
        submission_intent=intent,
        intent_truth=truth,
        clarification_script=clarification_script() if script is None else script,
        clarification_match_receipts=receipts,
        semantic_decisions=decisions,
        semantic_failures=failures,
        semantic_ungraded=ungraded,
    )


def hydrate_json(
    value: object,
    intent: SubmissionIntent | None,
    truth: IntentTruth,
    *,
    script: ClarificationScript | None = None,
    receipts: tuple[MaterialClaimMatchReceipt, ...] = (),
    decisions: tuple[IntentSemanticJudgeDecision, ...] = (),
    failures: tuple[IntentSemanticJudgeFailure, ...] = (),
    ungraded: tuple[IntentSemanticJudgeUngraded, ...] = (),
    evaluator: IntentEvaluator | None = None,
) -> IntentEvaluationResult:
    return IntentEvaluationResult.from_json(
        value,
        evaluator=_intent_evaluator() if evaluator is None else evaluator,
        submission_intent=intent,
        intent_truth=truth,
        clarification_script=clarification_script() if script is None else script,
        clarification_match_receipts=receipts,
        semantic_decisions=decisions,
        semantic_failures=failures,
        semantic_ungraded=ungraded,
    )


def semantic_fixture() -> tuple[SubmissionIntent, IntentTruth, IntentEvaluationResult]:
    intent = submission_intent(goal="Let users preview changes")
    truth = intent_truth(expected_claim("truth-goal", "Support dry-run mode"))
    return intent, truth, evaluate(intent, truth)


@pytest.mark.parametrize(
    ("generated_text", "truth_text", "match_kind", "weight", "reason"),
    [
        (
            "Support dry-run mode",
            "Support dry-run mode",
            IntentMatchKind.EXACT,
            EXACT_INTENT_WEIGHT,
            IntentReasonCode.DETERMINISTIC_EXACT,
        ),
        (
            "  CAFÉ\tMode ",
            "cafe\u0301 mode",
            IntentMatchKind.NORMALIZED,
            NORMALIZED_INTENT_WEIGHT,
            IntentReasonCode.DETERMINISTIC_NORMALIZED,
        ),
    ],
)
def test_exact_and_normalized_matches_are_deterministic(
    generated_text: str,
    truth_text: str,
    match_kind: IntentMatchKind,
    weight: int,
    reason: IntentReasonCode,
) -> None:
    intent = submission_intent(goal=generated_text)
    truth = intent_truth(expected_claim("truth-goal", truth_text))

    result = evaluate(intent, truth)

    assert normalize_intent_text("  CAFÉ\tMode! ") == "café mode!"
    assert result.status is IntentEvaluationStatus.GRADED
    assert result.judge_requests == ()
    assert len(result.assignments) == 1
    assert result.assignments[0].weight == weight
    assert result.candidates[0].match_kind is match_kind
    assert result.candidates[0].reason_codes == (reason,)
    assert result.claim_outcomes[0].judgement is IntentClaimJudgement.SUPPORTED
    assert result.metrics.intent_claim_precision_numerator == 1
    assert result.metrics.intent_claim_precision_denominator == 1
    assert result.metrics.intent_claim_recall_numerator == 1
    assert result.metrics.intent_claim_recall_denominator == 1
    assert result.metrics.intent_case_pass is True


def test_candidate_generation_is_isolated_by_intent_dimension() -> None:
    intent = submission_intent(goal="Keep the public API stable")
    truth = intent_truth(
        expected_claim(
            "truth-scope",
            "Keep the public API stable",
            dimension=IntentDimension.SCOPE,
        )
    )

    result = evaluate(intent, truth)

    assert result.candidates == ()
    assert result.judge_requests == ()
    assert result.assignments == ()
    assert result.claim_outcomes[0].judgement is IntentClaimJudgement.UNSUPPORTED
    assert result.claim_outcomes[0].reason_codes == (
        IntentReasonCode.NO_TRUTH_CANDIDATE,
    )
    assert result.unmatched_expected_truth_ids == ("truth-scope",)


def test_structured_and_provenance_claims_overlay_one_to_one() -> None:
    intent = submission_intent(
        acceptance_criteria=("Return JSON", "Return JSON"),
        claims=(
            submission_claim(
                "claim-c",
                " return json ",
                dimension=IntentDimension.ACCEPTANCE_CRITERION,
            ),
            submission_claim(
                "claim-a",
                "RETURN JSON",
                dimension=IntentDimension.ACCEPTANCE_CRITERION,
                source=IntentClaimSource.INFERRED,
            ),
            submission_claim(
                "claim-b",
                "Return JSON",
                dimension=IntentDimension.ACCEPTANCE_CRITERION,
            ),
        ),
    )

    result = evaluate(intent, intent_truth())

    assert len(result.generated_claims) == 3
    overlays = tuple(
        item
        for item in result.generated_claims
        if item.origin is IntentClaimOrigin.STRUCTURED_OVERLAY
    )
    provenance = tuple(
        item
        for item in result.generated_claims
        if item.origin is IntentClaimOrigin.PROVENANCE
    )
    assert {item.provenance_claim_id for item in overlays} == {
        "claim-a",
        "claim-b",
    }
    assert len(provenance) == 1
    assert provenance[0].provenance_claim_id == "claim-c"
    assert all(item.normalized_text == "return json" for item in result.generated_claims)


def test_overlay_never_crosses_dimensions() -> None:
    intent = submission_intent(
        goal="Keep the public API stable",
        claims=(
            submission_claim(
                "claim-scope",
                "KEEP THE PUBLIC API STABLE",
                dimension=IntentDimension.SCOPE,
            ),
        ),
    )

    result = evaluate(intent, intent_truth())

    assert {item.origin for item in result.generated_claims} == {
        IntentClaimOrigin.STRUCTURED,
        IntentClaimOrigin.PROVENANCE,
    }
    assert {item.dimension for item in result.generated_claims} == {
        IntentDimension.GOAL,
        IntentDimension.SCOPE,
    }


def test_projection_normalizes_each_input_claim_a_bounded_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = intent_evaluator_module.normalize_intent_text
    calls = 0

    def counting_normalizer(value: str) -> str:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(
        intent_evaluator_module, "normalize_intent_text", counting_normalizer
    )
    intent = submission_intent(
        acceptance_criteria=tuple(f"criterion {index}" for index in range(16)),
        claims=tuple(
            submission_claim(
                f"claim-{index}",
                f"criterion {index}",
                dimension=IntentDimension.ACCEPTANCE_CRITERION,
            )
            for index in range(16)
        ),
    )

    projected = intent_evaluator_module._project_generated_claims(intent)

    assert len(projected) == 16
    assert calls <= 2 * (len(intent.acceptance_criteria) + len(intent.claims))


def test_full_legal_submission_projection_exceeds_provenance_limit_safely() -> None:
    claims = tuple(
        submission_claim(f"claim-{index:04d}", f"claim text {index}")
        for index in range(1_024)
    )

    result = evaluate(
        submission_intent(goal="one additional structural goal", claims=claims),
        intent_truth(),
    )

    assert len(result.generated_claims) == 1_025
    assert len(result.claim_outcomes) == 1_025
    assert result.metrics.unsupported_claim_count == 1_025


def test_duplicate_generated_claims_receive_only_one_assignment() -> None:
    intent = submission_intent(scope=("Keep the API stable", "Keep the API stable"))
    truth = intent_truth(
        expected_claim(
            "truth-scope",
            "Keep the API stable",
            dimension=IntentDimension.SCOPE,
        )
    )

    result = evaluate(intent, truth)

    assert len(result.candidates) == 2
    assert len(result.assignments) == 1
    assert len({item.left_id for item in result.assignments}) == 1
    assert len({item.right_id for item in result.assignments}) == 1
    assert sorted(item.judgement.value for item in result.claim_outcomes) == [
        IntentClaimJudgement.SUPPORTED.value,
        IntentClaimJudgement.UNSUPPORTED.value,
    ]
    unmatched = next(
        item
        for item in result.claim_outcomes
        if item.judgement is IntentClaimJudgement.UNSUPPORTED
    )
    assert unmatched.reason_codes == (IntentReasonCode.UNMATCHED_DUPLICATE,)
    assert result.metrics.required_supported_count == 1
    assert result.metrics.required_missed_count == 0
    assert result.metrics.intent_claim_precision_numerator == 1
    assert result.metrics.intent_claim_precision_denominator == 2


def test_inferred_provenance_is_not_intrinsically_unsupported() -> None:
    intent = submission_intent(
        claims=(
            submission_claim(
                "claim-inferred",
                "Preserve backwards compatibility",
                source=IntentClaimSource.INFERRED,
            ),
        )
    )
    truth = intent_truth(
        expected_claim("truth-goal", "Preserve backwards compatibility")
    )

    result = evaluate(intent, truth)

    assert result.generated_claims[0].source is IntentClaimSource.INFERRED
    assert result.claim_outcomes[0].judgement is IntentClaimJudgement.SUPPORTED


def test_unresolved_semantic_candidate_is_pending_and_unknown() -> None:
    _, _, result = semantic_fixture()

    assert result.status is IntentEvaluationStatus.PENDING_JUDGE
    assert len(result.judge_requests) == 1
    candidate = result.candidates[0]
    assert candidate.match_kind is IntentMatchKind.SEMANTIC
    assert candidate.relation is None
    assert candidate.score_ppm is None
    assert candidate.edge_weight is None
    assert candidate.judgement is None
    assert result.assignments == ()
    assert result.claim_outcomes[0].judgement is IntentClaimJudgement.UNKNOWN
    assert result.claim_outcomes[0].reason_codes == (IntentReasonCode.JUDGE_PENDING,)
    assert result.metrics.unknown_claim_count == 1
    assert result.metrics.intent_case_pass is None


@pytest.mark.parametrize(
    (
        "relation",
        "expected_judgement",
        "expected_status",
        "expected_weight",
        "expected_reason",
    ),
    [
        (
            IntentJudgeRelation.EQUIVALENT,
            IntentClaimJudgement.SUPPORTED,
            IntentEvaluationStatus.GRADED,
            SEMANTIC_FULL_INTENT_WEIGHT_BASE + 700_000,
            IntentReasonCode.SEMANTIC_EQUIVALENT,
        ),
        (
            IntentJudgeRelation.PARTIALLY_EQUIVALENT,
            IntentClaimJudgement.PARTIALLY_SUPPORTED,
            IntentEvaluationStatus.GRADED,
            SEMANTIC_PARTIAL_INTENT_WEIGHT_BASE + 700_000,
            IntentReasonCode.SEMANTIC_PARTIAL,
        ),
        (
            IntentJudgeRelation.CONTRADICTED,
            IntentClaimJudgement.CONTRADICTED,
            IntentEvaluationStatus.GRADED,
            SEMANTIC_FULL_INTENT_WEIGHT_BASE + 700_000,
            IntentReasonCode.SEMANTIC_CONTRADICTED,
        ),
        (
            IntentJudgeRelation.DIFFERENT,
            IntentClaimJudgement.UNSUPPORTED,
            IntentEvaluationStatus.GRADED,
            None,
            IntentReasonCode.JUDGE_DIFFERENT,
        ),
        (
            IntentJudgeRelation.UNKNOWN,
            IntentClaimJudgement.UNKNOWN,
            IntentEvaluationStatus.UNGRADED,
            None,
            IntentReasonCode.JUDGE_UNKNOWN,
        ),
    ],
)
def test_typed_judge_relations_merge_fail_closed(
    relation: IntentJudgeRelation,
    expected_judgement: IntentClaimJudgement,
    expected_status: IntentEvaluationStatus,
    expected_weight: int | None,
    expected_reason: IntentReasonCode,
) -> None:
    intent, truth, pending = semantic_fixture()
    decision = judge_decision(pending.judge_requests[0], relation)

    result = evaluate(intent, truth, decisions=(decision,))

    assert result.status is expected_status
    candidate = result.candidates[0]
    assert candidate.relation is relation
    assert candidate.score_ppm == 700_000
    assert candidate.edge_weight == expected_weight
    assert candidate.reason_codes == (expected_reason,)
    assert result.claim_outcomes[0].judgement is expected_judgement
    assert (len(result.assignments) == 1) is (expected_weight is not None)
    assert result.metrics.intent_case_pass is (
        True if relation is IntentJudgeRelation.EQUIVALENT else None
        if relation is IntentJudgeRelation.UNKNOWN
        else False
    )


def test_partial_judge_decision_set_keeps_remaining_work_pending() -> None:
    intent = submission_intent(
        goal="Generated goal",
        scope=("Generated scope",),
    )
    truth = intent_truth(
        expected_claim("truth-goal", "Expected goal"),
        expected_claim(
            "truth-scope",
            "Expected scope",
            dimension=IntentDimension.SCOPE,
        ),
    )
    pending = evaluate(intent, truth)
    goal_request = next(
        item
        for item in pending.judge_requests
        if item.dimension is IntentDimension.GOAL
    )

    result = evaluate(
        intent,
        truth,
        decisions=(judge_decision(goal_request, IntentJudgeRelation.EQUIVALENT),),
    )

    assert result.status is IntentEvaluationStatus.PENDING_JUDGE
    assert len(result.judge_requests) == 2
    assert len(result.judge_decisions) == 1
    assert len(result.assignments) == 1
    assert {item.judgement for item in result.claim_outcomes} == {
        IntentClaimJudgement.SUPPORTED,
        IntentClaimJudgement.UNKNOWN,
    }


def test_judge_execution_failure_is_ungraded_not_pending_or_semantic_unknown() -> None:
    intent, truth, pending = semantic_fixture()
    request = pending.judge_requests[0]
    failure = judge_failure(request)

    result = evaluate(intent, truth, failures=(failure,))

    assert result.status is IntentEvaluationStatus.UNGRADED
    assert result.judge_failures == (failure,)
    assert result.judge_decisions == ()
    assert result.judge_ungraded == ()
    assert result.candidates[0].reason_codes == (IntentReasonCode.JUDGE_FAILED,)
    assert result.claim_outcomes[0].reason_codes == (
        IntentReasonCode.JUDGE_FAILED,
    )
    assert IntentReasonCode.JUDGE_FAILED in result.reason_codes
    assert IntentReasonCode.JUDGE_PENDING not in result.reason_codes
    assert IntentReasonCode.JUDGE_UNKNOWN not in result.reason_codes
    assert hydrate_json(
        result.to_json(),
        intent,
        truth,
        failures=(failure,),
    ) == result


@pytest.mark.parametrize(
    "ungraded_reason",
    ["policy_skipped", "upstream_missing", "not_scorable"],
)
def test_judge_ungraded_is_not_pending_failed_or_semantic_unknown(
    ungraded_reason: str,
) -> None:
    intent, truth, pending = semantic_fixture()
    receipt = judge_ungraded(
        pending.judge_requests[0],
        ungraded_reason=ungraded_reason,
    )

    result = evaluate(intent, truth, ungraded=(receipt,))

    assert result.status is IntentEvaluationStatus.UNGRADED
    assert result.judge_ungraded == (receipt,)
    assert result.judge_failures == ()
    assert result.judge_decisions == ()
    assert result.candidates[0].reason_codes == (
        IntentReasonCode.JUDGE_UNGRADED,
    )
    assert result.claim_outcomes[0].reason_codes == (
        IntentReasonCode.JUDGE_UNGRADED,
    )
    assert IntentReasonCode.JUDGE_UNGRADED in result.reason_codes
    assert IntentReasonCode.JUDGE_PENDING not in result.reason_codes
    assert IntentReasonCode.JUDGE_FAILED not in result.reason_codes
    assert IntentReasonCode.JUDGE_UNKNOWN not in result.reason_codes
    assert hydrate_json(
        result.to_json(),
        intent,
        truth,
        ungraded=(receipt,),
    ) == result


def test_failure_and_ungraded_receipts_have_strict_codes_and_provenance() -> None:
    _intent, _truth, pending = semantic_fixture()
    request = pending.judge_requests[0]

    with pytest.raises(IntentEvaluationError, match="unsupported failure_code"):
        judge_failure(request, failure_code="future_failure")
    with pytest.raises(IntentEvaluationError, match="unsupported ungraded_reason"):
        judge_ungraded(request, ungraded_reason="future_ungraded")
    with pytest.raises(IntentEvaluationError, match="SHA-256"):
        judge_failure(request, evaluator_execution_digest="short")
    with pytest.raises(IntentEvaluationError, match="SHA-256"):
        judge_ungraded(request, judge_result_digest="short")


def test_failure_ungraded_and_pending_are_distinct_mixed_resolutions() -> None:
    intent = submission_intent(
        goal="Generated goal",
        scope=("Generated scope",),
        constraints=("Generated constraint",),
    )
    truth = intent_truth(
        expected_claim("truth-goal", "Expected goal"),
        expected_claim(
            "truth-scope",
            "Expected scope",
            dimension=IntentDimension.SCOPE,
        ),
        expected_claim(
            "truth-constraint",
            "Expected constraint",
            dimension=IntentDimension.CONSTRAINT,
        ),
    )
    pending = evaluate(intent, truth)
    by_dimension = {item.dimension: item for item in pending.judge_requests}
    failure = judge_failure(by_dimension[IntentDimension.GOAL])
    ungraded_receipt = judge_ungraded(by_dimension[IntentDimension.SCOPE])

    mixed = evaluate(
        intent,
        truth,
        failures=(failure,),
        ungraded=(ungraded_receipt,),
    )

    assert mixed.status is IntentEvaluationStatus.PENDING_JUDGE
    assert mixed.judge_failures == (failure,)
    assert mixed.judge_ungraded == (ungraded_receipt,)
    assert {
        IntentReasonCode.JUDGE_FAILED,
        IntentReasonCode.JUDGE_UNGRADED,
        IntentReasonCode.JUDGE_PENDING,
    }.issubset(mixed.reason_codes)
    assert Counter(
        item.reason_codes[0] for item in mixed.candidates
    ) == Counter(
        {
            IntentReasonCode.JUDGE_FAILED: 1,
            IntentReasonCode.JUDGE_UNGRADED: 1,
            IntentReasonCode.JUDGE_PENDING: 1,
        }
    )

    decision = judge_decision(
        by_dimension[IntentDimension.CONSTRAINT],
        IntentJudgeRelation.DIFFERENT,
    )
    resolved = evaluate(
        intent,
        truth,
        decisions=(decision,),
        failures=(failure,),
        ungraded=(ungraded_receipt,),
    )
    assert resolved.status is IntentEvaluationStatus.UNGRADED
    assert IntentReasonCode.JUDGE_PENDING not in resolved.reason_codes
    assert hydrate_json(
        resolved.to_json(),
        intent,
        truth,
        decisions=(decision,),
        failures=(failure,),
        ungraded=(ungraded_receipt,),
    ) == resolved


def test_duplicate_and_unknown_judge_decisions_are_rejected() -> None:
    intent, truth, pending = semantic_fixture()
    decision = judge_decision(
        pending.judge_requests[0], IntentJudgeRelation.EQUIVALENT
    )

    with pytest.raises(IntentEvaluationError, match="duplicate Judge decision"):
        evaluate(intent, truth, decisions=(decision, decision))

    unknown = replace(decision, request_id="unknown-request")
    with pytest.raises(IntentEvaluationError, match="unknown request"):
        evaluate(intent, truth, decisions=(unknown,))

    with pytest.raises(IntentEvaluationError, match="invalid type"):
        IntentEvaluator().evaluate(
            intent,
            truth,
            clarification_script(),
            semantic_decisions=(decision.to_dict(),),  # type: ignore[arg-type]
        )


def test_judge_reason_refs_and_failure_resolution_cannot_cross_request_boundary() -> None:
    intent, truth, pending = semantic_fixture()
    request = pending.judge_requests[0]
    crossed = replace(
        judge_decision(request, IntentJudgeRelation.EQUIVALENT),
        reason_refs=("another-request-source",),
    )
    with pytest.raises(IntentEvaluationError, match="cross the canonical request"):
        evaluate(intent, truth, decisions=(crossed,))

    decision = judge_decision(request, IntentJudgeRelation.EQUIVALENT)
    failure = judge_failure(request, failure_code="provider_error")
    with pytest.raises(IntentEvaluationError, match="both a decision and a failure"):
        evaluate(intent, truth, decisions=(decision,), failures=(failure,))

    ungraded_receipt = judge_ungraded(request)
    with pytest.raises(IntentEvaluationError, match="more than one resolution"):
        evaluate(
            intent,
            truth,
            failures=(failure,),
            ungraded=(ungraded_receipt,),
        )
    with pytest.raises(IntentEvaluationError, match="more than one resolution"):
        evaluate(
            intent,
            truth,
            decisions=(decision,),
            ungraded=(ungraded_receipt,),
        )

    with pytest.raises(IntentEvaluationError, match="unknown request"):
        evaluate(
            intent,
            truth,
            ungraded=(replace(ungraded_receipt, request_id="unknown-request"),),
        )


def test_duplicate_failure_and_ungraded_receipts_are_rejected() -> None:
    intent, truth, pending = semantic_fixture()
    request = pending.judge_requests[0]
    failure = judge_failure(request)
    ungraded_receipt = judge_ungraded(request)

    with pytest.raises(IntentEvaluationError, match="duplicate Judge failure"):
        evaluate(intent, truth, failures=(failure, failure))
    with pytest.raises(IntentEvaluationError, match="duplicate Judge ungraded"):
        evaluate(intent, truth, ungraded=(ungraded_receipt, ungraded_receipt))


def test_empty_and_unscorable_branches_reject_all_semantic_receipts() -> None:
    intent, truth, pending = semantic_fixture()
    receipt = judge_ungraded(pending.judge_requests[0])

    with pytest.raises(IntentEvaluationError, match="without an Intent"):
        evaluate(None, truth, ungraded=(receipt,))
    with pytest.raises(IntentEvaluationError, match="unscorable Intent truth"):
        evaluate(intent, intent_truth(scorable=False), ungraded=(receipt,))


def test_case_wide_judge_reason_ref_expansion_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent, truth, pending = semantic_fixture()
    monkeypatch.setattr(
        intent_evaluator_module, "MAX_INTENT_JUDGE_REASON_REF_BYTES", 1
    )

    with pytest.raises(IntentEvaluationError, match="reason refs exceed"):
        evaluate(
            intent,
            truth,
            decisions=(
                judge_decision(
                    pending.judge_requests[0], IntentJudgeRelation.EQUIVALENT
                ),
            ),
        )


def test_case_wide_judge_decision_records_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent, truth, pending = semantic_fixture()
    monkeypatch.setattr(
        intent_evaluator_module, "MAX_INTENT_JUDGE_DECISION_BYTES", 1
    )

    with pytest.raises(IntentEvaluationError, match="decisions exceed"):
        evaluate(
            intent,
            truth,
            decisions=(
                judge_decision(
                    pending.judge_requests[0], IntentJudgeRelation.EQUIVALENT
                ),
            ),
        )


def test_required_optional_and_forbidden_truth_have_distinct_metrics() -> None:
    intent = submission_intent(
        goal="Ship dry-run mode",
        scope=("Rewrite the whole repository",),
    )
    truth = intent_truth(
        expected_claim("truth-required", "Ship dry-run mode", required=True),
        expected_claim(
            "truth-optional",
            "Add examples",
            dimension=IntentDimension.ACCEPTANCE_CRITERION,
            required=False,
        ),
        forbidden=(
            forbidden_claim(
                "truth-forbidden",
                "Rewrite the whole repository",
                dimension=IntentDimension.SCOPE,
            ),
        ),
    )

    result = evaluate(intent, truth)

    assert {item.judgement for item in result.claim_outcomes} == {
        IntentClaimJudgement.SUPPORTED,
        IntentClaimJudgement.CONTRADICTED,
    }
    assert result.metrics.required_truth_count == 1
    assert result.metrics.required_supported_count == 1
    assert result.metrics.required_missed_count == 0
    assert result.metrics.optional_truth_count == 1
    assert result.metrics.optional_supported_count == 0
    assert result.metrics.forbidden_truth_count == 1
    assert result.metrics.forbidden_hit_count == 1
    assert IntentReasonCode.OPTIONAL_TRUTH_MISSED in result.reason_codes
    assert IntentReasonCode.FORBIDDEN_TRUTH_HIT in result.reason_codes
    assert result.metrics.intent_case_pass is False


def test_missing_optional_truth_does_not_reduce_recall_or_case_pass() -> None:
    intent = submission_intent()
    truth = intent_truth(
        expected_claim("truth-optional", "Nice to have", required=False)
    )

    result = evaluate(intent, truth)

    assert result.metrics.required_truth_count == 0
    assert result.metrics.intent_claim_recall_numerator == 0
    assert result.metrics.intent_claim_recall_denominator == 0
    assert result.metrics.optional_truth_count == 1
    assert result.metrics.optional_supported_count == 0
    assert result.metrics.intent_case_pass is True


def test_missing_required_truth_is_counted_without_fabricating_a_claim() -> None:
    intent = submission_intent()
    truth = intent_truth(expected_claim("truth-required", "Required behavior"))

    result = evaluate(intent, truth)

    assert result.claim_outcomes == ()
    assert result.metrics.required_truth_count == 1
    assert result.metrics.required_supported_count == 0
    assert result.metrics.required_missed_count == 1
    assert result.metrics.intent_case_pass is False
    assert IntentReasonCode.REQUIRED_TRUTH_MISSED in result.reason_codes


def test_unscorable_truth_preserves_audit_projection_but_nulls_all_metrics() -> None:
    intent = submission_intent(goal="An auditable generated claim")

    result = evaluate(intent, intent_truth(scorable=False))

    assert result.status is IntentEvaluationStatus.NOT_SCORABLE
    assert len(result.generated_claims) == 1
    assert result.candidates == ()
    assert result.judge_requests == ()
    assert result.claim_outcomes[0].judgement is IntentClaimJudgement.UNKNOWN
    assert result.metrics.scorable is False
    assert all(
        value is None
        for key, value in result.metrics.to_dict().items()
        if key != "scorable"
    )
    assert result.reason_codes == (IntentReasonCode.INTENT_TRUTH_UNSCORABLE,)


def test_missing_submission_intent_is_ungraded_and_not_zero_scored() -> None:
    truth = intent_truth(
        expected_claim("truth-required", "Required behavior"),
        policy=ClarificationPolicy.REQUIRED,
    )

    result = evaluate(None, truth)

    assert result.status is IntentEvaluationStatus.UNGRADED
    assert result.generated_claims == ()
    assert result.claim_outcomes == ()
    assert result.submission_intent_digest is None
    assert result.unmatched_expected_truth_ids == ("truth-required",)
    assert result.metrics.scorable is False
    assert result.clarification.policy is ClarificationPolicy.REQUIRED
    assert result.clarification.decision_correct is None
    assert result.clarification.complete is None
    assert result.reason_codes == (IntentReasonCode.SUBMISSION_INTENT_MISSING,)
    assert hydrate_dict(result.to_dict(), None, truth) == result
    assert hydrate_json(result.to_json(), None, truth) == result


def test_candidate_cap_returns_ungraded_without_truncating_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(intent_evaluator_module, "MAX_INTENT_CANDIDATE_EDGES", 1)
    intent = submission_intent(scope=("Generated one", "Generated two"))
    truth = intent_truth(
        expected_claim(
            "truth-scope",
            "Truth scope",
            dimension=IntentDimension.SCOPE,
        )
    )

    result = evaluate(intent, truth)

    assert result.status is IntentEvaluationStatus.UNGRADED
    assert result.candidates == ()
    assert result.judge_requests == ()
    assert result.assignments == ()
    assert all(
        item.judgement is IntentClaimJudgement.UNKNOWN
        for item in result.claim_outcomes
    )
    assert result.metrics.scorable is False
    assert result.reason_codes == (
        IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED,
    )


def test_total_candidate_record_cap_also_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(intent_evaluator_module, "MAX_INTENT_TOTAL_CANDIDATES", 1)
    intent = submission_intent(scope=("same", "same"))
    truth = intent_truth(
        expected_claim(
            "truth-scope",
            "same",
            dimension=IntentDimension.SCOPE,
        )
    )

    result = evaluate(intent, truth)

    assert result.status is IntentEvaluationStatus.UNGRADED
    assert result.candidates == ()
    assert result.metrics.scorable is False
    assert result.reason_codes == (
        IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED,
    )


def test_semantic_request_text_expansion_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(intent_evaluator_module, "MAX_INTENT_REQUEST_TEXT_BYTES", 1)
    intent = submission_intent(goal="generated")
    truth = intent_truth(expected_claim("truth-goal", "expected"))

    result = evaluate(intent, truth)

    assert result.status is IntentEvaluationStatus.UNGRADED
    assert result.judge_requests == ()
    assert result.metrics.scorable is False
    assert result.reason_codes == (
        IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED,
    )


def test_candidate_record_byte_expansion_returns_ungraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intent_evaluator_module, "MAX_INTENT_CANDIDATE_RECORD_BYTES", 1
    )

    result = evaluate(
        submission_intent(goal="same"),
        intent_truth(expected_claim("truth-goal", "same")),
    )

    assert result.status is IntentEvaluationStatus.UNGRADED
    assert result.candidates == ()
    assert result.metrics.scorable is False
    assert result.reason_codes == (
        IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED,
    )


def test_semantic_request_record_expansion_returns_ungraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intent_evaluator_module, "MAX_INTENT_JUDGE_REQUEST_BYTES", 1
    )

    result = evaluate(
        submission_intent(goal="generated"),
        intent_truth(expected_claim("truth-goal", "expected")),
    )

    assert result.status is IntentEvaluationStatus.UNGRADED
    assert result.judge_requests == ()
    assert result.metrics.scorable is False
    assert result.reason_codes == (
        IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED,
    )


@pytest.mark.parametrize(
    ("policy", "decision_correct", "complete", "reason"),
    [
        (
            ClarificationPolicy.REQUIRED,
            False,
            False,
            IntentReasonCode.REQUIRED_CLARIFICATION_NOT_ASKED,
        ),
        (
            ClarificationPolicy.OPTIONAL,
            True,
            True,
            IntentReasonCode.CLARIFICATION_DECISION_CORRECT,
        ),
        (
            ClarificationPolicy.NOT_REQUIRED,
            True,
            True,
            IntentReasonCode.CLARIFICATION_DECISION_CORRECT,
        ),
    ],
)
def test_clarification_policy_without_a_question(
    policy: ClarificationPolicy,
    decision_correct: bool,
    complete: bool,
    reason: IntentReasonCode,
) -> None:
    result = evaluate(submission_intent(), intent_truth(policy=policy))

    assert result.clarification.policy is policy
    assert result.clarification.decision_correct is decision_correct
    assert result.clarification.complete is complete
    assert reason in result.clarification.reason_codes
    assert result.metrics.clarification_numerator == int(decision_correct)
    assert result.metrics.clarification_denominator == 1


@pytest.mark.parametrize(
    (
        "action",
        "material_claim",
        "response",
        "corrected_values",
        "resolved_values",
        "final_values",
        "expected_update",
        "expected_complete",
    ),
    [
        (
            ClarificationAction.CONFIRM,
            "Support dry-run mode",
            "Yes",
            (),
            ("Support dry-run mode",),
            ("Support dry-run mode",),
            True,
            True,
        ),
        (
            ClarificationAction.CORRECT,
            "Only update the CLI",
            "Update both interfaces",
            ("Update the CLI and API",),
            ("Update the CLI and API",),
            ("Update the CLI and API",),
            True,
            True,
        ),
        (
            ClarificationAction.REJECT,
            "Rewrite the whole repository",
            "No",
            (),
            (),
            (),
            True,
            True,
        ),
        (
            ClarificationAction.DEFER,
            "Choose a storage backend",
            "Decide later",
            (),
            (),
            (),
            None,
            False,
        ),
        (
            ClarificationAction.SKIP,
            "Choose a logging format",
            None,
            (),
            (),
            (),
            None,
            True,
        ),
    ],
)
def test_valid_receipt_enforces_clarification_update_rules(
    action: ClarificationAction,
    material_claim: str,
    response: str | None,
    corrected_values: tuple[str, ...],
    resolved_values: tuple[str, ...],
    final_values: tuple[str, ...],
    expected_update: bool | None,
    expected_complete: bool,
) -> None:
    answer = clarification_answer(
        material_claim=material_claim,
        action=action,
        response=response,
        corrected_values=corrected_values,
    )
    script = clarification_script(answer)
    exchange = clarification_exchange(
        material_claim=material_claim,
        action=action,
        response=response,
        resolved_values=resolved_values,
    )
    receipt = match_receipt(exchange, script)
    final_claims = tuple(
        submission_claim(f"final-{index}", value)
        for index, value in enumerate(final_values, start=1)
    )
    intent = submission_intent(claims=final_claims, transcript=(exchange,))

    result = evaluate(
        intent,
        intent_truth(policy=ClarificationPolicy.REQUIRED),
        script=script,
        receipts=(receipt,),
    )

    assert receipt.actual_claim_digest == canonical_sha256(
        {
            "dimension": exchange.dimension.value,
            "material_claim": exchange.material_claim,
        }
    )
    assert result.clarification.decision_correct is True
    assert result.clarification.complete is expected_complete
    exchange_result = result.clarification.exchanges[0]
    assert exchange_result.material is True
    assert exchange_result.answer_consumed is True
    assert exchange_result.update_applied is expected_update
    if action is ClarificationAction.DEFER:
        assert IntentReasonCode.CLARIFICATION_UNRESOLVED in exchange_result.reason_codes


@pytest.mark.parametrize(
    ("action", "material_claim", "corrected_values", "resolved_values", "final_values"),
    [
        (
            ClarificationAction.CONFIRM,
            "Support dry-run mode",
            (),
            ("Support dry-run mode",),
            (),
        ),
        (
            ClarificationAction.CORRECT,
            "Only update the CLI",
            ("Update the CLI and API",),
            ("Update the CLI and API",),
            ("Only update the CLI", "Update the CLI and API"),
        ),
        (
            ClarificationAction.REJECT,
            "Rewrite the whole repository",
            (),
            (),
            ("Rewrite the whole repository",),
        ),
    ],
)
def test_unapplied_clarification_update_is_recorded_separately(
    action: ClarificationAction,
    material_claim: str,
    corrected_values: tuple[str, ...],
    resolved_values: tuple[str, ...],
    final_values: tuple[str, ...],
) -> None:
    response = "Corrected" if action is ClarificationAction.CORRECT else "No"
    if action is ClarificationAction.CONFIRM:
        response = "Yes"
    answer = clarification_answer(
        material_claim=material_claim,
        action=action,
        response=response,
        corrected_values=corrected_values,
    )
    script = clarification_script(answer)
    exchange = clarification_exchange(
        material_claim=material_claim,
        action=action,
        response=response,
        resolved_values=resolved_values,
    )
    receipt = match_receipt(exchange, script)
    intent = submission_intent(
        claims=tuple(
            submission_claim(f"final-{index}", value)
            for index, value in enumerate(final_values, start=1)
        ),
        transcript=(exchange,),
    )

    result = evaluate(
        intent,
        intent_truth(policy=ClarificationPolicy.REQUIRED),
        script=script,
        receipts=(receipt,),
    )

    exchange_result = result.clarification.exchanges[0]
    assert result.clarification.decision_correct is True
    assert result.clarification.complete is False
    assert exchange_result.update_applied is False
    assert IntentReasonCode.CLARIFICATION_ANSWER_NOT_APPLIED in (
        exchange_result.reason_codes
    )


def test_optional_clarification_decision_is_neutral_but_update_is_still_checked() -> None:
    material_claim = "Support dry-run mode"
    answer = clarification_answer(material_claim=material_claim)
    script = clarification_script(answer)
    exchange = clarification_exchange(material_claim=material_claim)
    receipt = match_receipt(exchange, script)
    intent = submission_intent(transcript=(exchange,))

    result = evaluate(
        intent,
        intent_truth(policy=ClarificationPolicy.OPTIONAL),
        script=script,
        receipts=(receipt,),
    )

    assert result.clarification.decision_correct is True
    assert result.clarification.exchanges[0].update_applied is False
    assert result.clarification.complete is False
    assert result.metrics.clarification_numerator == 1


def test_not_required_question_is_unnecessary_even_when_material_and_applied() -> None:
    material_claim = "Support dry-run mode"
    answer = clarification_answer(material_claim=material_claim)
    script = clarification_script(answer)
    exchange = clarification_exchange(material_claim=material_claim)
    receipt = match_receipt(exchange, script)
    intent = submission_intent(
        goal=material_claim,
        transcript=(exchange,),
    )

    result = evaluate(
        intent,
        intent_truth(policy=ClarificationPolicy.NOT_REQUIRED),
        script=script,
        receipts=(receipt,),
    )

    assert result.clarification.decision_correct is False
    assert result.clarification.exchanges[0].material is True
    assert result.clarification.exchanges[0].update_applied is True
    assert IntentReasonCode.CLARIFICATION_UNNECESSARY_QUESTION in (
        result.clarification.reason_codes
    )
    assert result.metrics.clarification_numerator == 0
    assert result.metrics.intent_case_pass is False


def test_not_required_partial_intent_is_unnecessary_blocking() -> None:
    result = evaluate(
        submission_intent(status=IntentResult.PARTIAL),
        intent_truth(policy=ClarificationPolicy.NOT_REQUIRED),
    )

    assert result.clarification.decision_correct is True
    assert result.clarification.complete is False
    assert IntentReasonCode.CLARIFICATION_UNNECESSARY_BLOCKING in (
        result.clarification.reason_codes
    )
    assert IntentReasonCode.CLARIFICATION_UNRESOLVED in (
        result.clarification.reason_codes
    )


def test_missing_receipt_fails_closed_without_recomputing_materiality() -> None:
    material_claim = "Support dry-run mode"
    answer = clarification_answer(material_claim=material_claim)
    script = clarification_script(answer)
    exchange = clarification_exchange(material_claim=material_claim)
    intent = submission_intent(goal=material_claim, transcript=(exchange,))

    result = evaluate(
        intent,
        intent_truth(policy=ClarificationPolicy.REQUIRED),
        script=script,
    )

    assert result.status is IntentEvaluationStatus.UNGRADED
    assert result.clarification.decision_correct is None
    assert result.clarification.complete is False
    exchange_result = result.clarification.exchanges[0]
    assert exchange_result.material is None
    assert exchange_result.answer_consumed is None
    assert exchange_result.update_applied is None
    assert IntentReasonCode.CLARIFICATION_RECEIPT_MISSING in (
        result.clarification.reason_codes
    )


def test_benchmark_auto_accept_receipt_is_material_without_consuming_an_answer() -> None:
    material_claim = "Support dry-run mode"
    exchange = clarification_exchange(
        material_claim=material_claim,
        matched_answer_id=None,
        action=ClarificationAction.CONFIRM,
        response=None,
        resolved_values=(material_claim,),
    )
    receipt = benchmark_auto_accept_receipt(exchange)
    intent = submission_intent(
        goal=material_claim,
        claims=(submission_claim("claim-auto", material_claim),),
        transcript=(exchange,),
    )

    result = evaluate(
        intent,
        intent_truth(scorable=False),
        script=clarification_script(),
        receipts=(receipt,),
    )

    exchange_result = result.clarification.exchanges[0]
    assert exchange_result.material is True
    assert exchange_result.answer_consumed is None
    assert exchange_result.update_applied is True
    assert IntentReasonCode.CLARIFICATION_ANSWER_NOT_CONSUMED not in (
        exchange_result.reason_codes
    )

    with pytest.raises(IntentEvaluationError, match="requires its Harness receipt"):
        evaluate(
            intent,
            intent_truth(scorable=False),
            script=clarification_script(),
        )


def test_forged_benchmark_auto_accept_receipt_requires_private_case_authority() -> None:
    material_claim = "Support dry-run mode"
    exchange = clarification_exchange(
        material_claim=material_claim,
        matched_answer_id=None,
        action=ClarificationAction.CONFIRM,
        response=None,
        resolved_values=(material_claim,),
    )
    receipt = benchmark_auto_accept_receipt(exchange)
    intent = submission_intent(goal=material_claim, transcript=(exchange,))

    with pytest.raises(IntentEvaluationError, match="unauthorized"):
        evaluate(
            intent,
            intent_truth(policy=ClarificationPolicy.REQUIRED),
            script=clarification_script(),
            receipts=(receipt,),
        )

    with pytest.raises(IntentEvaluationError, match="unauthorized"):
        evaluate(
            intent,
            intent_truth(scorable=False),
            script=clarification_script(clarification_answer()),
            receipts=(receipt,),
        )


def test_extra_and_duplicate_receipts_are_rejected() -> None:
    answer = clarification_answer()
    script = clarification_script(answer)
    exchange = clarification_exchange()
    receipt = match_receipt(exchange, script)
    intent = submission_intent(goal=exchange.material_claim, transcript=(exchange,))
    truth = intent_truth(policy=ClarificationPolicy.REQUIRED)

    with pytest.raises(IntentEvaluationError, match="duplicate clarification receipt"):
        evaluate(intent, truth, script=script, receipts=(receipt, receipt))

    extra = replace(receipt, question_id="question-extra")
    with pytest.raises(IntentEvaluationError, match="does not belong to transcript"):
        evaluate(intent, truth, script=script, receipts=(receipt, extra))


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda receipt: replace(receipt, actual_claim_digest="f" * 64),
            "material-claim hash mismatch",
        ),
        (
            lambda receipt: replace(
                receipt,
                candidates=(
                    replace(receipt.candidates[0], request_digest="f" * 64),
                ),
            ),
            "candidate hash mismatch",
        ),
        (
            lambda receipt: replace(
                receipt,
                dimension=IntentDimension.CONSTRAINT,
            ),
            "dimension does not match",
        ),
    ],
)
def test_tampered_hash_bound_receipt_is_rejected(tamper: object, message: str) -> None:
    answer = clarification_answer()
    script = clarification_script(answer)
    exchange = clarification_exchange()
    receipt = match_receipt(exchange, script)
    intent = submission_intent(goal=exchange.material_claim, transcript=(exchange,))

    with pytest.raises(IntentEvaluationError, match=message):
        evaluate(
            intent,
            intent_truth(policy=ClarificationPolicy.REQUIRED),
            script=script,
            receipts=(tamper(receipt),),  # type: ignore[operator]
        )


@pytest.mark.parametrize(
    ("exchange_dimension", "exchange_claim", "reason"),
    [
        (
            IntentDimension.SCOPE,
            "No network access",
            IntentReasonCode.CLARIFICATION_WRONG_DIMENSION,
        ),
        (
            IntentDimension.CONSTRAINT,
            "Network access is permitted",
            IntentReasonCode.CLARIFICATION_WRONG_MATERIAL_CLAIM,
        ),
    ],
)
def test_valid_unmatched_receipt_preserves_wrong_question_diagnostic(
    exchange_dimension: IntentDimension,
    exchange_claim: str,
    reason: IntentReasonCode,
) -> None:
    answer = clarification_answer(
        dimension=IntentDimension.CONSTRAINT,
        material_claim="No network access",
        action=ClarificationAction.REJECT,
        response="Network access is allowed",
    )
    script = clarification_script(answer)
    exchange = clarification_exchange(
        dimension=exchange_dimension,
        material_claim=exchange_claim,
        matched_answer_id=None,
        action=None,
    )
    receipt = match_receipt(exchange, script, equivalent_answer_ids=set())
    intent = submission_intent(transcript=(exchange,))

    result = evaluate(
        intent,
        intent_truth(policy=ClarificationPolicy.REQUIRED),
        script=script,
        receipts=(receipt,),
    )

    exchange_result = result.clarification.exchanges[0]
    assert exchange_result.material is False
    assert exchange_result.answer_consumed is False
    assert reason in exchange_result.reason_codes
    assert IntentReasonCode.CLARIFICATION_ANSWER_NOT_CONSUMED in (
        exchange_result.reason_codes
    )


def decided_semantic_fixture() -> tuple[
    SubmissionIntent,
    IntentTruth,
    IntentSemanticJudgeDecision,
    IntentEvaluationResult,
]:
    intent, truth, pending = semantic_fixture()
    decision = judge_decision(
        pending.judge_requests[0],
        IntentJudgeRelation.EQUIVALENT,
    )
    return intent, truth, decision, evaluate(
        intent,
        truth,
        decisions=(decision,),
    )


def decided_semantic_result() -> IntentEvaluationResult:
    return decided_semantic_fixture()[3]


def test_canonical_serialization_and_strict_round_trip_hydration() -> None:
    intent, truth, decision, result = decided_semantic_fixture()
    payload = result.to_dict()
    encoded = result.to_json()

    assert result.schema_version == INTENT_EVALUATION_SCHEMA_VERSION
    assert result.normalization_version == INTENT_NORMALIZATION_POLICY_VERSION
    assert result.submission_intent_digest == canonical_sha256(intent.to_dict())
    assert result.intent_truth_digest == canonical_sha256(truth.to_dict())
    assert result.clarification_script_digest == canonical_sha256(
        clarification_script().to_dict()
    )
    assert encoded == canonical_json(payload)
    assert "\n" not in encoded
    assert "\r" not in encoded
    assert hydrate_dict(
        deepcopy(payload), intent, truth, decisions=(decision,)
    ) == result
    assert hydrate_json(encoded, intent, truth, decisions=(decision,)) == result
    assert hydrate_json(
        encoded.encode("utf-8"), intent, truth, decisions=(decision,)
    ) == result
    assert hydrate_json(
        encoded, intent, truth, decisions=(decision,)
    ).digest() == result.digest()


def test_bound_hydration_rejects_persisted_metrics_claims_and_digest_mutations() -> None:
    intent, truth, decision, result = decided_semantic_fixture()

    tampered_metrics = deepcopy(result.to_dict())
    tampered_metrics["metrics"]["supported_claim_count"] = 0
    with pytest.raises(IntentEvaluationError, match="metric|replay"):
        hydrate_dict(tampered_metrics, intent, truth, decisions=(decision,))

    tampered_claim = deepcopy(result.to_dict())
    tampered_claim["truth_claims"][0]["required"] = False
    tampered_claim["metrics"].update(
        {
            "required_truth_count": 0,
            "required_supported_count": 0,
            "required_missed_count": 0,
            "optional_truth_count": 1,
            "optional_supported_count": 1,
        }
    )
    with pytest.raises(IntentEvaluationError, match="deterministic replay"):
        hydrate_dict(tampered_claim, intent, truth, decisions=(decision,))

    for digest_field in (
        "submission_intent_digest",
        "intent_truth_digest",
        "clarification_script_digest",
    ):
        tampered_digest = deepcopy(result.to_dict())
        tampered_digest[digest_field] = "0" * 64
        with pytest.raises(IntentEvaluationError, match="deterministic replay"):
            hydrate_dict(tampered_digest, intent, truth, decisions=(decision,))


def test_bound_hydration_rejects_swapped_truth_script_and_judge_result() -> None:
    intent, truth, decision, result = decided_semantic_fixture()

    swapped_truth = intent_truth(
        expected_claim("truth-goal", "Support dry-run mode", required=False)
    )
    with pytest.raises(IntentEvaluationError, match="deterministic replay"):
        hydrate_dict(
            result.to_dict(),
            intent,
            swapped_truth,
            decisions=(decision,),
        )

    swapped_script = clarification_script(max_rounds=3)
    with pytest.raises(IntentEvaluationError, match="deterministic replay"):
        hydrate_dict(
            result.to_dict(),
            intent,
            truth,
            script=swapped_script,
            decisions=(decision,),
        )

    swapped_decision = judge_decision(
        semantic_fixture()[2].judge_requests[0],
        IntentJudgeRelation.DIFFERENT,
    )
    with pytest.raises(IntentEvaluationError, match="deterministic replay"):
        hydrate_dict(
            result.to_dict(),
            intent,
            truth,
            decisions=(swapped_decision,),
        )


def test_bound_hydration_rejects_a_swapped_clarification_match_receipt() -> None:
    answer = clarification_answer()
    script = clarification_script(answer)
    exchange = clarification_exchange()
    receipt = match_receipt(exchange, script)
    intent = submission_intent(
        goal=exchange.material_claim,
        transcript=(exchange,),
    )
    truth = intent_truth(
        expected_claim("truth-goal", exchange.material_claim),
        policy=ClarificationPolicy.REQUIRED,
    )
    result = evaluate(
        intent,
        truth,
        script=script,
        receipts=(receipt,),
    )
    swapped_receipt = match_receipt(
        exchange,
        script,
        matcher_digest="b" * 64,
    )

    with pytest.raises(IntentEvaluationError, match="deterministic replay"):
        hydrate_dict(
            result.to_dict(),
            intent,
            truth,
            script=script,
            receipts=(swapped_receipt,),
        )


def test_bound_hydration_requires_the_real_intent_evaluator() -> None:
    class DerivedIntentEvaluator(IntentEvaluator):
        pass

    intent, truth, decision, result = decided_semantic_fixture()
    derived = DerivedIntentEvaluator(evaluator_revision=EVALUATOR_REVISION)

    with pytest.raises(IntentEvaluationError, match="real IntentEvaluator"):
        hydrate_dict(
            result.to_dict(),
            intent,
            truth,
            decisions=(decision,),
            evaluator=derived,
        )


def test_public_hydrators_require_all_source_bindings() -> None:
    result = decided_semantic_result()

    with pytest.raises(TypeError, match="required keyword-only"):
        IntentEvaluationResult.from_dict(result.to_dict())  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="required keyword-only"):
        IntentEvaluationResult.from_json(result.to_json())  # type: ignore[call-arg]


def test_hydration_rejects_tampered_judge_ungraded_resolution_and_provenance() -> None:
    intent, truth, pending = semantic_fixture()
    receipt = judge_ungraded(pending.judge_requests[0])
    result = evaluate(
        intent,
        truth,
        ungraded=(receipt,),
    )

    tampered_reason = deepcopy(result.to_dict())
    tampered_reason["judge_ungraded"][0]["ungraded_reason"] = "future_ungraded"
    with pytest.raises(IntentEvaluationError, match="unsupported ungraded_reason"):
        hydrate_dict(tampered_reason, intent, truth, ungraded=(receipt,))

    tampered_execution = deepcopy(result.to_dict())
    tampered_execution["judge_ungraded"][0]["evaluator_execution_digest"] = "f" * 63
    with pytest.raises(IntentEvaluationError, match="SHA-256"):
        hydrate_dict(tampered_execution, intent, truth, ungraded=(receipt,))

    tampered_result = deepcopy(result.to_dict())
    tampered_result["judge_ungraded"][0]["judge_result_digest"] = "f" * 63
    with pytest.raises(IntentEvaluationError, match="SHA-256"):
        hydrate_dict(tampered_result, intent, truth, ungraded=(receipt,))

    tampered_request = deepcopy(result.to_dict())
    tampered_request["judge_ungraded"][0]["request_id"] = "another-request"
    with pytest.raises(IntentEvaluationError, match="unknown request"):
        hydrate_dict(tampered_request, intent, truth, ungraded=(receipt,))

    tampered_candidate = deepcopy(result.to_dict())
    tampered_candidate["candidates"][0]["reason_codes"] = [
        IntentReasonCode.JUDGE_PENDING.value
    ]
    with pytest.raises(IntentEvaluationError, match="candidate reason codes"):
        hydrate_dict(tampered_candidate, intent, truth, ungraded=(receipt,))

    tampered_outcome = deepcopy(result.to_dict())
    tampered_outcome["claim_outcomes"][0]["reason_codes"] = [
        IntentReasonCode.JUDGE_PENDING.value
    ]
    with pytest.raises(IntentEvaluationError, match="claim outcome"):
        hydrate_dict(tampered_outcome, intent, truth, ungraded=(receipt,))

    tampered_status = deepcopy(result.to_dict())
    tampered_status["status"] = IntentEvaluationStatus.PENDING_JUDGE.value
    with pytest.raises(IntentEvaluationError, match="pending Judge work|not canonical"):
        hydrate_dict(tampered_status, intent, truth, ungraded=(receipt,))

    missing_receipt = deepcopy(result.to_dict())
    missing_receipt["judge_ungraded"] = []
    with pytest.raises(IntentEvaluationError, match="candidate reason codes"):
        hydrate_dict(missing_receipt, intent, truth, ungraded=(receipt,))


def test_hydration_rejects_tampered_judge_failure_provenance_and_overlap() -> None:
    intent, truth, pending = semantic_fixture()
    request = pending.judge_requests[0]
    failure = judge_failure(request)
    result = evaluate(intent, truth, failures=(failure,))

    unknown_code = deepcopy(result.to_dict())
    unknown_code["judge_failures"][0]["failure_code"] = "future_failure"
    with pytest.raises(IntentEvaluationError, match="unsupported failure_code"):
        hydrate_dict(unknown_code, intent, truth, failures=(failure,))

    crossed = deepcopy(result.to_dict())
    crossed["judge_failures"][0]["request_id"] = "another-request"
    with pytest.raises(IntentEvaluationError, match="unknown request"):
        hydrate_dict(crossed, intent, truth, failures=(failure,))

    overlap = deepcopy(result.to_dict())
    overlap["judge_ungraded"] = [
        judge_ungraded(request).to_dict()
    ]
    with pytest.raises(IntentEvaluationError, match="more than one resolution"):
        hydrate_dict(overlap, intent, truth, failures=(failure,))


def test_from_dict_rejects_unknown_missing_and_tampered_fields() -> None:
    intent, truth, decision, result = decided_semantic_fixture()

    unknown = deepcopy(result.to_dict())
    unknown["future_field"] = True
    with pytest.raises(IntentEvaluationError, match="unknown or missing fields"):
        hydrate_dict(unknown, intent, truth, decisions=(decision,))

    missing = deepcopy(result.to_dict())
    missing.pop("metrics")
    with pytest.raises(IntentEvaluationError, match="unknown or missing fields"):
        hydrate_dict(missing, intent, truth, decisions=(decision,))

    nested_unknown = deepcopy(result.to_dict())
    nested_unknown["generated_claims"][0]["future_field"] = True
    with pytest.raises(IntentEvaluationError, match="unknown or missing fields"):
        hydrate_dict(nested_unknown, intent, truth, decisions=(decision,))

    tampered_metric = deepcopy(result.to_dict())
    tampered_metric["metrics"]["supported_claim_count"] = 99
    with pytest.raises(IntentEvaluationError, match="metric claim counts are inconsistent"):
        hydrate_dict(tampered_metric, intent, truth, decisions=(decision,))

    tampered_normalized_text = deepcopy(result.to_dict())
    tampered_normalized_text["generated_claims"][0]["normalized_text"] = "tampered"
    with pytest.raises(IntentEvaluationError, match="does not match text"):
        hydrate_dict(
            tampered_normalized_text, intent, truth, decisions=(decision,)
        )

    tampered_case_pass = deepcopy(result.to_dict())
    tampered_case_pass["metrics"]["intent_case_pass"] = False
    with pytest.raises(IntentEvaluationError, match="case pass is inconsistent"):
        hydrate_dict(tampered_case_pass, intent, truth, decisions=(decision,))

    tampered_clarification = deepcopy(result.to_dict())
    tampered_clarification["clarification"]["decision_correct"] = False
    with pytest.raises(IntentEvaluationError, match="does not match its policy"):
        hydrate_dict(
            tampered_clarification, intent, truth, decisions=(decision,)
        )

    tampered_binding = deepcopy(result.to_dict())
    tampered_binding["submission_intent_digest"] = "f" * 63
    with pytest.raises(IntentEvaluationError, match="SHA-256"):
        hydrate_dict(tampered_binding, intent, truth, decisions=(decision,))

    tampered_candidate_reason = deepcopy(result.to_dict())
    tampered_candidate_reason["candidates"][0]["reason_codes"] = [
        IntentReasonCode.JUDGE_DIFFERENT.value
    ]
    with pytest.raises(IntentEvaluationError, match="candidate reason codes"):
        hydrate_dict(
            tampered_candidate_reason, intent, truth, decisions=(decision,)
        )

    tampered_outcome_reason = deepcopy(result.to_dict())
    tampered_outcome_reason["claim_outcomes"][0]["reason_codes"] = [
        IntentReasonCode.JUDGE_DIFFERENT.value
    ]
    with pytest.raises(IntentEvaluationError, match="claim outcome"):
        hydrate_dict(
            tampered_outcome_reason, intent, truth, decisions=(decision,)
        )

    tampered_status = deepcopy(result.to_dict())
    tampered_status["status"] = IntentEvaluationStatus.UNGRADED.value
    tampered_status["metrics"]["intent_case_pass"] = None
    with pytest.raises(IntentEvaluationError, match="status is not canonical"):
        hydrate_dict(tampered_status, intent, truth, decisions=(decision,))


def test_hydration_recomputes_global_assignment_and_canonical_tie_break() -> None:
    intent = submission_intent(goal="Exact text")
    truth = intent_truth(
        expected_claim("truth-exact", "Exact text"),
        expected_claim("truth-normalized", " exact TEXT "),
    )
    result = evaluate(intent, truth)
    payload = result.to_dict()
    exact = next(
        item for item in payload["candidates"] if item["match_kind"] == "exact"
    )
    normalized = next(
        item
        for item in payload["candidates"]
        if item["match_kind"] == "normalized"
    )
    exact["selected"] = False
    normalized["selected"] = True
    payload["assignments"] = [
        {
            "left_id": normalized["generated_id"],
            "right_id": normalized["truth_id"],
            "weight": normalized["edge_weight"],
        }
    ]
    outcome = payload["claim_outcomes"][0]
    outcome["matched_truth_id"] = normalized["truth_id"]
    outcome["match_kind"] = "normalized"
    outcome["reason_codes"] = [
        IntentReasonCode.DETERMINISTIC_NORMALIZED.value,
        IntentReasonCode.MATCHED_EXPECTED.value,
    ]
    payload["unmatched_expected_truth_ids"] = [exact["truth_id"]]

    with pytest.raises(IntentEvaluationError, match="global maximum-weight"):
        hydrate_dict(payload, intent, truth)


def test_hydration_rejects_a_truncated_deterministic_candidate_graph() -> None:
    intent = submission_intent(goal="Exact text")
    truth = intent_truth(
        expected_claim("truth-exact", "Exact text"),
        expected_claim("truth-normalized", " exact TEXT "),
    )
    result = evaluate(intent, truth)
    payload = result.to_dict()
    payload["candidates"] = [
        item for item in payload["candidates"] if item["match_kind"] != "normalized"
    ]

    with pytest.raises(IntentEvaluationError, match="same-dimension pair"):
        hydrate_dict(payload, intent, truth)


def test_hydration_rejects_truncated_pending_semantic_work() -> None:
    intent, truth, result = semantic_fixture()
    payload = result.to_dict()
    payload["candidates"] = []
    payload["judge_requests"] = []

    with pytest.raises(IntentEvaluationError, match="same-dimension pair"):
        hydrate_dict(payload, intent, truth)


def test_hydration_rejects_a_forged_candidate_limit_outcome() -> None:
    intent = submission_intent(goal="same")
    truth = intent_truth(expected_claim("truth-goal", "same"))
    result = evaluate(intent, truth)
    payload = result.to_dict()
    generated_id = payload["generated_claims"][0]["generated_id"]
    payload["status"] = IntentEvaluationStatus.UNGRADED.value
    payload["candidates"] = []
    payload["assignments"] = []
    payload["claim_outcomes"] = [
        {
            "generated_id": generated_id,
            "judgement": IntentClaimJudgement.UNKNOWN.value,
            "matched_truth_id": None,
            "matched_truth_kind": None,
            "match_kind": None,
            "reason_codes": [
                IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED.value
            ],
        }
    ]
    payload["unmatched_generated_ids"] = [generated_id]
    payload["unmatched_expected_truth_ids"] = ["truth-goal"]
    payload["judge_requests"] = []
    payload["judge_decisions"] = []
    payload["metrics"] = IntentEvaluator._metrics_unscorable().to_dict()
    payload["reason_codes"] = [
        IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED.value
    ]

    with pytest.raises(IntentEvaluationError, match="candidate-limit outcome"):
        hydrate_dict(payload, intent, truth)


def test_hydration_rejects_a_forged_not_scorable_truth_outcome() -> None:
    intent = submission_intent(goal="same")
    truth = intent_truth(expected_claim("truth-goal", "same"))
    result = evaluate(intent, truth)
    payload = result.to_dict()
    generated_id = payload["generated_claims"][0]["generated_id"]
    payload["status"] = IntentEvaluationStatus.NOT_SCORABLE.value
    payload["candidates"] = []
    payload["assignments"] = []
    payload["claim_outcomes"] = [
        {
            "generated_id": generated_id,
            "judgement": IntentClaimJudgement.UNKNOWN.value,
            "matched_truth_id": None,
            "matched_truth_kind": None,
            "match_kind": None,
            "reason_codes": [IntentReasonCode.INTENT_TRUTH_UNSCORABLE.value],
        }
    ]
    payload["unmatched_generated_ids"] = [generated_id]
    payload["unmatched_expected_truth_ids"] = ["truth-goal"]
    payload["judge_requests"] = []
    payload["judge_decisions"] = []
    payload["clarification"] = {
        "policy": None,
        "decision_correct": None,
        "complete": None,
        "exchanges": [],
        "reason_codes": [IntentReasonCode.INTENT_TRUTH_UNSCORABLE.value],
    }
    payload["metrics"] = IntentEvaluator._metrics_unscorable().to_dict()
    payload["reason_codes"] = [IntentReasonCode.INTENT_TRUTH_UNSCORABLE.value]

    with pytest.raises(IntentEvaluationError, match="empty truth claims"):
        hydrate_dict(payload, intent, truth)


def test_hydration_rejects_not_scorable_with_deleted_claims_and_old_digest() -> None:
    intent = submission_intent(goal="same")
    truth = intent_truth(expected_claim("truth-goal", "same"))
    result = evaluate(intent, truth)
    payload = result.to_dict()
    generated_id = payload["generated_claims"][0]["generated_id"]
    payload["status"] = IntentEvaluationStatus.NOT_SCORABLE.value
    payload["truth_claims"] = []
    payload["candidates"] = []
    payload["assignments"] = []
    payload["claim_outcomes"] = [
        {
            "generated_id": generated_id,
            "judgement": IntentClaimJudgement.UNKNOWN.value,
            "matched_truth_id": None,
            "matched_truth_kind": None,
            "match_kind": None,
            "reason_codes": [IntentReasonCode.INTENT_TRUTH_UNSCORABLE.value],
        }
    ]
    payload["unmatched_generated_ids"] = [generated_id]
    payload["unmatched_expected_truth_ids"] = []
    payload["judge_requests"] = []
    payload["judge_decisions"] = []
    payload["clarification"] = {
        "policy": None,
        "decision_correct": None,
        "complete": None,
        "exchanges": [],
        "reason_codes": [IntentReasonCode.INTENT_TRUTH_UNSCORABLE.value],
    }
    payload["metrics"] = IntentEvaluator._metrics_unscorable().to_dict()
    payload["reason_codes"] = [IntentReasonCode.INTENT_TRUTH_UNSCORABLE.value]

    with pytest.raises(IntentEvaluationError, match="canonical unscorable truth"):
        hydrate_dict(payload, intent, truth)


@pytest.mark.parametrize(
    ("budget_name", "error"),
    [
        ("MAX_INTENT_CANDIDATE_RECORD_BYTES", "candidate records exceed"),
        ("MAX_INTENT_JUDGE_REQUEST_BYTES", "Judge requests exceed"),
        ("MAX_INTENT_JUDGE_DECISION_BYTES", "Judge decisions exceed"),
    ],
)
def test_hydration_revalidates_case_wide_record_byte_budgets(
    monkeypatch: pytest.MonkeyPatch,
    budget_name: str,
    error: str,
) -> None:
    intent, truth, decision, result = decided_semantic_fixture()
    monkeypatch.setattr(intent_evaluator_module, budget_name, 1)

    with pytest.raises(IntentEvaluationError, match=error):
        hydrate_dict(result.to_dict(), intent, truth, decisions=(decision,))


def test_from_json_rejects_duplicate_keys_invalid_json_and_non_text() -> None:
    intent, truth, decision, _result = decided_semantic_fixture()
    with pytest.raises(ValueError, match="duplicate JSON key"):
        hydrate_json(
            '{"schema_version":"one","schema_version":"two"}',
            intent,
            truth,
            decisions=(decision,),
        )
    with pytest.raises(ValueError, match="not valid strict JSON"):
        hydrate_json("{", intent, truth, decisions=(decision,))
    with pytest.raises(ValueError, match="only bytes or text"):
        hydrate_json({}, intent, truth, decisions=(decision,))


def test_evaluation_result_and_nested_records_are_frozen() -> None:
    result = decided_semantic_result()

    with pytest.raises(FrozenInstanceError):
        result.status = IntentEvaluationStatus.UNGRADED  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.generated_claims[0].text = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.judge_decisions[0].score_ppm = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.clarification.complete = False  # type: ignore[misc]
    assert isinstance(result.generated_claims, tuple)
    assert isinstance(result.candidates, tuple)
    assert isinstance(result.assignments, tuple)
    assert isinstance(result.reason_codes, tuple)


def test_generated_claim_hydration_requires_canonical_derived_normalization() -> None:
    claim = GeneratedIntentClaim(
        generated_id="generated-1",
        dimension=IntentDimension.GOAL,
        text="  CAFÉ\tMode ",
        normalized_text="café mode",
        source=None,
        provenance_claim_id=None,
        origin=IntentClaimOrigin.STRUCTURED,
    )

    assert GeneratedIntentClaim.from_dict(claim.to_dict()) == claim
    payload = claim.to_dict()
    payload["normalized_text"] = "cafe mode"
    with pytest.raises(IntentEvaluationError, match="does not match text"):
        GeneratedIntentClaim.from_dict(payload)


def test_deterministic_duplicate_matrix_does_not_consume_semantic_budget() -> None:
    """The 65,536 cap is for unresolved Judge pairs, not exact duplicates."""

    generated = tuple(
        submission_claim(
            f"claim-{index:03d}",
            "same exact claim",
            dimension=IntentDimension.SCOPE,
        )
        for index in range(257)
    )
    truth = intent_truth(
        *(
            expected_claim(
                f"truth-{index:03d}",
                "same exact claim",
                dimension=IntentDimension.SCOPE,
            )
            for index in range(256)
        )
    )

    result = evaluate(
        submission_intent(claims=generated),
        truth,
    )

    assert result.status is IntentEvaluationStatus.GRADED
    assert result.judge_requests == ()
    assert len(result.candidates) == 257 * 256
    assert result.metrics.unknown_claim_count == 0
    assert result.metrics.supported_claim_count == 256
    assert result.metrics.unsupported_claim_count == 1

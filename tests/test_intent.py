import pytest

from review_agent.git_repo import ChangeSummary
from review_agent.intent import (
    apply_user_decision,
    build_intent_packet,
    collect_deterministic_claims,
    compute_intent_status,
    finalize_intent_packet,
    generate_material_questions,
    merge_inference_claims,
)
from review_agent.models import (
    ClarificationQuestion,
    ClarificationStatus,
    ConclusionImpact,
    IntentClaim,
    IntentClaimState,
    IntentConfidence,
    IntentDecision,
    IntentDecisionAction,
    IntentField,
    IntentOrigin,
    IntentSource,
    IntentStatus,
    ReviewRequest,
)


def test_user_intent_is_explicit_and_focus_is_not_intent():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        user_intent="Add idempotency to payment callback",
        review_focus="duplicate execution and retry safety",
    )
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["payments/callback.py"], "", [])

    packet = build_intent_packet(request, summary)

    assert packet.goal == "Add idempotency to payment callback"
    assert packet.sources["goal"] is IntentSource.EXPLICIT
    assert "review_focus" not in packet.sources
    assert "duplicate execution and retry safety" not in packet.acceptance_criteria
    assert "acceptance criteria are not explicitly declared" in packet.uncertainties


def test_missing_user_intent_creates_inferred_goal():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["auth/session.py"], "", ["+def validate_session(token):"])

    packet = build_intent_packet(request, summary)

    assert packet.goal == "Review changes touching auth/session.py"
    assert packet.sources["goal"] is IntentSource.INFERRED
    assert "user did not provide explicit intent" in packet.uncertainties
    assert packet.status is IntentStatus.PARTIAL


def test_empty_change_set_is_insufficient():
    request = ReviewRequest(repository_path="C:/repo", base_revision="main", head_revision="HEAD")
    summary = ChangeSummary("C:/repo", "main", "HEAD", [], "", [])

    packet = build_intent_packet(request, summary)

    assert packet.goal is None
    assert packet.status is IntentStatus.INSUFFICIENT
    assert "no changed files were detected" in packet.uncertainties


def test_project_rules_are_explicit_constraints():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        project_rules=("preserve public API compatibility",),
    )
    summary = ChangeSummary("C:/repo", "main", "HEAD", ["api.py"], "", [])

    packet = build_intent_packet(request, summary)

    assert packet.constraints == ["preserve public API compatibility"]
    assert packet.sources["constraints"] is IntentSource.EXPLICIT


def test_claim_question_and_decision_ids_are_stable_and_models_validate_input():
    first = _claim(IntentField.GOAL, "Ship retry safety")
    second = IntentClaim(
        field=IntentField.GOAL,
        value="SHIP RETRY SAFETY",
        source=IntentSource.INFERRED,
        origin=IntentOrigin.CHANGED_FILES,
        confidence=IntentConfidence.LOW,
        source_refs=["diff:worker.py"],
    )
    assert first.claim_id == second.claim_id

    question_a = ClarificationQuestion(
        field=IntentField.GOAL,
        question="Is this the intended goal?",
        rationale="The answer changes behavioral correctness conclusions.",
        proposed_values=[first.value],
        claim_ids=[first.claim_id],
    )
    question_b = ClarificationQuestion(
        field=IntentField.GOAL,
        question="Confirm the goal.",
        rationale="This affects the conclusion.",
        proposed_values=[first.value],
        claim_ids=[first.claim_id],
    )
    assert question_a.question_id == question_b.question_id

    decision_a = IntentDecision(
        question_id=question_a.question_id,
        action=IntentDecisionAction.SKIPPED,
        continuation_basis="Continue and disclose the uncertainty.",
    )
    decision_b = IntentDecision(
        question_id=question_a.question_id,
        action=IntentDecisionAction.SKIPPED,
        continuation_basis="Continue and disclose the uncertainty.",
    )
    assert decision_a.decision_id == decision_b.decision_id

    with pytest.raises(ValueError, match="IntentField"):
        IntentClaim(  # type: ignore[arg-type]
            field="goal",
            value="Ship retry safety",
            source=IntentSource.INFERRED,
            origin=IntentOrigin.LLM_INFERENCE,
            confidence=IntentConfidence.MEDIUM,
        )
    with pytest.raises(ValueError, match="corrected decision"):
        IntentDecision(
            question_id=question_a.question_id,
            action=IntentDecisionAction.CORRECTED,
            corrected_values=["Use the corrected goal"],
        )


def test_deterministic_collection_tracks_request_and_baseline_provenance_without_focus():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        user_intent="Add bounded retries",
        description="A failed job is retried at most three times",
        linked_requirements=("Existing successful jobs remain unchanged",),
        review_focus="logging style",
        project_rules=("Preserve the public job payload",),
    )
    summary = ChangeSummary(
        "C:/repo", "main", "HEAD", ["jobs/retry.py", "jobs/test_retry.py"], "", []
    )

    claims = collect_deterministic_claims(request, summary)

    assert [(claim.field, claim.source) for claim in claims] == [
        (IntentField.GOAL, IntentSource.EXPLICIT),
        (IntentField.ACCEPTANCE_CRITERIA, IntentSource.EXPLICIT),
        (IntentField.ACCEPTANCE_CRITERIA, IntentSource.EXPLICIT),
        (IntentField.SCOPE, IntentSource.INFERRED),
        (IntentField.SCOPE, IntentSource.INFERRED),
        (IntentField.CONSTRAINTS, IntentSource.EXPLICIT),
    ]
    assert all("logging style" not in claim.value for claim in claims)
    assert all(
        "review_focus" not in source_ref
        for claim in claims
        for source_ref in claim.source_refs
    )


def test_merge_deduplicates_equivalent_values_and_conflicting_goals_are_insufficient():
    explicit_goal = _claim(
        IntentField.GOAL,
        "Preserve retry safety",
        source=IntentSource.EXPLICIT,
        origin=IntentOrigin.REQUEST_METADATA,
    )
    duplicate = _claim(IntentField.GOAL, "PRESERVE RETRY SAFETY")
    conflicting = _claim(IntentField.GOAL, "Remove all retries")

    merged = merge_inference_claims([explicit_goal], [duplicate, conflicting])
    goals = [claim for claim in merged if claim.field is IntentField.GOAL]
    packet = finalize_intent_packet(merged)

    assert len(goals) == 2
    assert goals[0].source is IntentSource.EXPLICIT
    assert packet.status is IntentStatus.INSUFFICIENT
    assert any("conflicting goal claims" in item for item in packet.uncertainties)


def test_inference_source_validation_downgrades_spoofed_explicit_and_rejects_unauthorized_evidence():
    spoofed = _claim(
        IntentField.ACCEPTANCE_CRITERIA,
        "Retries stop after three attempts",
        source=IntentSource.EXPLICIT,
        origin=IntentOrigin.LLM_INFERENCE,
    )
    unauthorized = IntentClaim(
        field=IntentField.CONSTRAINTS,
        value="Do not change the wire format",
        source=IntentSource.INFERRED,
        origin=IntentOrigin.LLM_INFERENCE,
        confidence=IntentConfidence.MEDIUM,
        source_refs=["src/protocol.py"],
        evidence_refs=["O-not-authorized"],
    )

    merged = merge_inference_claims([], [spoofed, unauthorized])

    assert merged[0].source is IntentSource.INFERRED
    assert merged[1].claim_state is IntentClaimState.INVALID


def test_material_questions_ignore_non_material_optional_claims_and_have_stable_ids():
    claims = [
        _claim(
            IntentField.GOAL,
            "Add retry limits",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.USER_INPUT,
        ),
        _claim(IntentField.ACCEPTANCE_CRITERIA, "Stop after three attempts"),
        _claim(
            IntentField.SCOPE,
            "jobs/retry.py",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.REQUEST_METADATA,
        ),
        _claim(
            IntentField.CONSTRAINTS,
            "Prefer the existing helper name",
            impact=ConclusionImpact.SUPPLEMENTAL,
        ),
    ]

    first = generate_material_questions(claims)
    second = generate_material_questions(claims)

    assert [question.field for question in first] == [IntentField.ACCEPTANCE_CRITERIA]
    assert first[0].proposed_values == ["Stop after three attempts"]
    assert first[0].question_id == second[0].question_id
    assert first[0].status is ClarificationStatus.PENDING


def test_confirm_upgrades_only_the_inferred_values_in_a_mixed_list_field():
    explicit_criterion = _claim(
        IntentField.ACCEPTANCE_CRITERIA,
        "Existing successful jobs stay unchanged",
        source=IntentSource.EXPLICIT,
        origin=IntentOrigin.REQUEST_METADATA,
    )
    inferred_criterion = _claim(
        IntentField.ACCEPTANCE_CRITERIA,
        "Failed jobs stop after three retries",
    )
    claims = [
        _claim(
            IntentField.GOAL,
            "Bound job retries",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.USER_INPUT,
        ),
        explicit_criterion,
        inferred_criterion,
        _claim(
            IntentField.SCOPE,
            "jobs/retry.py",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.REQUEST_METADATA,
        ),
    ]
    questions = generate_material_questions(claims)
    before = finalize_intent_packet(claims, questions)

    updated_claims, updated_questions = apply_user_decision(
        claims,
        questions,
        IntentDecision(
            question_id=questions[0].question_id,
            action=IntentDecisionAction.CONFIRMED,
            user_response="Yes, use that retry limit.",
        ),
    )
    after = finalize_intent_packet(updated_claims, updated_questions)

    assert before.sources["acceptance_criteria"] is IntentSource.INFERRED
    assert after.sources["acceptance_criteria"] is IntentSource.EXPLICIT
    assert after.status is IntentStatus.SUFFICIENT
    confirmed = next(
        claim for claim in updated_claims if claim.claim_id == inferred_criterion.claim_id
    )
    assert confirmed.origin is IntentOrigin.USER_CONFIRMATION
    assert next(
        claim for claim in updated_claims if claim.claim_id == explicit_criterion.claim_id
    ).origin is IntentOrigin.REQUEST_METADATA
    assert updated_questions[0].status is ClarificationStatus.CONFIRMED
    assert updated_questions[0].user_response == "Yes, use that retry limit."


def test_confirm_rejects_multiple_conflicting_goal_candidates():
    claims = [
        _claim(
            IntentField.GOAL,
            "Preserve retries",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.USER_INPUT,
        ),
        _claim(IntentField.GOAL, "Remove retries"),
    ]
    question = generate_material_questions(claims)[0]

    with pytest.raises(ValueError, match="require correction or rejection"):
        apply_user_decision(
            claims,
            [question],
            IntentDecision(
                question_id=question.question_id,
                action=IntentDecisionAction.CONFIRMED,
            ),
        )


def test_correct_replaces_candidates_with_explicit_user_values():
    inferred_goal = _claim(IntentField.GOAL, "Remove retries")
    claims = [
        inferred_goal,
        _claim(
            IntentField.ACCEPTANCE_CRITERIA,
            "Retry no more than three times",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.REQUEST_METADATA,
        ),
        _claim(
            IntentField.SCOPE,
            "jobs/retry.py",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.REQUEST_METADATA,
        ),
    ]
    questions = generate_material_questions(claims)
    decision = IntentDecision(
        question_id=questions[0].question_id,
        action=IntentDecisionAction.CORRECTED,
        corrected_values=["Bound retries without removing them"],
        user_response="Keep retries, but cap them at three.",
    )

    updated_claims, updated_questions = apply_user_decision(claims, questions, decision)
    packet = finalize_intent_packet(updated_claims, updated_questions)

    assert packet.goal == "Bound retries without removing them"
    assert packet.sources["goal"] is IntentSource.EXPLICIT
    assert packet.status is IntentStatus.SUFFICIENT
    assert any(
        claim.claim_id == inferred_goal.claim_id
        and claim.claim_state is IntentClaimState.SUPERSEDED
        for claim in updated_claims
    )
    corrected = next(
        claim for claim in updated_claims if claim.value == packet.goal
    )
    assert corrected.origin is IntentOrigin.USER_CORRECTION
    assert updated_questions[0].status is ClarificationStatus.CORRECTED
    assert updated_questions[0].resolved_values == [packet.goal]


def test_reject_invalidates_the_candidate_and_can_make_intent_insufficient():
    inferred_goal = _claim(IntentField.GOAL, "Remove retries")
    claims = [
        inferred_goal,
        _claim(
            IntentField.ACCEPTANCE_CRITERIA,
            "Jobs complete",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.REQUEST_METADATA,
        ),
        _claim(
            IntentField.SCOPE,
            "jobs/retry.py",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.REQUEST_METADATA,
        ),
    ]
    questions = generate_material_questions(claims)

    updated_claims, updated_questions = apply_user_decision(
        claims,
        questions,
        IntentDecision(
            question_id=questions[0].question_id,
            action=IntentDecisionAction.REJECTED,
            user_response="That is not the intended behavior.",
        ),
    )
    packet = finalize_intent_packet(updated_claims, updated_questions)

    assert packet.goal is None
    assert packet.status is IntentStatus.INSUFFICIENT
    assert updated_claims[0].claim_state is IntentClaimState.REJECTED
    assert updated_questions[0].status is ClarificationStatus.REJECTED
    assert any("candidate was rejected" in item for item in packet.uncertainties)


@pytest.mark.parametrize(
    ("action", "expected_status"),
    [
        (IntentDecisionAction.SKIPPED, ClarificationStatus.SKIPPED),
        (
            IntentDecisionAction.SKIPPED_NON_INTERACTIVE,
            ClarificationStatus.SKIPPED_NON_INTERACTIVE,
        ),
    ],
)
def test_skip_keeps_inference_and_records_continuation_basis(action, expected_status):
    claims = [
        _claim(
            IntentField.GOAL,
            "Bound retries",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.USER_INPUT,
        ),
        _claim(IntentField.ACCEPTANCE_CRITERIA, "Stop after three retries"),
        _claim(
            IntentField.SCOPE,
            "jobs/retry.py",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.REQUEST_METADATA,
        ),
    ]
    questions = generate_material_questions(claims)
    basis = "Continue without blocking and disclose the unconfirmed criterion."

    updated_claims, updated_questions = apply_user_decision(
        claims,
        questions,
        IntentDecision(
            question_id=questions[0].question_id,
            action=action,
            continuation_basis=basis,
        ),
    )
    packet = finalize_intent_packet(updated_claims, updated_questions)

    criterion = next(
        claim
        for claim in updated_claims
        if claim.field is IntentField.ACCEPTANCE_CRITERIA
    )
    assert criterion.source is IntentSource.INFERRED
    assert criterion.claim_state is IntentClaimState.ACTIVE
    assert updated_questions[0].status is expected_status
    assert updated_questions[0].continuation_basis == basis
    assert packet.status is IntentStatus.PARTIAL
    assert any("clarification skipped" in item for item in packet.uncertainties)


def test_runtime_computes_sufficient_partial_and_insufficient_statuses():
    explicit = [
        _claim(
            IntentField.GOAL,
            "Bound retries",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.USER_INPUT,
        ),
        _claim(
            IntentField.ACCEPTANCE_CRITERIA,
            "Stop after three retries",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.REQUEST_METADATA,
        ),
        _claim(
            IntentField.SCOPE,
            "jobs/retry.py",
            source=IntentSource.EXPLICIT,
            origin=IntentOrigin.REQUEST_METADATA,
        ),
    ]
    partial = [*explicit[:1], _claim(IntentField.ACCEPTANCE_CRITERIA, "Stop after three"), *explicit[2:]]
    conflicting = [*explicit, _claim(IntentField.GOAL, "Remove retries")]

    assert compute_intent_status(explicit) is IntentStatus.SUFFICIENT
    assert compute_intent_status(partial) is IntentStatus.PARTIAL
    assert compute_intent_status(conflicting) is IntentStatus.INSUFFICIENT


def _claim(
    field: IntentField,
    value: str,
    *,
    source: IntentSource = IntentSource.INFERRED,
    origin: IntentOrigin = IntentOrigin.LLM_INFERENCE,
    impact: ConclusionImpact = ConclusionImpact.MATERIAL,
) -> IntentClaim:
    return IntentClaim(
        field=field,
        value=value,
        source=source,
        origin=origin,
        confidence=(
            IntentConfidence.HIGH
            if source is IntentSource.EXPLICIT
            else IntentConfidence.MEDIUM
        ),
        source_refs=["test:source"],
        conclusion_impact=impact,
    )

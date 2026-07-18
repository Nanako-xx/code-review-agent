from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import pytest

from tests.eval.test_evidence_checker import _build_harness
from tests.eval.test_evidence_checker import _file_evidence
from tests.eval.test_judge import _Factory, _execution
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent_eval.judge import (
    JudgeContextKind,
    JudgeContextTrust,
    repository_context,
    SemanticJudge,
)
from review_agent_eval.judge import JudgeTask
from review_agent_eval.models import (
    EVAL_SUBMISSION_SCHEMA_VERSION,
    DiffSide,
    EvalSubmission,
    EvidenceIntegrity,
    ExpectedFinding,
    FindingSeverity,
    IntentResult,
    IssueJudgement,
    KnownInvalidFinding,
    RequiredContextLevel,
    ReviewTruth,
    SubmissionFinding,
    SubmissionIntent,
    SubmissionReview,
    SubmissionStatus,
    SubmissionUsage,
    SubmissionEvidence,
    TruthCompleteness,
    NovelFindingPolicy,
)
from review_agent_eval.review_evaluator import (
    FindingDisposition,
    FindingResolution,
    ReviewEvaluationPhase,
    ReviewEvaluationResult,
    ReviewEvaluationStatus,
    ReviewEvaluator,
    ReviewContextBundle,
    ReviewFindingContextEntry,
    ReviewPairContextEntry,
    ReviewTruthKind,
)


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[object]:
    value = _build_harness(tmp_path, heavy=False)
    try:
        yield value
    finally:
        value.preparer.__exit__(None, None, None)


def _finding(
    finding_id: str,
    claim: str,
    *,
    evidence_refs: tuple[str, ...] = (),
) -> SubmissionFinding:
    return SubmissionFinding(
        finding_id=finding_id,
        claim=claim,
        severity=FindingSeverity.HIGH,
        path="src/app.py",
        side=DiffSide.RIGHT,
        from_line=1,
        to_line=1,
        evidence_refs=evidence_refs,
        suggested_action="handle the error",
    )


def _truth(truth_id: str, claim: str) -> ReviewTruth:
    return ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(
            ExpectedFinding(
                truth_id=truth_id,
                claim=claim,
                severity=FindingSeverity.HIGH,
                category="correctness",
                required=True,
                locations=(),
                evidence_anchors=(),
                required_context_level=RequiredContextLevel.DIFF,
                rationale="The changed code can lose an error condition.",
            ),
        ),
        known_invalid_findings=(),
    )


def _submission(
    harness: object,
    finding: SubmissionFinding,
    *,
    extra_findings: tuple[SubmissionFinding, ...] = (),
    evidence: tuple[SubmissionEvidence, ...] = (),
) -> EvalSubmission:
    return EvalSubmission(
        schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        task_id=harness.eval_input.task_id,
        agent_id="agent-test",
        trial_id="trial-review-evaluator",
        status=SubmissionStatus.COMPLETED,
        intent=SubmissionIntent(
            status=IntentResult.SUFFICIENT,
            goal="review the change",
            acceptance_criteria=(),
            scope=(),
            constraints=(),
            claims=(),
            clarification_questions=(),
            uncertainties=(),
        ),
        review=SubmissionReview(findings=(finding, *extra_findings), uncertainties=()),
        evidence=evidence,
        usage=SubmissionUsage(
            elapsed_seconds=0,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            tool_calls=None,
            cost_amount=None,
            cost_currency=None,
        ),
        trace_ref=None,
        failure=None,
    )


def _evaluator(harness: object) -> ReviewEvaluator:
    return ReviewEvaluator(
        eval_input=harness.eval_input,
        replay=harness.replay,
        trial_id="trial-review-evaluator",
        evaluator_execution=_execution(),
    )


def test_exact_expected_finding_is_assigned_without_a_model_call(harness: object) -> None:
    evaluator = _evaluator(harness)
    submission = _submission(
        harness,
        _finding("finding-exact", "The changed branch can return the wrong result."),
    )
    truth = _truth("truth-exact", "The changed branch can return the wrong result.")

    result = evaluator.evaluate(submission, truth)

    assert result.status is ReviewEvaluationStatus.GRADED
    assert result.phase is ReviewEvaluationPhase.COMPLETE
    assert [(item.finding_id, item.truth_id) for item in result.assignments] == [
        ("finding-exact", "truth-exact")
    ]
    assert result.judge_requests == ()
    outcome = result.finding_outcomes[0]
    assert outcome.disposition is FindingDisposition.MATCHED
    assert outcome.issue_resolution is FindingResolution.RESOLVED
    assert outcome.matched_expected_truth_id == "truth-exact"


def test_non_exact_expected_pair_emits_a_typed_pending_request(harness: object) -> None:
    evaluator = _evaluator(harness)
    submission = _submission(
        harness,
        _finding("finding-pending", "A different description of the same defect."),
    )
    truth = _truth("truth-pending", "The changed branch can return the wrong result.")

    result = evaluator.evaluate(submission, truth)

    assert result.status is ReviewEvaluationStatus.PENDING_JUDGE
    assert result.phase is ReviewEvaluationPhase.EXPECTED_ASSIGNMENT
    assert len(result.judge_requests) == 1
    assert result.judge_requests[0].task is JudgeTask.FINDING_EQUIVALENCE
    assert result.judge_requests[0].finding_id == "finding-pending"
    assert result.finding_outcomes[0].issue_resolution is FindingResolution.PENDING_JUDGE
    assert result.finding_outcomes[0].disposition is FindingDisposition.UNGRADED


def test_review_evaluation_round_trip_replays_against_bound_inputs(harness: object) -> None:
    evaluator = _evaluator(harness)
    submission = _submission(
        harness,
        _finding("finding-round-trip", "The changed branch can return the wrong result."),
    )
    truth = _truth("truth-round-trip", "The changed branch can return the wrong result.")
    result = evaluator.evaluate(submission, truth)

    hydrated = ReviewEvaluationResult.from_json(
        result.to_json(),
        submission=submission,
        review_truth=truth,
        evaluator=evaluator,
        judge_results=(),
    )

    assert hydrated == result


def _run_scripted_judge(request_record, execution, **fields):
    request = request_record.request
    output = {
        "schema_version": request.rubric.response_schema,
        "request_id": request.request_id,
        "reason_refs": ["item-a"],
        **fields,
    }
    return SemanticJudge(
        adapter_factory=_Factory(
            [
                ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=json.dumps(output),
                )
            ]
        ),
        evaluator_execution=execution,
    ).execute(request)


def test_global_assignment_marks_only_one_duplicate_finding(harness: object) -> None:
    evaluator = _evaluator(harness)
    claim = "The changed branch can return the wrong result."
    submission = _submission(
        harness,
        _finding("finding-a", claim),
        extra_findings=(_finding("finding-b", claim),),
    )
    result = evaluator.evaluate(submission, _truth("truth-one", claim))

    outcomes = {item.finding_id: item for item in result.finding_outcomes}
    assert sum(item.disposition is FindingDisposition.MATCHED for item in outcomes.values()) == 1
    duplicate = next(item for item in outcomes.values() if item.disposition is FindingDisposition.DUPLICATE)
    assert duplicate.duplicate_truth_id == "truth-one"
    assert duplicate.duplicate_of_finding_id in outcomes
    assert len(result.assignments) == 1


def test_known_invalid_has_precedence_over_expected_assignment(harness: object) -> None:
    evaluator = _evaluator(harness)
    invalid_claim = "The changed branch always returns None."
    expected_claim = "The changed branch returns the wrong result."
    submission = _submission(harness, _finding("finding-invalid", invalid_claim))
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(
            ExpectedFinding(
                truth_id="truth-expected",
                claim=expected_claim,
                severity=FindingSeverity.HIGH,
                category="correctness",
                required=True,
                locations=(),
                evidence_anchors=(),
                required_context_level=RequiredContextLevel.DIFF,
                rationale="expected",
            ),
        ),
        known_invalid_findings=(
            # Keep the claim distinct from expected so the Case itself is valid.
            KnownInvalidFinding(
                truth_id="truth-invalid",
                claim=invalid_claim,
                category="correctness",
                locations=(),
                rationale="known trap",
            ),
        ),
    )

    result = evaluator.evaluate(submission, truth)

    outcome = result.finding_outcomes[0]
    assert outcome.issue_judgement.value == "fabricated"
    assert outcome.disposition is FindingDisposition.KNOWN_INVALID
    assert outcome.matched_known_invalid_truth_id == "truth-invalid"
    assert result.assignments == ()


def test_partial_equivalence_never_creates_an_assignment_edge(harness: object) -> None:
    evaluator = _evaluator(harness)
    submission = _submission(harness, _finding("finding-partial", "A compound defect claim."))
    truth = _truth("truth-partial", "The changed branch can return the wrong result.")
    pending = evaluator.evaluate(submission, truth)
    assert len(pending.judge_requests) == 1
    typed = _run_scripted_judge(
        pending.judge_requests[0],
        evaluator.evaluator_execution,
        relation="partially_equivalent",
        score_ppm=900_000,
        severity_assessment="consistent",
        actionability="actionable",
    )

    result = evaluator.evaluate(submission, truth, judge_results=(typed,))

    assert result.assignments == ()
    assert result.finding_outcomes[0].disposition is FindingDisposition.NOVEL_DISALLOWED


def test_valid_evidence_support_is_independent_from_issue_matching(harness: object) -> None:
    evaluator = _evaluator(harness)
    finding = _finding(
        "finding-supported",
        "The changed branch can return the wrong result.",
        evidence_refs=("evidence-supported",),
    )
    evidence = _file_evidence(
        harness,
        evidence_id="evidence-supported",
        from_line=1,
        to_line=1,
        excerpt="alpha\n",
    )
    submission = _submission(harness, finding, evidence=(evidence,))
    truth = _truth("truth-supported", finding.claim)
    pending = evaluator.evaluate(submission, truth)
    assert len(pending.judge_requests) == 1
    assert pending.judge_requests[0].phase is ReviewEvaluationPhase.EVIDENCE_SUPPORT
    assert pending.finding_outcomes[0].issue_judgement.value == "confirmed"
    assert pending.finding_outcomes[0].evidence_integrity is EvidenceIntegrity.VALID

    support_result = _run_scripted_judge(
        pending.judge_requests[0],
        evaluator.evaluator_execution,
        support="supported",
    )
    result = evaluator.evaluate(submission, truth, judge_results=(support_result,))

    outcome = result.finding_outcomes[0]
    assert outcome.evidence_integrity is EvidenceIntegrity.VALID
    assert outcome.evidence_support.value == "supported"
    assert outcome.strict_publishable is True


def test_invalid_evidence_does_not_change_a_confirmed_issue_into_fabricated(
    harness: object,
) -> None:
    evaluator = _evaluator(harness)
    finding = _finding(
        "finding-invalid-evidence",
        "The changed branch can return the wrong result.",
        evidence_refs=("evidence-invalid",),
    )
    invalid_evidence = _file_evidence(
        harness,
        evidence_id="evidence-invalid",
        from_line=1,
        to_line=1,
        excerpt="not the repository excerpt\n",
    )
    result = evaluator.evaluate(
        _submission(harness, finding, evidence=(invalid_evidence,)),
        _truth("truth-invalid-evidence", finding.claim),
    )

    outcome = result.finding_outcomes[0]
    assert outcome.issue_judgement.value == "confirmed"
    assert outcome.evidence_integrity is EvidenceIntegrity.INVALID
    assert outcome.strict_publishable is False
    assert result.judge_requests == ()


def test_scoped_context_is_bound_to_the_finding_judge_request(harness: object) -> None:
    source = repository_context(
        source_id="scoped-code",
        kind=JudgeContextKind.CODE,
        content="return value",
        revision="head",
        path="src/app.py",
    )
    context = ReviewContextBundle.create(
        finding_entries=(ReviewFindingContextEntry.create("finding-context", (source,)),)
    )
    evaluator = ReviewEvaluator(
        eval_input=harness.eval_input,
        replay=harness.replay,
        trial_id="trial-review-evaluator",
        evaluator_execution=_execution(),
        context_bundle=context,
    )
    finding = _finding("finding-context", "A different description of the same defect.")
    result = evaluator.evaluate(_submission(harness, finding), _truth("truth-context", "The changed branch can return the wrong result."))

    request = result.judge_requests[0].request
    assert any(item.source_id == "scoped-code" for item in request.reference_bindings)


def test_hydration_rejects_a_deleted_candidate_row(harness: object) -> None:
    evaluator = _evaluator(harness)
    submission = _submission(harness, _finding("finding-delete", "The changed branch can return the wrong result."))
    truth = _truth("truth-delete", "The changed branch can return the wrong result.")
    result = evaluator.evaluate(submission, truth)
    forged = result.to_dict()
    forged["expected_candidates"] = []

    with pytest.raises(Exception, match="candidate|replay|persisted"):
        ReviewEvaluationResult.from_dict(
            forged,
            submission=submission,
            review_truth=truth,
            evaluator=evaluator,
            judge_results=(),
        )


def test_direct_result_rejects_assignment_outcome_and_unmatched_tampering(
    harness: object,
) -> None:
    evaluator = _evaluator(harness)
    finding = _finding(
        "finding-direct-integrity",
        "The changed branch can return the wrong result.",
    )
    truth = _truth("truth-direct-integrity", finding.claim)
    submission = _submission(harness, finding)
    result = evaluator.evaluate(submission, truth)

    with pytest.raises(TypeError, match="ReviewEvaluator.evaluate|source-bound"):
        replace(result, submission_digest="0" * 64)

    forged = result.to_dict()
    forged["unmatched_expected_truth_ids"] = ["truth-direct-integrity"]
    with pytest.raises(Exception, match="unmatched|Assignment|replay|persisted"):
        ReviewEvaluationResult.from_dict(
            forged,
            submission=submission,
            review_truth=truth,
            evaluator=evaluator,
            judge_results=(),
        )

    forged_outcome = replace(
        result.finding_outcomes[0],
        issue_resolution=FindingResolution.UNGRADED,
        issue_judgement=IssueJudgement.UNKNOWN,
        disposition=FindingDisposition.UNGRADED,
        matched_expected_truth_id=None,
        strict_publishable=False,
    )
    forged = result.to_dict()
    forged["finding_outcomes"] = [forged_outcome.to_dict()]
    with pytest.raises(Exception, match="outcomes|Assignments|replay|persisted"):
        ReviewEvaluationResult.from_dict(
            forged,
            submission=submission,
            review_truth=truth,
            evaluator=evaluator,
            judge_results=(),
        )

    forged = result.to_dict()
    forged["metrics"]["matched_finding_count"] = 0
    with pytest.raises(Exception, match="metric|replay|persisted"):
        ReviewEvaluationResult.from_dict(
            forged,
            submission=submission,
            review_truth=truth,
            evaluator=evaluator,
            judge_results=(),
        )


def test_direct_result_rejects_candidate_request_rebinding(harness: object) -> None:
    evaluator = _evaluator(harness)
    submission = _submission(
        harness,
        _finding("finding-request-binding", "A semantic description."),
    )
    truth = _truth(
        "truth-request-binding",
        "The changed branch can return the wrong result.",
    )
    result = evaluator.evaluate(submission, truth)
    candidate = replace(
        result.expected_candidates[0],
        request_id="forged-request-id",
    )
    forged = result.to_dict()
    forged["expected_candidates"] = [candidate.to_dict()]

    with pytest.raises(Exception, match="request binding|Judge request|replay|persisted"):
        ReviewEvaluationResult.from_dict(
            forged,
            submission=submission,
            review_truth=truth,
            evaluator=evaluator,
            judge_results=(),
        )


def test_direct_result_rejects_unselected_known_invalid_hit(harness: object) -> None:
    evaluator = _evaluator(harness)
    claim = "The changed branch always returns None."
    submission = _submission(harness, _finding("finding-known-binding", claim))
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(),
        known_invalid_findings=(
            KnownInvalidFinding(
                truth_id="truth-known-binding",
                claim=claim,
                category="correctness",
                locations=(),
                rationale="known invalid trap",
            ),
        ),
    )
    result = evaluator.evaluate(submission, truth)
    candidate = replace(result.known_invalid_candidates[0], selected=False)
    forged = result.to_dict()
    forged["known_invalid_candidates"] = [candidate.to_dict()]

    with pytest.raises(
        Exception,
        match="known-invalid outcomes|selected candidates|replay|persisted",
    ):
        ReviewEvaluationResult.from_dict(
            forged,
            submission=submission,
            review_truth=truth,
            evaluator=evaluator,
            judge_results=(),
        )


def test_review_context_rejects_a_noncanonical_source_digest() -> None:
    source = repository_context(
        source_id="tampered-context",
        kind=JudgeContextKind.CODE,
        content="return value",
        revision="head",
        path="src/app.py",
    )
    tampered = replace(source, source_digest="0" * 64)

    with pytest.raises(Exception, match="source_digest|canonical"):
        ReviewContextBundle.create(global_sources=(tampered,))


def test_bound_hydration_rejects_forged_judge_projection_and_result_digest(
    harness: object,
) -> None:
    evaluator = _evaluator(harness)
    submission = _submission(
        harness,
        _finding("finding-forged-decision", "A different defect claim."),
    )
    truth = _truth(
        "truth-forged-decision",
        "The changed branch can return the wrong result.",
    )
    pending = evaluator.evaluate(submission, truth)
    judge_result = _run_scripted_judge(
        pending.judge_requests[0],
        evaluator.evaluator_execution,
        relation="different",
        score_ppm=900_000,
        severity_assessment="consistent",
        actionability="actionable",
    )
    result = evaluator.evaluate(
        submission,
        truth,
        judge_results=(judge_result,),
    )
    assert result.judge_decisions[0].decision.relation.value == "different"

    forged = result.to_dict()
    candidate = forged["expected_candidates"][0]
    candidate["relation"] = "equivalent"
    candidate["edge_weight"] = 1_000_000 + candidate["score_ppm"]
    candidate["selected"] = True
    candidate["reason_codes"] = ["semantic_equivalent"]
    forged["assignments"] = [
        {
            "finding_id": "finding-forged-decision",
            "truth_id": "truth-forged-decision",
            "match_kind": "semantic",
            "weight": candidate["edge_weight"],
            "request_id": candidate["request_id"],
        }
    ]
    outcome = forged["finding_outcomes"][0]
    outcome.update(
        {
            "issue_resolution": "resolved",
            "issue_judgement": "confirmed",
            "disposition": "matched",
            "matched_expected_truth_id": "truth-forged-decision",
            "severity_assessment": "consistent",
            "actionability": "actionable",
            "reason_codes": sorted(
                [
                    "evidence_missing",
                    "evidence_support_not_requested",
                    "expected_match",
                    "not_strict_publishable",
                ]
            ),
        }
    )
    forged["unmatched_expected_truth_ids"] = []
    forged["status"] = "graded"
    forged["coverage"]["finding_resolved_count"] = 1
    forged["metrics"].update(
        {
            "scorable": True,
            "matched_finding_count": 1,
            "matched_expected_truth_count": 1,
            "matched_required_truth_count": 1,
            "unknown_finding_count": 0,
            "unmatched_expected_truth_count": 0,
            "unmatched_required_truth_count": 0,
        }
    )

    with pytest.raises(Exception, match="replay|persisted"):
        ReviewEvaluationResult.from_dict(
            forged,
            submission=submission,
            review_truth=truth,
            evaluator=evaluator,
            judge_results=(judge_result,),
        )

    forged_digest = result.to_dict()
    forged_digest["judge_decisions"][0]["judge_result_digest"] = "0" * 64
    with pytest.raises(Exception, match="replay|persisted"):
        ReviewEvaluationResult.from_dict(
            forged_digest,
            submission=submission,
            review_truth=truth,
            evaluator=evaluator,
            judge_results=(judge_result,),
        )


def test_bound_hydration_rejects_forged_source_binding_digests(
    harness: object,
) -> None:
    evaluator = _evaluator(harness)
    finding = _finding(
        "finding-source-binding",
        "The changed branch can return the wrong result.",
    )
    submission = _submission(harness, finding)
    truth = _truth("truth-source-binding", finding.claim)
    result = evaluator.evaluate(submission, truth)
    forged = result.to_dict()
    for field in (
        "submission_digest",
        "submission_review_digest",
        "review_truth_digest",
        "deterministic_context_digest",
        "evaluator_execution_digest",
    ):
        forged[field] = "0" * 64

    with pytest.raises(Exception, match="replay|persisted"):
        ReviewEvaluationResult.from_dict(
            forged,
            submission=submission,
            review_truth=truth,
            evaluator=evaluator,
            judge_results=(),
        )

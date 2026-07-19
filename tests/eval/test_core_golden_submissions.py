from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from typing import Iterator

import pytest

from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse
from review_agent_eval.evidence_checker import EvidenceReasonCode
from review_agent_eval.intent_evaluator import (
    IntentEvaluationStatus,
    IntentEvaluator,
)
from review_agent_eval.judge import JudgeExecutionResult, JudgeTask, SemanticJudge
from review_agent_eval.models import (
    ClarificationPolicy,
    EvalCase,
    EvalSubmission,
    EvidenceIntegrity,
    EvidenceSupport,
    FindingSeverity,
    IntentClaimJudgement,
    IntentClaimSource,
    IntentResult,
    NovelFindingPolicy,
    SubmissionStatus,
    TruthCompleteness,
    validate_submission_for_case,
)
from review_agent_eval.repository import PreparedRepositoryReplay, RepositoryPreparer
from review_agent_eval.review_evaluator import (
    FindingDisposition,
    ReviewEvaluationPhase,
    ReviewEvaluationResult,
    ReviewEvaluationStatus,
    ReviewEvaluator,
    ReviewJudgeRequestRecord,
)
from tests.eval.test_core_suite import EVAL_ROOT, _banks
from tests.eval.test_judge import _execution, _Factory


GOLDEN_FILENAMES = {
    "perfect": "perfect.json",
    "empty": "empty.json",
    "duplicate": "duplicate.json",
    "fabricated": "fabricated.json",
    "bad_evidence_hash": "bad-evidence.json",
    "bad_evidence_path": "bad-evidence-path.json",
    "bad_evidence_line": "bad-evidence-line.json",
    "unsupported_evidence": "unsupported-evidence.json",
    "compound": "compound.json",
    "judge_unknown": "judge-unknown.json",
    "unsupported_intent": "unsupported-intent.json",
    "contradicted_intent": "contradicted-intent.json",
}

BAD_EVIDENCE_REASON_CODES = {
    "bad_evidence_hash": EvidenceReasonCode.CONTENT_HASH_MISMATCH,
    "bad_evidence_path": EvidenceReasonCode.PATH_NOT_FOUND,
    "bad_evidence_line": EvidenceReasonCode.LINE_RANGE_OUT_OF_BOUNDS,
}


def _golden_paths() -> dict[str, Path]:
    root = EVAL_ROOT / "cases" / "core"
    assert root.is_dir(), "Task 13 Core Case directory is missing: eval/cases/core"
    result = {}
    for scenario, filename in GOLDEN_FILENAMES.items():
        matches = sorted(root.glob("*/golden/%s" % filename))
        assert len(matches) == 1, (
            "Task 13 requires exactly one %s Golden Submission; found %d"
            % (scenario, len(matches))
        )
        result[scenario] = matches[0]
    return result


def _case_index() -> dict[str, EvalCase]:
    return {
        handle.task_id: handle.load()
        for bank in _banks()
        for handle in bank
    }


def _goldens() -> dict[str, tuple[EvalCase, EvalSubmission]]:
    cases = _case_index()
    result = {}
    for scenario, path in _golden_paths().items():
        raw = path.read_bytes()
        submission = EvalSubmission.from_json(raw)
        assert raw == submission.to_json().encode("utf-8"), (
            "%s must be canonical EvalSubmission v1 JSON" % path
        )
        assert submission.task_id in cases, (
            "%s references a task outside the two Core manifests" % path
        )
        assert path.parent.parent.name == submission.task_id, (
            "%s is stored under the wrong Case directory" % path
        )
        validate_submission_for_case(submission, cases[submission.task_id])
        result[scenario] = (cases[submission.task_id], submission)
    return result


def _git_executable() -> Path:
    executable = shutil.which("git")
    assert executable is not None, "Git is required to replay Core Golden Evidence"
    return Path(executable).absolute()


@contextmanager
def _replay(case: EvalCase, tmp_path: Path) -> Iterator[PreparedRepositoryReplay]:
    control = tmp_path / ("golden-replay-" + case.task_id)
    control.mkdir()
    with RepositoryPreparer(
        suite_root=EVAL_ROOT,
        data_root=control / ".eval-data",
        workspace_root=control / ".eval-workspaces",
        git_executable=_git_executable(),
    ) as preparer:
        prepared = preparer.prepare(case.input.repository)
        yield preparer.open_replay(prepared)


def _judge_result(
    evaluator: ReviewEvaluator,
    record: ReviewJudgeRequestRecord,
    mode: str,
) -> JudgeExecutionResult:
    request = record.request
    output = {
        "schema_version": request.rubric.response_schema,
        "request_id": request.request_id,
        "reason_refs": ["item-a"],
    }
    if request.task is JudgeTask.FINDING_EQUIVALENCE:
        compound_truth_id = (
            mode.partition(":")[2] if mode.startswith("compound:") else None
        )
        relation = (
            "equivalent"
            if compound_truth_id is not None and record.truth_id == compound_truth_id
            else "different"
        )
        output.update(
            {
                "relation": relation,
                "score_ppm": 999_999 if relation == "equivalent" else 0,
                "severity_assessment": (
                    "consistent" if relation == "equivalent" else "unknown"
                ),
                "actionability": (
                    "actionable" if relation == "equivalent" else "unknown"
                ),
            }
        )
    elif request.task is JudgeTask.EVIDENCE_SUPPORT:
        output["support"] = (
            "unsupported" if mode == "unsupported_evidence" else "supported"
        )
    elif request.task is JudgeTask.NOVEL_FACTUALITY:
        factuality = "unknown" if mode == "unknown" else "fabricated"
        output.update(
            {
                "factuality": factuality,
                "severity_assessment": (
                    "unknown" if factuality == "unknown" else "consistent"
                ),
                "actionability": (
                    "unknown" if factuality == "unknown" else "actionable"
                ),
            }
        )
    else:  # pragma: no cover - ReviewEvaluator never emits Intent requests
        raise AssertionError("unexpected Review Judge task: %s" % request.task)

    response = ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(output),
    )
    return SemanticJudge(
        adapter_factory=_Factory([response]),
        evaluator_execution=evaluator.evaluator_execution,
    ).execute(request)


def _resolve_review(
    evaluator: ReviewEvaluator,
    submission: EvalSubmission,
    case: EvalCase,
    *,
    mode: str = "different",
) -> ReviewEvaluationResult:
    results = []
    for _round in range(12):
        current = evaluator.evaluate(
            submission,
            case.review_truth,
            judge_results=tuple(results),
        )
        resolved_ids = {
            item.request_id
            for item in (
                *current.judge_decisions,
                *current.judge_failures,
                *current.judge_ungraded,
            )
        }
        pending = [
            item for item in current.judge_requests if item.request_id not in resolved_ids
        ]
        if not pending:
            return current
        results.extend(_judge_result(evaluator, item, mode) for item in pending)
    raise AssertionError("Golden Review Judge work did not converge within 12 rounds")


def _evaluator(
    case: EvalCase,
    replay: PreparedRepositoryReplay,
    submission: EvalSubmission,
) -> ReviewEvaluator:
    return ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        evaluator_execution=_execution(),
    )


def _all_evidence_reason_codes(result: ReviewEvaluationResult) -> set[EvidenceReasonCode]:
    return {
        diagnostic.reason_code
        for evidence_result in result.evidence_integrity_results
        for diagnostic in (
            *evidence_result.diagnostics,
            *(
                diagnostic
                for item_result in evidence_result.item_results
                for diagnostic in item_result.diagnostics
            ),
        )
    }


def _assert_submission_evidence_is_closed(submission: EvalSubmission) -> None:
    assert submission.review is not None
    evidence_ids = {item.evidence_id for item in submission.evidence}
    refs = {
        ref
        for finding in submission.review.findings
        for ref in finding.evidence_refs
    }
    assert evidence_ids == refs
    assert all(finding.evidence_refs for finding in submission.review.findings)


def _assert_integrity_is_healthy(result: ReviewEvaluationResult) -> None:
    assert result.metrics.evidence_invalid_count == 0
    assert result.metrics.evidence_missing_count == 0
    assert result.metrics.evidence_valid_count == result.metrics.generated_finding_count


def test_golden_submissions_are_canonical_case_bound_and_cover_intent_protocol() -> None:
    goldens = _goldens()
    assert set(goldens) == set(GOLDEN_FILENAMES)
    assert all(
        submission.status is SubmissionStatus.COMPLETED
        and submission.failure is None
        and submission.trace_ref is None
        for _case, submission in goldens.values()
    )
    submissions = [submission for _case, submission in goldens.values()]
    assert len({item.trial_id for item in submissions}) == len(submissions)
    assert len({item.digest() for item in submissions}) == len(submissions)
    for case, submission in goldens.values():
        assert case.intent_truth.scorable is True
        assert case.review_truth.completeness is TruthCompleteness.CLOSED_WORLD
        assert submission.intent is not None
        assert submission.intent.status in {IntentResult.SUFFICIENT, IntentResult.PARTIAL}
        _assert_submission_evidence_is_closed(submission)
        assert {claim.source for claim in submission.intent.claims} <= set(
            IntentClaimSource
        )
    assert {
        submission.intent.status
        for _case, submission in goldens.values()
        if submission.intent is not None
    } == {IntentResult.SUFFICIENT, IntentResult.PARTIAL}
    assert {
        claim.source
        for _case, submission in goldens.values()
        if submission.intent is not None
        for claim in submission.intent.claims
    } == set(IntentClaimSource)
    assert goldens["judge_unknown"][0].review_truth.novel_finding_policy is (
        NovelFindingPolicy.VERIFY
    )
    assert goldens["fabricated"][0].review_truth.known_invalid_findings
    assert any(
        finding.severity is FindingSeverity.CRITICAL
        for finding in goldens["empty"][0].review_truth.expected_findings
        if finding.required
    )


def test_perfect_golden_passes_intent_review_and_non_unique_evidence(
    tmp_path: Path,
) -> None:
    case, submission = _goldens()["perfect"]
    assert submission.intent is not None and submission.review is not None
    assert submission.intent.status is IntentResult.SUFFICIENT
    assert submission.intent.goal is None
    assert not submission.intent.acceptance_criteria
    assert not submission.intent.scope
    assert not submission.intent.constraints
    assert Counter(claim.text for claim in submission.intent.claims) == Counter(
        claim.text for claim in case.intent_truth.expected_claims
    )
    assert Counter(finding.claim for finding in submission.review.findings) == Counter(
        finding.claim for finding in case.review_truth.expected_findings
    )
    assert all(finding.evidence_refs for finding in submission.review.findings)
    assert {item.evidence_id for item in submission.evidence} == {
        ref
        for finding in submission.review.findings
        for ref in finding.evidence_refs
    }
    assert {item.revision for item in submission.evidence} == {
        case.input.repository.base_revision,
        case.input.repository.head_revision,
    }
    assert all(
        item.path == "src/timeout.py"
        and item.from_line == 1
        and item.to_line == 4
        and "return value" in item.excerpt
        for item in submission.evidence
    )

    intent_result = IntentEvaluator().evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    assert intent_result.status is IntentEvaluationStatus.GRADED
    assert intent_result.judge_requests == ()
    assert intent_result.metrics.intent_case_pass is True

    anchor_locations = {
        (location.path, location.from_line, location.to_line)
        for finding in case.review_truth.expected_findings
        for anchor in finding.evidence_anchors
        for location in anchor.locations
    }
    evidence_locations = {
        (evidence.path, evidence.from_line, evidence.to_line)
        for evidence in submission.evidence
        if evidence.path is not None
    }
    assert any(
        not anchor.locations
        for finding in case.review_truth.expected_findings
        for anchor in finding.evidence_anchors
    ) or bool(evidence_locations - anchor_locations), (
        "perfect Evidence must demonstrate that anchors are not a unique legal path"
    )

    with _replay(case, tmp_path) as replay:
        result = _resolve_review(_evaluator(case, replay, submission), submission, case)
    assert result.status is ReviewEvaluationStatus.GRADED
    assert result.metrics.matched_expected_truth_count == len(
        case.review_truth.expected_findings
    )
    assert result.metrics.matched_required_truth_count == len(
        case.review_truth.expected_findings
    )
    assert result.metrics.unmatched_required_truth_count == 0
    assert result.metrics.fabricated_finding_count == 0
    assert result.metrics.duplicate_finding_count == 0
    assert result.metrics.evidence_valid_count == len(submission.review.findings)
    assert result.metrics.evidence_supported_count == len(submission.review.findings)
    assert result.metrics.strict_publishable_count == len(submission.review.findings)
    assert all(item.strict_publishable for item in result.finding_outcomes)


def test_unsupported_and_contradicted_intent_goldens_are_distinct(
    tmp_path: Path,
) -> None:
    goldens = _goldens()

    unsupported_case, unsupported = goldens["unsupported_intent"]
    assert unsupported.intent is not None
    unsupported_result = IntentEvaluator().evaluate(
        unsupported.intent,
        unsupported_case.intent_truth,
        unsupported_case.clarification_script,
    )
    assert unsupported_result.metrics.unsupported_claim_count == 1
    assert unsupported_result.metrics.contradicted_claim_count == 0
    assert any(
        item.judgement is IntentClaimJudgement.UNSUPPORTED
        for item in unsupported_result.claim_outcomes
    )

    contradicted_case, contradicted = goldens["contradicted_intent"]
    assert contradicted.intent is not None
    contradicted_result = IntentEvaluator().evaluate(
        contradicted.intent,
        contradicted_case.intent_truth,
        contradicted_case.clarification_script,
    )
    assert contradicted_result.metrics.contradicted_claim_count == 1
    assert contradicted_result.metrics.unsupported_claim_count == 0
    assert any(
        item.judgement is IntentClaimJudgement.CONTRADICTED
        for item in contradicted_result.claim_outcomes
    )
    assert contradicted.review is not None
    assert len(contradicted.review.findings) == 1
    assert contradicted.review.findings[0].claim == (
        contradicted_case.review_truth.expected_findings[0].claim
    )
    with _replay(contradicted_case, tmp_path) as replay:
        review_result = _resolve_review(
            _evaluator(contradicted_case, replay, contradicted),
            contradicted,
            contradicted_case,
        )
    assert review_result.status is ReviewEvaluationStatus.GRADED
    assert review_result.metrics.unmatched_required_truth_count == 0
    assert review_result.metrics.matched_required_truth_count == 1
    _assert_integrity_is_healthy(review_result)
    assert review_result.metrics.evidence_supported_count == 1
    assert review_result.metrics.strict_publishable_count == 1


def test_valid_but_irrelevant_evidence_is_unsupported(tmp_path: Path) -> None:
    case, submission = _goldens()["unsupported_evidence"]
    assert submission.review is not None and len(submission.review.findings) == 1
    with _replay(case, tmp_path) as replay:
        result = _resolve_review(
            _evaluator(case, replay, submission),
            submission,
            case,
            mode="unsupported_evidence",
        )
    assert result.status is ReviewEvaluationStatus.GRADED
    assert result.metrics.matched_expected_truth_count == 1
    _assert_integrity_is_healthy(result)
    assert result.metrics.evidence_unsupported_count == 1
    assert result.metrics.strict_publishable_count == 0
    outcome = result.finding_outcomes[0]
    assert outcome.issue_judgement.value == "confirmed"
    assert outcome.evidence_integrity is EvidenceIntegrity.VALID
    assert outcome.evidence_support is EvidenceSupport.UNSUPPORTED
    assert not outcome.strict_publishable


def test_empty_duplicate_fabricated_and_compound_goldens(
    tmp_path: Path,
) -> None:
    goldens = _goldens()

    empty_case, empty = goldens["empty"]
    assert empty.review is not None
    assert empty.review.findings == () and empty.evidence == ()
    with _replay(empty_case, tmp_path) as replay:
        empty_result = _resolve_review(
            _evaluator(empty_case, replay, empty), empty, empty_case
        )
    assert empty_result.status is ReviewEvaluationStatus.GRADED
    assert empty_result.metrics.generated_finding_count == 0
    assert empty_result.metrics.unmatched_required_truth_count == sum(
        item.required for item in empty_case.review_truth.expected_findings
    )
    assert any(
        item.required and item.severity is FindingSeverity.CRITICAL
        for item in empty_case.review_truth.expected_findings
    )

    duplicate_case, duplicate = goldens["duplicate"]
    assert duplicate.review is not None
    duplicate_counts = Counter(item.claim for item in duplicate.review.findings)
    assert max(duplicate_counts.values(), default=0) >= 2
    with _replay(duplicate_case, tmp_path) as replay:
        duplicate_result = _resolve_review(
            _evaluator(duplicate_case, replay, duplicate),
            duplicate,
            duplicate_case,
        )
    assert duplicate_result.status is ReviewEvaluationStatus.GRADED
    assert duplicate_result.metrics.matched_expected_truth_count == 1
    assert duplicate_result.metrics.unmatched_required_truth_count == 0
    assert duplicate_result.metrics.duplicate_finding_count >= 1
    assert len(duplicate_result.assignments) == 1
    _assert_integrity_is_healthy(duplicate_result)
    assert any(
        item.disposition is FindingDisposition.DUPLICATE
        for item in duplicate_result.finding_outcomes
    )

    fabricated_case, fabricated = goldens["fabricated"]
    assert fabricated.review is not None
    assert len(fabricated.review.findings) == 1
    assert fabricated.review.findings[0].claim in {
        item.claim for item in fabricated_case.review_truth.known_invalid_findings
    }
    with _replay(fabricated_case, tmp_path) as replay:
        fabricated_result = _resolve_review(
            _evaluator(fabricated_case, replay, fabricated),
            fabricated,
            fabricated_case,
        )
    assert fabricated_result.status is ReviewEvaluationStatus.GRADED
    assert fabricated_result.metrics.fabricated_finding_count == 1
    _assert_integrity_is_healthy(fabricated_result)
    assert fabricated_result.finding_outcomes[0].disposition is (
        FindingDisposition.KNOWN_INVALID
    )

    compound_case, compound = goldens["compound"]
    assert compound.review is not None and len(compound.review.findings) == 1
    compound_text = " ".join(compound.review.findings[0].claim.casefold().split())
    assert sum(
        " ".join(item.claim.casefold().split()) in compound_text
        for item in compound_case.review_truth.expected_findings
    ) >= 2
    with _replay(compound_case, tmp_path) as replay:
        compound_result = _resolve_review(
            _evaluator(compound_case, replay, compound),
            compound,
            compound_case,
            mode="compound:" + compound_case.review_truth.expected_findings[0].truth_id,
        )
    assert len(compound_result.assignments) == 1
    assert compound_result.metrics.matched_expected_truth_count == 1
    assert compound_result.metrics.unmatched_required_truth_count == 1
    assert compound_result.metrics.generated_finding_count == 1
    _assert_integrity_is_healthy(compound_result)
    assert compound_result.metrics.evidence_supported_count == 1


@pytest.mark.parametrize(
    ("scenario", "expected_reason"),
    tuple(BAD_EVIDENCE_REASON_CODES.items()),
)
def test_each_bad_evidence_golden_breaks_only_one_integrity_dimension(
    tmp_path: Path,
    scenario: str,
    expected_reason: EvidenceReasonCode,
) -> None:
    case, submission = _goldens()[scenario]
    assert submission.review is not None and len(submission.review.findings) == 1
    assert submission.review.findings[0].claim == (
        case.review_truth.expected_findings[0].claim
    )
    with _replay(case, tmp_path) as replay:
        result = _resolve_review(
            _evaluator(case, replay, submission), submission, case
        )
    assert result.status is ReviewEvaluationStatus.GRADED
    assert result.metrics.matched_expected_truth_count == 1
    assert result.metrics.unmatched_required_truth_count == 0
    assert result.metrics.evidence_invalid_count == 1
    assert result.metrics.evidence_missing_count == 0
    assert result.metrics.strict_publishable_count == 0
    assert all(
        item.issue_judgement.value == "confirmed"
        and item.evidence_integrity is EvidenceIntegrity.INVALID
        and not item.strict_publishable
        for item in result.finding_outcomes
    )
    target_reasons = set(BAD_EVIDENCE_REASON_CODES.values())
    assert _all_evidence_reason_codes(result) & target_reasons == {expected_reason}


def test_judge_unknown_golden_stays_unknown_and_fail_closed(tmp_path: Path) -> None:
    case, submission = _goldens()["judge_unknown"]
    assert case.review_truth.novel_finding_policy is NovelFindingPolicy.VERIFY
    assert submission.review is not None and len(submission.review.findings) == 1
    assert "repository-external consumers" in submission.review.findings[0].claim
    assert submission.review.findings[0].claim not in {
        item.claim
        for item in (
            *case.review_truth.expected_findings,
            *case.review_truth.known_invalid_findings,
        )
    }

    with _replay(case, tmp_path) as replay:
        result = _resolve_review(
            _evaluator(case, replay, submission),
            submission,
            case,
            mode="unknown",
        )

    assert result.status is ReviewEvaluationStatus.UNGRADED
    assert any(
        item.request.task is JudgeTask.NOVEL_FACTUALITY
        for item in result.judge_requests
    )
    assert result.coverage.semantic_unknown_count >= 1
    assert result.coverage.judge_pending_count == 0
    assert result.assignments == ()
    assert result.metrics.unknown_finding_count == 1
    assert result.metrics.fabricated_finding_count == 0
    _assert_integrity_is_healthy(result)
    assert result.finding_outcomes[0].issue_judgement.value == "unknown"

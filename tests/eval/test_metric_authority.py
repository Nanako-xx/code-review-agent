from __future__ import annotations

from dataclasses import replace

import pytest

from review_agent_eval.metrics import (
    CoreMetric,
    MetricNullReason,
    MetricSourceStatus,
    MetricsAggregator,
    TrialScorer,
    _metric_authority_policy_digest,
)
from review_agent_eval.models import (
    DiffSide,
    ExpectedFinding,
    FindingSeverity,
    MetricAuthority,
    MetricAuthoritySource,
    RequiredContextLevel,
    TruthLocation,
    UnsupportedProtocolVersionError,
)
from review_agent_eval.review_evaluator import (
    FindingMatchRelation,
    ReviewEvaluationResult,
    ReviewEvaluator,
)
from tests.eval.test_frozen_evidence import (
    _evaluator as _frozen_evaluator,
    _evidence as _frozen_evidence,
    _finding as _frozen_finding,
    _submission as _frozen_submission,
    _truth as _frozen_truth,
    frozen_harness,
)
from tests.eval.test_metrics import (
    CLAIM,
    _completed_submission,
    _failed_submission,
    _score_completed,
    _score_sources,
)
from tests.eval.test_review_evaluator import (
    _evaluator as _review_evaluator,
    _finding as _review_finding,
    _run_scripted_judge,
    _submission as _review_submission,
    _truth as _review_truth,
    harness,
)
from tests.eval.test_review_truth_completeness import TARGET_MATERIALIZATION_ID


CORE_AUTHORITY = MetricAuthority(
    severity_scorable=True,
    severity_authority=MetricAuthoritySource.EXPERT_ANNOTATION,
    location_scorable=True,
    location_authority=MetricAuthoritySource.EXPERT_ANNOTATION,
)
AACR_AUTHORITY = MetricAuthority(
    severity_scorable=False,
    severity_authority=None,
    location_scorable=True,
    location_authority=MetricAuthoritySource.UPSTREAM_ANNOTATION,
)
SWE_AUTHORITY = MetricAuthority(
    severity_scorable=False,
    severity_authority=None,
    location_scorable=False,
    location_authority=None,
)

SEVERITY_METRICS = (
    CoreMetric.SEVERITY_WEIGHTED_RECALL,
    CoreMetric.CRITICAL_HIGH_MISS_COUNT,
)
LOCATION_METRICS = (CoreMetric.LINE_PRECISION, CoreMetric.LINE_RECALL)
AUTHORITY_METRICS = (*SEVERITY_METRICS, *LOCATION_METRICS)


def _expected(
    authority: MetricAuthority,
    *,
    truth_id: str = "truth-review",
    claim: str = CLAIM,
    required: bool = True,
    line: int = 1,
) -> ExpectedFinding:
    return ExpectedFinding(
        truth_id=truth_id,
        claim=claim,
        severity=(FindingSeverity.HIGH if authority.severity_scorable else None),
        category="correctness",
        required=required,
        metric_authority=authority,
        # Unscorable locations intentionally remain as equivalence metadata.
        locations=(
            TruthLocation(
                path="src/app.py",
                side=DiffSide.RIGHT,
                from_line=line,
                to_line=line,
            ),
        ),
        evidence_anchors=(),
        required_context_level=RequiredContextLevel.DIFF,
        rationale="metric authority regression truth",
    )


def _completed_score(*findings: ExpectedFinding):
    case, replay, execution, config = _score_sources(review_findings=findings)
    submission, intent_result, review_result, score = _score_completed(
        case, replay, execution, config, 1
    )
    return case, replay, execution, config, submission, intent_result, review_result, score


def _failed_score(*findings: ExpectedFinding):
    case, _replay, execution, config = _score_sources(review_findings=findings)
    submission = _failed_submission(config, 1)
    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metric-authority-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=None,
        review_result=None,
    )
    return score


def _assert_not_scorable(score, metrics=AUTHORITY_METRICS) -> None:
    for metric in metrics:
        contribution = score.contribution(metric)
        assert contribution.source_status is MetricSourceStatus.NOT_SCORABLE
        assert contribution.numerator is None
        assert contribution.denominator is None


def test_core_authority_scores_all_four_sensitive_metrics() -> None:
    *_, score = _completed_score(_expected(CORE_AUTHORITY))

    assert score.contribution(CoreMetric.SEVERITY_WEIGHTED_RECALL).numerator == 4
    assert score.contribution(CoreMetric.SEVERITY_WEIGHTED_RECALL).denominator == 4
    assert score.contribution(CoreMetric.CRITICAL_HIGH_MISS_COUNT).numerator == 0
    assert score.contribution(CoreMetric.CRITICAL_HIGH_MISS_COUNT).denominator == 1
    assert score.contribution(CoreMetric.LINE_PRECISION).numerator == 1
    assert score.contribution(CoreMetric.LINE_PRECISION).denominator == 1
    assert score.contribution(CoreMetric.LINE_RECALL).numerator == 1
    assert score.contribution(CoreMetric.LINE_RECALL).denominator == 1


def test_aacr_authority_excludes_severity_but_scores_locations() -> None:
    *_, score = _completed_score(_expected(AACR_AUTHORITY))

    _assert_not_scorable(score, SEVERITY_METRICS)
    assert score.contribution(CoreMetric.LINE_PRECISION).source_status is MetricSourceStatus.GRADED
    assert score.contribution(CoreMetric.LINE_PRECISION).numerator == 1
    assert score.contribution(CoreMetric.LINE_RECALL).source_status is MetricSourceStatus.GRADED
    assert score.contribution(CoreMetric.LINE_RECALL).numerator == 1


def test_optional_location_authority_scores_precision_but_not_recall() -> None:
    optional = _expected(
        CORE_AUTHORITY,
        truth_id="truth-optional-location",
        required=False,
    )
    *_, score = _completed_score(optional)

    precision = score.contribution(CoreMetric.LINE_PRECISION)
    recall = score.contribution(CoreMetric.LINE_RECALL)
    assert (precision.source_status, precision.numerator, precision.denominator) == (
        MetricSourceStatus.GRADED,
        1,
        1,
    )
    assert (recall.source_status, recall.numerator, recall.denominator) == (
        MetricSourceStatus.NOT_SCORABLE,
        None,
        None,
    )


def test_unassigned_optional_location_keeps_precision_eligible_zero_over_zero() -> None:
    optional = _expected(
        CORE_AUTHORITY,
        truth_id="truth-optional-unassigned",
        required=False,
    )
    case, replay, execution, config = _score_sources(review_findings=(optional,))
    *_, score = _score_completed(
        case,
        replay,
        execution,
        config,
        1,
        finding_count=0,
    )

    precision = score.contribution(CoreMetric.LINE_PRECISION)
    recall = score.contribution(CoreMetric.LINE_RECALL)
    assert (precision.source_status, precision.numerator, precision.denominator) == (
        MetricSourceStatus.GRADED,
        0,
        0,
    )
    assert recall.source_status is MetricSourceStatus.NOT_SCORABLE


def test_swe_repository_authority_is_not_scorable_without_hiding_issue_metrics() -> None:
    *_, review_result, score = _completed_score(_expected(SWE_AUTHORITY))

    assert review_result.location_candidates == ()
    _assert_not_scorable(score)
    assert score.contribution(CoreMetric.ISSUE_PRECISION).source_status is MetricSourceStatus.GRADED
    assert score.contribution(CoreMetric.ISSUE_RECALL).source_status is MetricSourceStatus.GRADED
    assert score.contribution(CoreMetric.ISSUE_RECALL).numerator == 1
    assert score.contribution(CoreMetric.ISSUE_RECALL).denominator == 1

    case_score = MetricsAggregator().aggregate_case((score,), planned_trial_count=1)
    aggregate = MetricsAggregator().aggregate_cases(
        (case_score,), source_trials=(score,)
    )
    for metric in AUTHORITY_METRICS:
        case_metric = case_score.metric(metric)
        aggregate_metric = aggregate.metric(metric)
        assert case_metric.null_reason is MetricNullReason.NOT_SCORABLE
        assert case_metric.coverage.not_scorable_count == 1
        assert case_metric.coverage.zero_denominator_count == 0
        assert aggregate_metric.null_reason is MetricNullReason.NOT_SCORABLE
        assert aggregate_metric.coverage.not_scorable_count == 1
        assert aggregate_metric.coverage.zero_denominator_count == 0


def test_mixed_authority_excludes_placeholders_without_changing_eligible_scores() -> None:
    eligible = _expected(CORE_AUTHORITY, truth_id="truth-a-eligible")
    excluded = _expected(SWE_AUTHORITY, truth_id="truth-z-placeholder")
    *_, review_result, score = _completed_score(eligible, excluded)

    assert review_result.assignments[0].truth_id == eligible.truth_id
    assert {item.truth_id for item in review_result.location_candidates} == {
        eligible.truth_id
    }
    assert score.contribution(CoreMetric.SEVERITY_WEIGHTED_RECALL).numerator == 4
    assert score.contribution(CoreMetric.SEVERITY_WEIGHTED_RECALL).denominator == 4
    assert score.contribution(CoreMetric.LINE_PRECISION).numerator == 1
    assert score.contribution(CoreMetric.LINE_PRECISION).denominator == 1
    assert score.contribution(CoreMetric.LINE_RECALL).numerator == 1
    assert score.contribution(CoreMetric.LINE_RECALL).denominator == 1
    # Non-authority issue metrics still see both required truths.
    assert score.contribution(CoreMetric.ISSUE_RECALL).numerator == 1
    assert score.contribution(CoreMetric.ISSUE_RECALL).denominator == 2


def test_failure_as_miss_filters_by_authority_and_never_fabricates_empty_dimensions() -> None:
    eligible = _expected(CORE_AUTHORITY, truth_id="truth-a-eligible")
    excluded = _expected(SWE_AUTHORITY, truth_id="truth-z-placeholder")
    mixed = _failed_score(eligible, excluded)

    severity = mixed.contribution(CoreMetric.SEVERITY_WEIGHTED_RECALL)
    severe_misses = mixed.contribution(CoreMetric.CRITICAL_HIGH_MISS_COUNT)
    line_recall = mixed.contribution(CoreMetric.LINE_RECALL)
    assert (severity.source_status, severity.numerator, severity.denominator) == (
        MetricSourceStatus.FAILURE_AS_MISS,
        0,
        4,
    )
    assert (severe_misses.source_status, severe_misses.numerator) == (
        MetricSourceStatus.FAILURE_AS_MISS,
        1,
    )
    assert (line_recall.source_status, line_recall.numerator, line_recall.denominator) == (
        MetricSourceStatus.FAILURE_AS_MISS,
        0,
        1,
    )
    assert mixed.contribution(CoreMetric.LINE_PRECISION).source_status is MetricSourceStatus.FAILURE_EXCLUDED
    assert mixed.contribution(CoreMetric.ISSUE_RECALL).denominator == 2

    no_authority = _failed_score(_expected(SWE_AUTHORITY))
    _assert_not_scorable(no_authority)
    assert no_authority.contribution(CoreMetric.ISSUE_RECALL).source_status is MetricSourceStatus.FAILURE_AS_MISS
    assert no_authority.contribution(CoreMetric.ISSUE_RECALL).denominator == 1


def test_unscorable_location_has_no_audit_and_hydration_cannot_forge_one(
    harness,
) -> None:
    evaluator = _review_evaluator(harness)
    finding = _review_finding("finding-authority-location", CLAIM)
    submission = _review_submission(harness, finding)
    core_truth = _review_truth("truth-authority-location", CLAIM)
    core_result = evaluator.evaluate(submission, core_truth)
    assert len(core_result.location_candidates) == 1

    unscorable_expected = replace(
        core_truth.expected_findings[0],
        severity=None,
        metric_authority=SWE_AUTHORITY,
    )
    unscorable_truth = replace(
        core_truth, expected_findings=(unscorable_expected,)
    )
    unscorable_result = evaluator.evaluate(submission, unscorable_truth)
    assert unscorable_result.location_candidates == ()

    forged = unscorable_result.to_dict()
    forged["location_candidates"] = [core_result.location_candidates[0].to_dict()]
    with pytest.raises(
        ValueError, match="location|MetricAuthority|deterministic replay"
    ):
        ReviewEvaluationResult.from_dict(
            forged,
            submission=submission,
            review_truth=unscorable_truth,
            evaluator=evaluator,
            judge_results=(),
        )


def test_frozen_unscorable_location_metadata_does_not_require_a_matcher(
    frozen_harness,
) -> None:
    evidence = _frozen_evidence(frozen_harness)
    finding = _frozen_finding(evidence.evidence_id)
    truth = _frozen_truth(finding.claim)
    expected = replace(
        truth.expected_findings[0],
        locations=(
            TruthLocation(
                path="metadata/only.py",
                side=DiffSide.RIGHT,
                from_line=1,
                to_line=1,
            ),
        ),
    )
    truth = replace(truth, expected_findings=(expected,))

    result = _frozen_evaluator(frozen_harness).evaluate(
        _frozen_submission(frozen_harness, finding, evidence), truth
    )

    assert result.location_candidates == ()
    assert result.assignments[0].truth_id == expected.truth_id


def test_authority_does_not_change_exact_or_semantic_assignment_edges(harness) -> None:
    evaluator = _review_evaluator(harness)

    exact_finding = _review_finding("finding-exact-authority", CLAIM)
    exact_submission = _review_submission(harness, exact_finding)
    exact_core_truth = _review_truth("truth-edge", CLAIM)
    exact_swe_truth = replace(
        exact_core_truth,
        expected_findings=(
            replace(
                exact_core_truth.expected_findings[0],
                severity=None,
                metric_authority=SWE_AUTHORITY,
            ),
        ),
    )
    exact_core = evaluator.evaluate(exact_submission, exact_core_truth)
    exact_swe = evaluator.evaluate(exact_submission, exact_swe_truth)
    assert exact_core.expected_candidates[0].relation is FindingMatchRelation.EQUIVALENT
    assert exact_swe.expected_candidates[0].relation is FindingMatchRelation.EQUIVALENT
    assert exact_core.expected_candidates[0].edge_weight == exact_swe.expected_candidates[0].edge_weight
    assert exact_core.assignments == exact_swe.assignments

    semantic_finding = _review_finding(
        "finding-semantic-authority",
        "A differently worded description of the same defect.",
    )
    semantic_submission = _review_submission(harness, semantic_finding)
    semantic_core_pending = evaluator.evaluate(semantic_submission, exact_core_truth)
    semantic_swe_pending = evaluator.evaluate(semantic_submission, exact_swe_truth)
    core_decision = _run_scripted_judge(
        semantic_core_pending.judge_requests[0],
        evaluator.evaluator_execution,
        relation="equivalent",
        score_ppm=900_000,
        severity_assessment="consistent",
        actionability="actionable",
    )
    swe_decision = _run_scripted_judge(
        semantic_swe_pending.judge_requests[0],
        evaluator.evaluator_execution,
        relation="equivalent",
        score_ppm=900_000,
        severity_assessment="consistent",
        actionability="actionable",
    )
    semantic_core = evaluator.evaluate(
        semantic_submission, exact_core_truth, judge_results=(core_decision,)
    )
    semantic_swe = evaluator.evaluate(
        semantic_submission, exact_swe_truth, judge_results=(swe_decision,)
    )
    assert semantic_core.expected_candidates[0].relation is FindingMatchRelation.EQUIVALENT
    assert semantic_swe.expected_candidates[0].relation is FindingMatchRelation.EQUIVALENT
    assert semantic_core.expected_candidates[0].edge_weight == semantic_swe.expected_candidates[0].edge_weight
    assert tuple(
        (item.finding_id, item.truth_id, item.match_kind, item.weight)
        for item in semantic_core.assignments
    ) == tuple(
        (item.finding_id, item.truth_id, item.match_kind, item.weight)
        for item in semantic_swe.assignments
    )


def test_repository_profile_helper_uses_bound_target_materialization() -> None:
    case, replay, execution, config = _score_sources(
        review_findings=(_expected(CORE_AUTHORITY),)
    )
    submission = _completed_submission(config, 1)
    result = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(submission, case.review_truth)

    assert result.assignments


def test_unknown_metric_authority_runtime_is_rejected_by_digest_and_scorer() -> None:
    (
        case,
        _replay,
        execution,
        config,
        submission,
        intent_result,
        review_result,
        _score,
    ) = _completed_score(_expected(CORE_AUTHORITY))
    unknown_version = "metric-authority-v3"
    unsupported = replace(
        execution,
        metric_authority_policy_version=unknown_version,
    )

    with pytest.raises(UnsupportedProtocolVersionError):
        _metric_authority_policy_digest(unknown_version)
    with pytest.raises(UnsupportedProtocolVersionError):
        TrialScorer().score(
            run_config=config,
            evaluator_execution=unsupported,
            evaluation_revision="metric-authority-unknown-v1",
            eval_case=case,
            submission=submission,
            trial_index=1,
            intent_result=intent_result,
            review_result=review_result,
        )

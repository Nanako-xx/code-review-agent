from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace

import pytest

from review_agent_eval.cases import CaseDimension, RunCaseSnapshot, SuiteManifest
from review_agent_eval.intent_evaluator import IntentEvaluator
from review_agent_eval.metrics import (
    AggregateScore,
    CaseScore,
    CoreMetric,
    DEFAULT_LINE_METRIC_POLICY,
    DEFAULT_METRICS_POLICY,
    DEFAULT_SEVERITY_WEIGHT_POLICY,
    FailureOutcomePolicy,
    LineMetricPolicy,
    MetricNullReason,
    MetricSourceStatus,
    MetricsAggregator,
    MetricsPolicy,
    SeverityWeightPolicy,
    TrialScore,
    TrialScorer,
)
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    CaseOrigin,
    CaseSource,
    ClarificationPolicy,
    ClarificationScript,
    DiffSide,
    EvalCase,
    EvalCaseInput,
    EvalSubmission,
    ExpectedFinding,
    ExpectedIntentClaim,
    FailureCode,
    FindingSeverity,
    IntentAuthority,
    IntentDimension,
    IntentResult,
    IntentTruth,
    MetricAuthority,
    MetricAuthoritySource,
    NovelFindingPolicy,
    RequiredContextLevel,
    ReviewTruth,
    ReviewEvaluatorContext,
    SubmissionFailure,
    SubmissionFinding,
    SubmissionIntent,
    SubmissionReview,
    SubmissionStatus,
    SubmissionUsage,
    TruthCompleteness,
    TruthLocation,
    canonical_sha256,
)
from review_agent_eval.review_evaluator import ReviewEvaluator
from tests.eval.test_config import run_config
from tests.eval.test_intent_evaluator import judge_failure
from tests.eval.test_review_evaluator import _run_scripted_judge
from tests.eval.test_review_truth_completeness import _execution, _fixture
from tests.eval.test_review_truth_completeness import TARGET_MATERIALIZATION_ID


CLAIM = "The changed branch can return the wrong result."
GOAL = "Preserve the changed branch result."


def _core_authority() -> MetricAuthority:
    return MetricAuthority(
        severity_scorable=True,
        severity_authority=MetricAuthoritySource.EXPERT_ANNOTATION,
        location_scorable=True,
        location_authority=MetricAuthoritySource.EXPERT_ANNOTATION,
    )


def _case_and_snapshot(
    *,
    with_location: bool = False,
    intent_scorable: bool = True,
    review_findings=None,
):
    eval_input, replay = _fixture()
    truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(
            review_findings
            if review_findings is not None
            else (
                ExpectedFinding(
                    truth_id="truth-review",
                    claim=CLAIM,
                    severity=FindingSeverity.HIGH,
                    category="correctness",
                    required=True,
                    metric_authority=(
                        _core_authority()
                        if with_location
                        else replace(
                            _core_authority(),
                            location_scorable=False,
                            location_authority=None,
                        )
                    ),
                    locations=(
                        (
                            TruthLocation(
                                path="src/app.py",
                                side=DiffSide.RIGHT,
                                from_line=1,
                                to_line=1,
                            ),
                        )
                        if with_location
                        else ()
                    ),
                    evidence_anchors=(),
                    required_context_level=RequiredContextLevel.DIFF,
                    rationale="required defect",
                ),
            )
        ),
        known_invalid_findings=(),
    )
    case = EvalCase(
        schema_version=EVAL_CASE_SCHEMA_VERSION,
        task_id=eval_input.task_id,
        case_version=1,
        source=CaseSource(
            suite="metrics-suite",
            origin=CaseOrigin.HAND_AUTHORED,
            source_id="metrics-source",
            source_version="v1",
            source_uri=None,
            license=None,
            content_hash="9" * 64,
        ),
        input=EvalCaseInput(
            review_target=eval_input.review_target,
        ),
        clarification_script=ClarificationScript(max_rounds=1, answers=()),
        intent_truth=(
            IntentTruth(
                scorable=True,
                authority=IntentAuthority.EXPLICIT_AUTHOR_METADATA,
                expected_claims=(
                    ExpectedIntentClaim(
                        truth_id="truth-intent",
                        dimension=IntentDimension.GOAL,
                        text=GOAL,
                        required=True,
                    ),
                ),
                forbidden_claims=(),
                clarification_policy=ClarificationPolicy.NOT_REQUIRED,
            )
            if intent_scorable
            else IntentTruth(
                scorable=False,
                authority=None,
                expected_claims=(),
                forbidden_claims=(),
                clarification_policy=None,
            )
        ),
        review_truth=truth,
        review_evaluator_context=ReviewEvaluatorContext(truth_contexts=()),
    )
    case_bytes = case.to_json().encode("utf-8")
    manifest = SuiteManifest.from_dict(
        {
            "schema_version": "suite_manifest_v2",
            "suite_id": "metrics-suite",
            "suite_version": "v1",
            "wire_contract": {
                "case_schema_version": "eval_case_v2",
                "input_schema_version": "eval_input_v2",
                "submission_schema_version": "eval_submission_v2",
                "review_target_kind": "repository",
                "materializer_protocol": "repository-materializer-v2",
            },
            "source": {
                "kind": "core",
                "source_id": "metrics-suite-source",
                "source_version": "v1",
                "source_uri": None,
                "license": None,
                "content_hash": "8" * 64,
                "preparation_binding": None,
            },
            "cases": [
                {
                    "task_id": case.task_id,
                    "case_version": case.case_version,
                    "path": "cases/metrics.json",
                    "split": "regression",
                    "protocol_id": "native_repository",
                    "dimensions": [
                        {"name": "language", "value": "python"},
                        {"name": "pr_size", "value": "small"},
                    ],
                    "raw_file_size_bytes": len(case_bytes),
                    "raw_file_sha256": hashlib.sha256(case_bytes).hexdigest(),
                    "canonical_case_digest": case.digest(),
                    "eval_input_digest": case.eval_input().digest(),
                    "truth_completeness": "closed_world",
                }
            ],
        }
    )
    snapshot = RunCaseSnapshot.build(manifest, ((manifest.cases[0], case),))
    return case, snapshot, replay


def _finding(finding_id: str) -> SubmissionFinding:
    return SubmissionFinding(
        finding_id=finding_id,
        claim=CLAIM,
        severity=FindingSeverity.HIGH,
        path="src/app.py",
        side=DiffSide.RIGHT,
        from_line=1,
        to_line=1,
        evidence_refs=(),
        suggested_action="fix the branch",
    )


def _completed_submission(config, trial_index: int, finding_count: int = 1, *, usage=None):
    return EvalSubmission(
        schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        task_id=config.suite.cases[0].task_id,
        agent_id=config.agent.agent_id,
        trial_id=config.trial_id(config.suite.cases[0].task_id, trial_index),
        eval_input_digest=config.suite.cases[0].eval_input_digest,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        status=SubmissionStatus.COMPLETED,
        intent=SubmissionIntent(
            status=IntentResult.SUFFICIENT,
            goal=GOAL,
            acceptance_criteria=(),
            scope=(),
            constraints=(),
            claims=(),
            clarification_questions=(),
            uncertainties=(),
        ),
        review=SubmissionReview(
            findings=tuple(_finding(f"finding-{trial_index}-{index}") for index in range(1, finding_count + 1)),
            uncertainties=(),
        ),
        evidence=(),
        usage=usage
        or SubmissionUsage(
            elapsed_seconds=2,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            tool_calls=1,
            cost_amount=None,
            cost_currency=None,
        ),
        trace_ref=None,
        failure=None,
    )


def _failed_submission(config, trial_index: int, *, usage=None):
    return EvalSubmission(
        schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        task_id=config.suite.cases[0].task_id,
        agent_id=config.agent.agent_id,
        trial_id=config.trial_id(config.suite.cases[0].task_id, trial_index),
        eval_input_digest=config.suite.cases[0].eval_input_digest,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        status=SubmissionStatus.FAILED,
        intent=None,
        review=None,
        evidence=(),
        usage=usage
        or SubmissionUsage(
            elapsed_seconds=1,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            tool_calls=None,
            cost_amount=None,
            cost_currency=None,
        ),
        trace_ref=None,
        failure=SubmissionFailure(
            code=FailureCode.TIMEOUT,
            message="timed out",
            retryable=True,
        ),
    )


def _partial_failed_submission(config, trial_index: int, *, usage=None):
    return replace(
        _completed_submission(config, trial_index, usage=usage),
        status=SubmissionStatus.FAILED,
        failure=SubmissionFailure(
            code=FailureCode.TIMEOUT,
            message="timed out after producing a partial result",
            retryable=True,
        ),
    )


def _score_sources(
    *,
    with_location: bool = False,
    intent_scorable: bool = True,
    trial_count: int = 1,
    review_findings=None,
):
    case, snapshot, replay = _case_and_snapshot(
        with_location=with_location,
        intent_scorable=intent_scorable,
        review_findings=review_findings,
    )
    execution = _execution()
    config = run_config(snapshot, evaluator=execution.evaluator, trial_count=trial_count)
    return case, replay, execution, config


def _score_completed(case, replay, execution, config, trial_index: int, finding_count: int = 1, *, scorer=None, usage=None):
    submission = _completed_submission(config, trial_index, finding_count, usage=usage)
    intent_result = IntentEvaluator().evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    review_result = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(submission, case.review_truth)
    scorer = scorer or TrialScorer()
    score = scorer.score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=trial_index,
        intent_result=intent_result,
        review_result=review_result,
    )
    return submission, intent_result, review_result, score


def test_v1_severity_and_line_policies_are_fixed_and_digest_bound() -> None:
    assert {
        item.severity.value: item.weight
        for item in DEFAULT_SEVERITY_WEIGHT_POLICY.weights
    } == {"critical": 8, "high": 4, "low": 1, "medium": 2}
    assert DEFAULT_SEVERITY_WEIGHT_POLICY.digest == canonical_sha256(
        DEFAULT_SEVERITY_WEIGHT_POLICY._identity_dict()
    )
    assert DEFAULT_LINE_METRIC_POLICY.version == "assigned-truth-location-v1"
    assert DEFAULT_METRICS_POLICY.failure_outcome_policy is FailureOutcomePolicy.COUNT_AS_MISSED

    with pytest.raises(ValueError, match="fixed 1/2/4/8"):
        SeverityWeightPolicy.create(
            {
                FindingSeverity.LOW: 10,
                FindingSeverity.MEDIUM: 20,
                FindingSeverity.HIGH: 40,
                FindingSeverity.CRITICAL: 80,
            }
        )
    altered_line_identity = {
        "version": "assigned-truth-location-v1",
        "precision_rule": "any-candidate-location",
        "recall_rule": "any-candidate-location",
    }
    with pytest.raises(ValueError, match="fixed line rules"):
        LineMetricPolicy(
            version=altered_line_identity["version"],
            precision_rule=altered_line_identity["precision_rule"],
            recall_rule=altered_line_identity["recall_rule"],
            digest=canonical_sha256(altered_line_identity),
        )


def test_trial_score_binds_sources_and_exposes_unscorable_authority_as_null() -> None:
    case, replay, execution, config = _score_sources()
    submission, intent_result, review_result, score = _score_completed(
        case, replay, execution, config, 1
    )

    assert score.contribution(CoreMetric.INTENT_CLAIM_PRECISION).numerator == 1
    assert score.contribution(CoreMetric.ISSUE_PRECISION).denominator == 1
    assert score.contribution(CoreMetric.SEVERITY_WEIGHTED_RECALL).numerator == 4
    assert score.contribution(CoreMetric.SEVERITY_WEIGHTED_RECALL).denominator == 4
    assert score.contribution(CoreMetric.LINE_RECALL).source_status is MetricSourceStatus.NOT_SCORABLE
    assert score.contribution(CoreMetric.LINE_RECALL).numerator is None
    assert score.contribution(CoreMetric.LINE_RECALL).denominator is None
    assert score.contribution(CoreMetric.EVIDENCE_VALIDITY).numerator == 0
    assert score.contribution(CoreMetric.PUBLISHABLE_FINDING_PRECISION).numerator == 0
    assert score.compatibility.metrics_policy == DEFAULT_METRICS_POLICY
    compatibility = score.to_dict()["compatibility"]
    assert compatibility["target_kind"] == "repository"
    assert compatibility["wire_contract"] == config.wire_contract.to_dict()
    assert compatibility["wire_contract_digest"] == canonical_sha256(
        config.wire_contract.to_dict()
    )
    assert (
        compatibility["adapter_capabilities_digest"]
        == config.adapter_capabilities_digest
    )
    assert (
        compatibility["isolation_profile"]
        == config.adapter_capabilities.isolation_profile
    )
    assert compatibility["metric_authority_profile"] == {
        "authorities": [
            {
                "severity_scorable": True,
                "severity_authority": "expert_annotation",
                "location_scorable": False,
                "location_authority": None,
            }
        ]
    }
    assert compatibility["metric_authority_profile_digest"] == canonical_sha256(
        compatibility["metric_authority_profile"]
    )
    assert (
        compatibility["metric_authority_policy_version"]
        == execution.metric_authority_policy_version
    )
    assert len(compatibility["metric_authority_policy_digest"]) == 64
    assert score.to_dict()["authority_coverage"] == {
        "expected_truth_count": 1,
        "required_expected_truth_count": 1,
        "severity_eligible_required_truth_count": 1,
        "severity_excluded_required_truth_count": 0,
        "location_precision_eligible_truth_count": 0,
        "location_precision_excluded_truth_count": 1,
        "location_recall_eligible_required_truth_count": 0,
        "location_recall_excluded_required_truth_count": 1,
    }
    assert score.to_dict()["compatibility"]["metrics_policy"]["severity_weights"]["weights"] == [
        {"severity": "critical", "weight": 8},
        {"severity": "high", "weight": 4},
        {"severity": "low", "weight": 1},
        {"severity": "medium", "weight": 2},
    ]

    with pytest.raises(TypeError, match="TrialScorer"):
        replace(score, submission_digest="0" * 64)

    hydrated = TrialScore.from_json(
        score.to_json(),
        scorer=TrialScorer(),
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=intent_result,
        review_result=review_result,
    )
    assert hydrated == score

    case_score = MetricsAggregator().aggregate_case((score,), planned_trial_count=1)
    line = case_score.metric(CoreMetric.LINE_RECALL)
    assert line.numerator is None
    assert line.denominator is None
    assert line.value_ppm is None
    assert line.null_reason is MetricNullReason.NOT_SCORABLE
    assert line.coverage.not_scorable_count == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("target_kind", "frozen_context"),
        ("wire_contract_digest", "0" * 64),
        ("adapter_capabilities_digest", "1" * 64),
        ("isolation_profile", "forged-isolation-v9"),
        ("metric_authority_profile_digest", "2" * 64),
        ("metric_authority_policy_version", "forged-authority-v9"),
        ("metric_authority_policy_digest", "3" * 64),
    ),
)
def test_trial_score_hydration_rejects_forged_v2_compatibility_binding(
    field: str,
    replacement: str,
) -> None:
    case, replay, execution, config = _score_sources()
    submission, intent_result, review_result, score = _score_completed(
        case, replay, execution, config, 1
    )
    forged = deepcopy(score.to_dict())
    forged["compatibility"][field] = replacement

    with pytest.raises(ValueError, match="source-bound replay"):
        TrialScore.from_dict(
            forged,
            scorer=TrialScorer(),
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_case=case,
            submission=submission,
            trial_index=1,
            intent_result=intent_result,
            review_result=review_result,
        )


@pytest.mark.parametrize(
    "path",
    ("wire_contract", "metric_authority_profile", "authority_coverage"),
)
def test_trial_score_hydration_rejects_forged_v2_snapshot_or_coverage(
    path: str,
) -> None:
    case, replay, execution, config = _score_sources()
    submission, intent_result, review_result, score = _score_completed(
        case, replay, execution, config, 1
    )
    forged = deepcopy(score.to_dict())
    if path == "wire_contract":
        forged["compatibility"][path]["materializer_protocol"] = (
            "frozen-context-materializer-v2"
        )
    elif path == "metric_authority_profile":
        forged["compatibility"][path]["authorities"] = []
    else:
        forged[path]["expected_truth_count"] = 0

    with pytest.raises(ValueError, match="source-bound replay"):
        TrialScore.from_dict(
            forged,
            scorer=TrialScorer(),
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_case=case,
            submission=submission,
            trial_index=1,
            intent_result=intent_result,
            review_result=review_result,
        )


def test_line_metrics_use_only_the_final_assignment_truth_location() -> None:
    case, replay, execution, config = _score_sources(with_location=True)
    _, _, _, score = _score_completed(case, replay, execution, config, 1)

    assert score.contribution(CoreMetric.LINE_PRECISION).numerator == 1
    assert score.contribution(CoreMetric.LINE_PRECISION).denominator == 1
    assert score.contribution(CoreMetric.LINE_RECALL).numerator == 1
    assert score.contribution(CoreMetric.LINE_RECALL).denominator == 1


def test_line_metrics_ignore_a_matching_location_on_an_unassigned_truth() -> None:
    assigned_truth = ExpectedFinding(
        truth_id="truth-assigned",
        claim=CLAIM,
        severity=FindingSeverity.HIGH,
        category="correctness",
        required=True,
        metric_authority=_core_authority(),
        locations=(
            TruthLocation(
                path="src/app.py",
                side=DiffSide.RIGHT,
                from_line=10,
                to_line=10,
            ),
        ),
        evidence_anchors=(),
        required_context_level=RequiredContextLevel.DIFF,
        rationale="semantic assignment target with a different location",
    )
    unassigned_truth = ExpectedFinding(
        truth_id="truth-location-decoy",
        claim="A distinct optional issue that is not the submitted defect.",
        severity=FindingSeverity.MEDIUM,
        category="correctness",
        required=False,
        metric_authority=_core_authority(),
        locations=(
            TruthLocation(
                path="src/app.py",
                side=DiffSide.RIGHT,
                from_line=1,
                to_line=1,
            ),
        ),
        evidence_anchors=(),
        required_context_level=RequiredContextLevel.DIFF,
        rationale="location-only decoy",
    )
    case, replay, execution, config = _score_sources(
        review_findings=(assigned_truth, unassigned_truth)
    )
    submission = _completed_submission(config, 1)
    intent_result = IntentEvaluator().evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    evaluator = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    )
    pending = evaluator.evaluate(submission, case.review_truth)
    assert len(pending.judge_requests) == 1
    decision = _run_scripted_judge(
        pending.judge_requests[0],
        execution,
        relation="different",
        score_ppm=0,
        severity_assessment="consistent",
        actionability="actionable",
    )
    review_result = evaluator.evaluate(
        submission,
        case.review_truth,
        judge_results=(decision,),
    )
    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=intent_result,
        review_result=review_result,
    )

    assert review_result.assignments[0].truth_id == "truth-assigned"
    assert any(
        item.truth_id == "truth-location-decoy" and item.match.matched
        for item in review_result.location_candidates
    )
    assert score.contribution(CoreMetric.LINE_PRECISION).numerator == 0
    assert score.contribution(CoreMetric.LINE_PRECISION).denominator == 1
    assert score.contribution(CoreMetric.LINE_RECALL).numerator == 0
    assert score.contribution(CoreMetric.LINE_RECALL).denominator == 1


def test_case_aggregation_uses_ratio_of_sums_not_average_of_percentages() -> None:
    case, replay, execution, config = _score_sources(trial_count=2)
    _, _, _, first = _score_completed(case, replay, execution, config, 1, finding_count=1)
    _, _, _, second = _score_completed(case, replay, execution, config, 2, finding_count=2)

    aggregate = MetricsAggregator().aggregate_case((first, second), planned_trial_count=2)
    precision = aggregate.metric(CoreMetric.ISSUE_PRECISION)

    assert precision.numerator == 2
    assert precision.denominator == 3
    assert precision.value_ppm == 666_667
    assert aggregate.metric(CoreMetric.ISSUE_RECALL).value_ppm == 1_000_000
    assert aggregate.metric(CoreMetric.PLAUSIBLE_RATE).numerator == 1
    assert first.authority_coverage == second.authority_coverage
    assert aggregate.authority_coverage == first.authority_coverage

    hydrated_case = CaseScore.from_json(
        aggregate.to_json(),
        aggregator=MetricsAggregator(),
        trials=(first, second),
        planned_trial_count=2,
    )
    assert hydrated_case == aggregate
    overall = MetricsAggregator().aggregate_cases(
        (aggregate,),
        source_trials=(first, second),
    )
    assert overall.authority_coverage == aggregate.authority_coverage
    hydrated_overall = AggregateScore.from_json(
        overall.to_json(),
        aggregator=MetricsAggregator(),
        case_scores=(aggregate,),
        source_trials=(first, second),
    )
    assert hydrated_overall == overall
    groups = MetricsAggregator().group_case_scores(
        (aggregate,),
        (first, second),
        dimension_names=("language",),
    )
    assert len(groups) == 1
    assert groups[0].group_dimensions[0].to_dict() == {
        "name": "language",
        "value": "python",
    }


def test_failed_trial_counts_in_failure_rate_and_recall_by_explicit_policy() -> None:
    case, replay, execution, config = _score_sources(trial_count=2)
    _, _, _, passed = _score_completed(case, replay, execution, config, 1)
    failed_submission = _failed_submission(config, 2)
    failed = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=failed_submission,
        trial_index=2,
        intent_result=None,
        review_result=None,
    )

    assert failed.contribution(CoreMetric.ISSUE_RECALL).source_status is MetricSourceStatus.FAILURE_AS_MISS
    assert failed.contribution(CoreMetric.ISSUE_RECALL).denominator == 1
    assert failed.contribution(CoreMetric.ISSUE_PRECISION).source_status is MetricSourceStatus.FAILURE_EXCLUDED
    assert failed.contribution(CoreMetric.INTENT_CLAIM_RECALL).source_status is MetricSourceStatus.FAILURE_AS_MISS
    assert failed.contribution(CoreMetric.INTENT_CLAIM_PRECISION).source_status is MetricSourceStatus.FAILURE_EXCLUDED

    aggregate = MetricsAggregator().aggregate_case((passed, failed), planned_trial_count=2)
    assert aggregate.metric(CoreMetric.AGENT_FAILURE_RATE).numerator == 1
    assert aggregate.metric(CoreMetric.AGENT_FAILURE_RATE).denominator == 2
    assert aggregate.metric(CoreMetric.ISSUE_RECALL).numerator == 1
    assert aggregate.metric(CoreMetric.ISSUE_RECALL).denominator == 2
    assert aggregate.metric(CoreMetric.ISSUE_PRECISION).coverage.failure_excluded_count == 1
    assert failed.trial_id in aggregate.failed_trial_ids
    assert failed.trial_id in aggregate.critical_high_miss_trial_ids
    f1 = aggregate.metric(CoreMetric.ISSUE_F1)
    assert f1.coverage is None
    derived = {item.metric: item.coverage for item in f1.derived_coverages}
    assert derived[CoreMetric.ISSUE_PRECISION].failure_excluded_count == 1
    assert derived[CoreMetric.ISSUE_RECALL].failure_as_miss_count == 1


def test_failed_trial_scores_non_null_partial_review_and_failure_independently() -> None:
    case, replay, execution, config = _score_sources()
    submission = _partial_failed_submission(config, 1)
    intent_result = IntentEvaluator().evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    review_result = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(submission, case.review_truth)

    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=intent_result,
        review_result=review_result,
    )
    aggregate = MetricsAggregator().aggregate_case((score,))

    assert score.contribution(CoreMetric.AGENT_FAILURE_RATE).numerator == 1
    assert score.contribution(CoreMetric.ISSUE_RECALL).source_status is MetricSourceStatus.GRADED
    assert aggregate.metric(CoreMetric.ISSUE_RECALL).value_ppm == 1_000_000
    assert aggregate.intent_scored_trial_count == 1
    assert aggregate.review_scored_trial_count == 1
    assert aggregate.fully_scored_trial_count == 1
    assert score.trial_id in aggregate.failed_trial_ids
    assert score.trial_id not in aggregate.critical_high_miss_trial_ids


def test_partial_submission_parts_are_scored_independently_when_one_result_is_missing() -> None:
    case, replay, execution, config = _score_sources()
    submission = _partial_failed_submission(config, 1)
    review_result = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(submission, case.review_truth)

    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=None,
        review_result=review_result,
    )
    assert score.contribution(CoreMetric.ISSUE_RECALL).source_status is MetricSourceStatus.GRADED
    assert score.contribution(CoreMetric.INTENT_CLAIM_RECALL).source_status is MetricSourceStatus.MISSING

    intent_absent_submission = replace(submission, intent=None)
    intent_absent_review = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=intent_absent_submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(intent_absent_submission, case.review_truth)
    intent_absent_score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=intent_absent_submission,
        trial_index=1,
        intent_result=None,
        review_result=intent_absent_review,
    )
    assert intent_absent_score.contribution(CoreMetric.ISSUE_RECALL).source_status is MetricSourceStatus.GRADED
    assert intent_absent_score.contribution(CoreMetric.INTENT_CLAIM_RECALL).source_status is MetricSourceStatus.FAILURE_AS_MISS

    completed_submission = _completed_submission(config, 1)
    completed_review = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=completed_submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(completed_submission, case.review_truth)
    completed_partial_score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=completed_submission,
        trial_index=1,
        intent_result=None,
        review_result=completed_review,
    )
    assert completed_partial_score.contribution(CoreMetric.AGENT_FAILURE_RATE).numerator == 0
    assert completed_partial_score.contribution(CoreMetric.ISSUE_RECALL).source_status is MetricSourceStatus.GRADED
    assert completed_partial_score.contribution(CoreMetric.INTENT_CLAIM_RECALL).source_status is MetricSourceStatus.MISSING


def test_failure_exclusion_policy_is_visible_and_incompatible_with_count_as_missed() -> None:
    case, replay, execution, config = _score_sources(trial_count=2)
    failed_submission = _failed_submission(config, 1)
    excluded_scorer = TrialScorer(
        MetricsPolicy.create(FailureOutcomePolicy.EXCLUDE_WITH_COVERAGE)
    )
    excluded = excluded_scorer.score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=failed_submission,
        trial_index=1,
        intent_result=None,
        review_result=None,
    )
    assert excluded.contribution(CoreMetric.ISSUE_RECALL).source_status is MetricSourceStatus.FAILURE_EXCLUDED

    _, _, _, included = _score_completed(case, replay, execution, config, 2)
    with pytest.raises(ValueError, match="incompatible"):
        MetricsAggregator().aggregate_case((excluded, included), planned_trial_count=2)


def test_usage_coverage_does_not_replace_missing_values_with_zero() -> None:
    case, replay, execution, config = _score_sources(trial_count=2)
    _, _, _, observed = _score_completed(case, replay, execution, config, 1)
    missing_submission = _failed_submission(config, 2)
    missing = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=missing_submission,
        trial_index=2,
        intent_result=None,
        review_result=None,
    )
    aggregate = MetricsAggregator().aggregate_case((observed, missing), planned_trial_count=2)

    assert aggregate.usage.total_tokens.sum_value == 15
    assert aggregate.usage.total_tokens.observed_count == 1
    assert aggregate.usage.total_tokens.missing_count == 1
    assert aggregate.usage.total_tokens.mean_value == 15
    assert aggregate.usage.cost.sum_value is None
    assert aggregate.usage.cost.observed_count == 0
    assert aggregate.usage.cost.population_count == 2
    assert aggregate.usage.cost.missing_count == 2
    assert aggregate.usage.cost_currency is None


def test_unscorable_intent_is_null_not_zero_or_one_hundred_percent() -> None:
    case, replay, execution, config = _score_sources(intent_scorable=False)
    _, _, _, score = _score_completed(case, replay, execution, config, 1)
    contribution = score.contribution(CoreMetric.INTENT_CLAIM_PRECISION)
    assert contribution.source_status is MetricSourceStatus.NOT_SCORABLE
    assert contribution.numerator is None
    aggregate = MetricsAggregator().aggregate_case((score,), planned_trial_count=1)
    metric = aggregate.metric(CoreMetric.INTENT_CLAIM_PRECISION)
    assert metric.numerator is None
    assert metric.denominator is None
    assert metric.null_reason is MetricNullReason.NOT_SCORABLE


def test_ungraded_review_does_not_enter_quality_ratio() -> None:
    case, replay, execution, config = _score_sources()
    submission = _completed_submission(config, 1)
    submission = replace(
        submission,
        review=SubmissionReview(
            findings=(
                replace(
                    submission.review.findings[0],
                    claim="A partially overlapping defect claim.",
                ),
            ),
            uncertainties=(),
        ),
    )
    intent_result = IntentEvaluator().evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    evaluator = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    )
    pending = evaluator.evaluate(submission, case.review_truth)
    decision = _run_scripted_judge(
        pending.judge_requests[0],
        execution,
        relation="partially_equivalent",
        score_ppm=800_000,
        severity_assessment="consistent",
        actionability="actionable",
    )
    review_result = evaluator.evaluate(
        submission,
        case.review_truth,
        judge_results=(decision,),
    )
    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=intent_result,
        review_result=review_result,
    )

    assert review_result.status.value == "ungraded"
    assert score.contribution(CoreMetric.ISSUE_PRECISION).source_status is MetricSourceStatus.UNGRADED
    aggregate = MetricsAggregator().aggregate_case((score,), planned_trial_count=1)
    assert aggregate.metric(CoreMetric.ISSUE_PRECISION).null_reason is MetricNullReason.UNGRADED
    assert aggregate.intent_scored_trial_count == 1
    assert aggregate.review_scored_trial_count == 0
    assert aggregate.fully_scored_trial_count == 0

    failed_submission = replace(
        submission,
        status=SubmissionStatus.FAILED,
        failure=SubmissionFailure(
            code=FailureCode.TIMEOUT,
            message="timed out after partial output",
            retryable=True,
        ),
    )
    failed_intent_result = IntentEvaluator().evaluate(
        failed_submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    failed_evaluator = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=failed_submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    )
    failed_pending = failed_evaluator.evaluate(failed_submission, case.review_truth)
    failed_decision = _run_scripted_judge(
        failed_pending.judge_requests[0],
        execution,
        relation="partially_equivalent",
        score_ppm=800_000,
        severity_assessment="consistent",
        actionability="actionable",
    )
    failed_review_result = failed_evaluator.evaluate(
        failed_submission,
        case.review_truth,
        judge_results=(failed_decision,),
    )
    failed_score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=failed_submission,
        trial_index=1,
        intent_result=failed_intent_result,
        review_result=failed_review_result,
    )
    assert failed_score.contribution(CoreMetric.ISSUE_RECALL).source_status is MetricSourceStatus.UNGRADED
    assert failed_score.contribution(CoreMetric.AGENT_FAILURE_RATE).numerator == 1


def test_intent_and_review_scored_coverage_are_reported_independently() -> None:
    case, replay, execution, config = _score_sources()
    submission = _completed_submission(config, 1)
    submission = replace(
        submission,
        intent=replace(
            submission.intent,
            goal="Introduce an unrelated preview workflow.",
        ),
    )
    intent_evaluator = IntentEvaluator()
    pending_intent = intent_evaluator.evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    assert pending_intent.judge_requests
    intent_result = intent_evaluator.evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
        semantic_failures=(
            judge_failure(
                pending_intent.judge_requests[0],
                evaluator_execution_digest=execution.digest(),
            ),
        ),
    )
    review_result = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(submission, case.review_truth)
    score = TrialScorer().score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=1,
        intent_result=intent_result,
        review_result=review_result,
    )
    aggregate = MetricsAggregator().aggregate_case((score,))

    assert intent_result.status.value == "ungraded"
    assert review_result.status.value == "graded"
    assert aggregate.intent_scored_trial_count == 0
    assert aggregate.review_scored_trial_count == 1
    assert aggregate.fully_scored_trial_count == 0


def test_costs_with_different_currencies_cannot_be_aggregated() -> None:
    case, replay, execution, config = _score_sources(trial_count=2)
    cny = SubmissionUsage(
        elapsed_seconds=1,
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        tool_calls=None,
        cost_amount=1.0,
        cost_currency="CNY",
    )
    usd = replace(cny, cost_currency="USD")
    _, _, _, first = _score_completed(
        case, replay, execution, config, 1, usage=cny
    )
    _, _, _, second = _score_completed(
        case, replay, execution, config, 2, usage=usd
    )

    with pytest.raises(ValueError, match="currencies"):
        MetricsAggregator().aggregate_case((first, second), planned_trial_count=2)


def test_trial_score_hydration_rejects_a_forged_metric_or_source_digest() -> None:
    case, replay, execution, config = _score_sources()
    submission, intent_result, review_result, score = _score_completed(
        case, replay, execution, config, 1
    )
    forged = score.to_dict()
    forged["submission_digest"] = "0" * 64
    forged["contributions"][0]["numerator"] = 0

    with pytest.raises(ValueError, match="source-bound replay"):
        TrialScore.from_dict(
            forged,
            scorer=TrialScorer(),
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_case=case,
            submission=submission,
            trial_index=1,
            intent_result=intent_result,
            review_result=review_result,
        )


def test_case_and_aggregate_hydration_are_source_bound_and_sealed() -> None:
    case, replay, execution, config = _score_sources()
    _, _, _, score = _score_completed(case, replay, execution, config, 1)
    aggregator = MetricsAggregator()
    case_score = aggregator.aggregate_case((score,))
    aggregate = aggregator.aggregate_cases(
        (case_score,),
        source_trials=(score,),
    )

    with pytest.raises(TypeError, match="MetricsAggregator"):
        replace(case_score, planned_trial_count=2)
    with pytest.raises(TypeError, match="MetricsAggregator"):
        replace(aggregate, planned_trial_count=2)

    forged_case = case_score.to_dict()
    forged_case["planned_trial_count"] = 2
    with pytest.raises(ValueError, match="source-bound replay"):
        CaseScore.from_dict(
            forged_case,
            aggregator=aggregator,
            trials=(score,),
        )

    forged_aggregate = aggregate.to_dict()
    forged_aggregate["terminal_trial_count"] = 0
    with pytest.raises(ValueError, match="source-bound replay"):
        AggregateScore.from_dict(
            forged_aggregate,
            aggregator=aggregator,
            case_scores=(case_score,),
            source_trials=(score,),
        )


def test_planned_terminal_and_scored_coverage_remain_distinct() -> None:
    case, replay, execution, config = _score_sources(trial_count=3)
    _, _, _, score = _score_completed(case, replay, execution, config, 1)

    case_score = MetricsAggregator().aggregate_case((score,))

    assert case_score.planned_trial_count == 3
    assert case_score.terminal_trial_count == 1
    assert case_score.intent_scored_trial_count == 1
    assert case_score.review_scored_trial_count == 1
    assert case_score.fully_scored_trial_count == 1
    assert case_score.metric(CoreMetric.ISSUE_RECALL).coverage.total_trial_count == 1


def test_aggregate_rejects_source_trials_that_differ_from_case_score_refs() -> None:
    case, replay, execution, config = _score_sources()
    _, _, _, original = _score_completed(case, replay, execution, config, 1)
    case_score = MetricsAggregator().aggregate_case((original,))
    changed_usage = SubmissionUsage(
        elapsed_seconds=9,
        input_tokens=90,
        output_tokens=10,
        total_tokens=100,
        tool_calls=7,
        cost_amount=None,
        cost_currency=None,
    )
    _, _, _, changed = _score_completed(
        case,
        replay,
        execution,
        config,
        1,
        usage=changed_usage,
    )

    assert changed.trial_id == original.trial_id
    assert changed.compatibility == original.compatibility
    with pytest.raises(ValueError, match="digest differs"):
        MetricsAggregator().aggregate_cases(
            (case_score,),
            source_trials=(changed,),
        )


def test_evaluator_revision_is_part_of_score_compatibility() -> None:
    case, replay, execution, config = _score_sources(trial_count=2)
    _, _, _, first = _score_completed(case, replay, execution, config, 1)
    submission = _completed_submission(config, 2)
    intent_result = IntentEvaluator(
        evaluator_revision="intent-evaluator-v2"
    ).evaluate(
        submission.intent,
        case.intent_truth,
        case.clarification_script,
    )
    review_result = ReviewEvaluator(
        eval_input=case.eval_input(),
        replay=replay,
        trial_id=submission.trial_id,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
        evaluator_execution=execution,
    ).evaluate(submission, case.review_truth)

    with pytest.raises(ValueError, match="Intent result is not bound"):
        TrialScorer().score(
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_case=case,
            submission=submission,
            trial_index=2,
            intent_result=intent_result,
            review_result=review_result,
        )

    second = TrialScorer(
        intent_evaluator_revision="intent-evaluator-v2"
    ).score(
        run_config=config,
        evaluator_execution=execution,
        evaluation_revision="metrics-eval-v1",
        eval_case=case,
        submission=submission,
        trial_index=2,
        intent_result=intent_result,
        review_result=review_result,
    )
    assert second.compatibility.intent_evaluator_revision == "intent-evaluator-v2"
    with pytest.raises(ValueError, match="incompatible"):
        MetricsAggregator().aggregate_case((first, second))


def test_aggregate_group_dimensions_must_be_a_true_common_case_projection() -> None:
    case, replay, execution, config = _score_sources()
    _, _, _, score = _score_completed(case, replay, execution, config, 1)
    aggregator = MetricsAggregator()
    case_score = aggregator.aggregate_case((score,))

    with pytest.raises(ValueError, match="common Case projection"):
        aggregator.aggregate_cases(
            (case_score,),
            group_dimensions=(CaseDimension("language", "java"),),
            source_trials=(score,),
        )

    valid = aggregator.aggregate_cases(
        (case_score,),
        group_dimensions=(CaseDimension("language", "python"),),
        source_trials=(score,),
    )
    assert valid.group_dimensions == (CaseDimension("language", "python"),)

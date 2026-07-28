from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from review_agent_eval.config import EvalRunConfig
from review_agent_eval.intent_evaluator import IntentEvaluator, IntentJudgeRelation
from review_agent_eval.judge import JudgeUngradedReason, SemanticJudge
from review_agent_eval.metrics import (
    CoreMetric,
    MetricAggregate,
    MetricCoverage,
    MetricKind,
    MetricNullReason,
    MetricsAggregator,
    TrialScore,
    TrialScorer,
)
from review_agent_eval.models import (
    ClarificationPolicy,
    ExpectedIntentClaim,
    IntentAuthority,
    IntentClaimSource,
    IntentDimension,
    IntentTruth,
    SubmissionIntentClaim,
    ReviewTruth,
    TruthCompleteness,
    NovelFindingPolicy,
    canonical_sha256,
    stable_id,
)
from review_agent_eval.review_evaluator import ReviewEvaluator
import review_agent_eval.statistics as statistics_module
from review_agent_eval.statistics import (
    STATISTICS_ALGORITHM_VERSION,
    ConfidenceIntervalV1,
    ConfidenceIntervalStatus,
    DerivedCaseContributionV1,
    MetricDirection,
    MetricSourceCoverageV1,
    RunStatisticsV1,
    StatisticsError,
    StatisticsMetricV1,
    StatisticsMetricStatus,
    StatisticsPolicyV1,
    TrialMetricProjectionV1,
    compute_run_statistics,
    paired_bootstrap_interval,
)

from .test_intent_evaluator import judge_decision, judge_failure
from .test_judge import _Factory, _execution
from .test_metrics import (
    TARGET_MATERIALIZATION_ID,
    _completed_submission,
    _failed_submission,
    _score_completed,
    _score_sources,
)
from .test_orchestrator_target_replay_v2 import (
    _FrozenRun,
    _FrozenSuccessAdapter,
    _case_bank,
    _frozen_orchestrator,
    _frozen_runner,
    _frozen_snapshot_and_case,
    _prepared_bundle,
    _run_config,
)


POLICY = StatisticsPolicyV1(
    algorithm_version=STATISTICS_ALGORITHM_VERSION,
    bootstrap_seed=1729,
    bootstrap_iterations=256,
    confidence_level_ppm=950_000,
)


class _JudgeRateAdapter(_FrozenSuccessAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.slot = 0

    def run(self, *args: Any, **kwargs: Any):
        self.slot += 1
        submission = super().run(*args, **kwargs)
        assert submission.intent is not None
        claim_count = 2 if self.slot == 1 else 100
        claims = tuple(
            SubmissionIntentClaim(
                claim_id=f"candidate-{index:03d}",
                dimension=IntentDimension.SCOPE,
                text=f"Candidate scope {index:03d}",
                source=IntentClaimSource.EXPLICIT,
            )
            for index in range(1, claim_count + 1)
        )
        return replace(
            submission,
            intent=replace(submission.intent, goal=None, claims=claims),
        )


class _JudgeFailureRatio:
    def __init__(self, execution: Any) -> None:
        self.execution = execution
        self.calls = 0

    def execute(self, request: Any):
        self.calls += 1
        failure = self.calls == 1 or 3 <= self.calls <= 101
        judge = SemanticJudge(
            adapter_factory=_Factory(
                [TimeoutError("judge attempt one"), TimeoutError("judge attempt two")]
                if failure
                else []
            ),
            evaluator_execution=self.execution,
        )
        if failure:
            return judge.execute(request)
        return judge.execute(
            request,
            ungraded_reason=JudgeUngradedReason.POLICY_SKIPPED,
        )


def _repeated_bundle(
    root: Path,
    *,
    adapter: Any,
    trial_count: int,
    instance: str,
    intent_truth: IntentTruth,
    review_truth: ReviewTruth,
    judge: Any,
    execution: Any = None,
):
    prepared = _prepared_bundle(root)
    snapshot, case = _frozen_snapshot_and_case(
        prepared,
        intent_truth=intent_truth,
        review_truth=review_truth,
    )
    base = _run_config(snapshot, current=False, instance=instance)
    config = EvalRunConfig.create(
        run_instance_key=instance,
        agent=base.agent,
        clarification_matcher=base.clarification_matcher,
        evaluator=base.evaluator,
        suite=base.suite,
        adapter_capabilities=base.adapter_capabilities,
        trial_count=trial_count,
        resource_budgets=base.resource_budgets,
    )
    runner = _frozen_runner(root, prepared, adapter)
    run_result = runner.run(config, snapshot)
    run = _FrozenRun(
        prepared=prepared,
        snapshot=snapshot,
        case=case,
        config=config,
        runner=runner,
        trial=run_result.trials[0],
        bank=_case_bank(root, snapshot, case),
    )
    orchestrator = _frozen_orchestrator(run, judge=judge)
    execution = execution or _execution()
    evaluated = orchestrator.evaluate_run(
        config.run_id,
        evaluator_execution=execution,
        evaluation_revision="statistics-fixture-v1",
    )
    return run, evaluated


@pytest.fixture(scope="module")
def ratio_source(tmp_path_factory: pytest.TempPathFactory):
    truth = IntentTruth(
        scorable=True,
        authority=IntentAuthority.EXPLICIT_AUTHOR_METADATA,
        expected_claims=(
            ExpectedIntentClaim(
                truth_id="expected-scope",
                dimension=IntentDimension.SCOPE,
                text="Expected canonical scope",
                required=True,
            ),
        ),
        forbidden_claims=(),
        clarification_policy=ClarificationPolicy.NOT_REQUIRED,
    )
    review_truth = ReviewTruth(
        completeness=TruthCompleteness.HUMAN_OBSERVED,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
        expected_findings=(),
        known_invalid_findings=(),
    )
    execution = _execution()
    judge = _JudgeFailureRatio(execution)
    run, bundle = _repeated_bundle(
        tmp_path_factory.mktemp("statistics-ratio"),
        adapter=_JudgeRateAdapter(),
        trial_count=2,
        instance="statistics-ratio",
        intent_truth=truth,
        review_truth=review_truth,
        judge=judge,
        execution=execution,
    )
    assert judge.calls == 102
    return run, bundle


@pytest.fixture(scope="module")
def ratio_statistics(ratio_source: Any):
    run, bundle = ratio_source
    return compute_run_statistics(
        bundle,
        run_config=run.config,
        case_snapshot=run.snapshot,
        policy=POLICY,
    )


@pytest.fixture(scope="module")
def coverage_projection():
    case, replay, execution, config = _score_sources(trial_count=4)
    _, _, _, completed = _score_completed(case, replay, execution, config, 1)

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

    def judged_score(index: int, *, unknown: bool):
        submission = _completed_submission(config, index)
        assert submission.intent is not None
        submission = replace(
            submission,
            intent=replace(
                submission.intent,
                goal=f"Semantically unresolved goal {index}",
            ),
        )
        evaluator = IntentEvaluator()
        pending = evaluator.evaluate(
            submission.intent,
            case.intent_truth,
            case.clarification_script,
        )
        assert len(pending.judge_requests) == 1
        request = pending.judge_requests[0]
        if unknown:
            intent_result = evaluator.evaluate(
                submission.intent,
                case.intent_truth,
                case.clarification_script,
                semantic_decisions=(
                    judge_decision(
                        request,
                        IntentJudgeRelation.UNKNOWN,
                        score_ppm=0,
                    ),
                ),
            )
        else:
            intent_result = evaluator.evaluate(
                submission.intent,
                case.intent_truth,
                case.clarification_script,
                semantic_failures=(
                    judge_failure(
                        request,
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
        return TrialScorer().score(
            run_config=config,
            evaluator_execution=execution,
            evaluation_revision="metrics-eval-v1",
            eval_case=case,
            submission=submission,
            trial_index=index,
            intent_result=intent_result,
            review_result=review_result,
        )

    unknown = judged_score(3, unknown=True)
    judge_failed = judged_score(4, unknown=False)
    scores = (completed, failed, unknown, judge_failed)
    aggregate = MetricsAggregator().aggregate_case(
        scores,
        planned_trial_count=4,
    )
    return statistics_module._statistics_coverage(
        aggregate.metric(CoreMetric.INTENT_CLAIM_RECALL),
        scores,
    )


def test_statistics_reaggregates_rates_from_numerators_and_denominators(
    ratio_statistics: Any,
) -> None:
    metric = ratio_statistics.metric(CoreMetric.JUDGE_FAILURE_RATE)

    assert metric.numerator == 100
    assert metric.denominator == 102
    assert metric.value == 980_392
    assert metric.value != (500_000 + 990_000) // 2
    assert tuple(
        ratio_statistics.trial_metric(index, CoreMetric.JUDGE_FAILURE_RATE).value
        for index in (1, 2)
    ) == (500_000, 990_000)
    assert metric.direction is MetricDirection.LOWER_IS_BETTER


def test_statistics_keeps_failed_and_ungraded_trials_in_coverage(
    coverage_projection: Any,
) -> None:
    source = coverage_projection.metric_sources[0]

    assert coverage_projection.total_trial_count == 4
    assert coverage_projection.completed_trial_count == 3
    assert coverage_projection.agent_failure_count == 1
    assert coverage_projection.judge_request_count == 2
    assert coverage_projection.judge_semantic_unknown_count == 1
    assert coverage_projection.judge_failure_count == 1
    assert source.failure_as_miss_count == 1
    assert source.ungraded_count == 2


def test_statistics_reports_each_trial_index_without_best_trial_selection(
    ratio_statistics: Any,
) -> None:
    projections = tuple(
        item
        for item in ratio_statistics.trial_metrics
        if item.metric is CoreMetric.JUDGE_FAILURE_RATE
    )

    assert tuple(item.trial_index for item in projections) == (1, 2)
    assert tuple(item.value for item in projections) == (500_000, 990_000)
    assert ratio_statistics.trial_count == 2
    assert not hasattr(ratio_statistics, "best_trial_index")


def test_statistics_trial_projection_plans_one_trial_per_case(
    ratio_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, bundle = ratio_source
    original = statistics_module._aggregate_scores
    source_policy = bundle.trials[0].trial_score.compatibility.metrics_policy
    calls = []

    def tracked(scores: Any, *, planned_trial_count: int):
        aggregate, cases, by_task = original(
            scores,
            planned_trial_count=planned_trial_count,
        )
        assert aggregate.compatibility.metrics_policy == source_policy
        assert all(
            item.compatibility.metrics_policy == source_policy for item in cases
        )
        calls.append(
            (
                len(tuple(scores)),
                planned_trial_count,
                aggregate.planned_trial_count,
                aggregate.terminal_trial_count,
                tuple(
                    (item.planned_trial_count, item.terminal_trial_count)
                    for item in cases
                ),
            )
        )
        return aggregate, cases, by_task

    monkeypatch.setattr(statistics_module, "_aggregate_scores", tracked)
    result = compute_run_statistics(
        bundle,
        run_config=run.config,
        case_snapshot=run.snapshot,
        policy=POLICY,
    )

    projection_calls = calls[: run.config.trial_count]
    assert all(
        planned == terminal == 1
        and case_counts == ((1, 1),)
        for _source_count, _requested, planned, terminal, case_counts in projection_calls
    )
    assert calls[-1][1:4] == (run.config.trial_count, 2, 2)
    assert result.metric(CoreMetric.JUDGE_FAILURE_RATE).value == 980_392
    assert tuple(
        result.trial_metric(index, CoreMetric.JUDGE_FAILURE_RATE).coverage.total_trial_count
        for index in (1, 2)
    ) == (1, 1)


def test_statistics_marks_missing_authority_not_scorable_not_zero(
    ratio_statistics: Any,
) -> None:
    metric = ratio_statistics.metric(CoreMetric.LINE_RECALL)

    assert metric.status is StatisticsMetricStatus.NOT_SCORABLE
    assert metric.numerator is None
    assert metric.denominator is None
    assert metric.value is None
    assert metric.coverage.metric_sources[0].not_scorable_count == 2


def _f1_source_aggregate(
    metric: CoreMetric,
    status: StatisticsMetricStatus,
    numerator: int | None,
    denominator: int | None,
    *,
    trial_count: int,
) -> MetricAggregate:
    unavailable_counts = {
        StatisticsMetricStatus.NOT_SCORABLE: "not_scorable_count",
        StatisticsMetricStatus.UNGRADED: "ungraded_count",
        StatisticsMetricStatus.FAILURE_EXCLUDED: "failure_excluded_count",
        StatisticsMetricStatus.MISSING: "missing_count",
    }
    coverage_counts = {
        "total_trial_count": trial_count,
        "included_trial_count": 0,
        "failure_as_miss_count": 0,
        "zero_denominator_count": 0,
        "not_scorable_count": 0,
        "ungraded_count": 0,
        "failure_excluded_count": 0,
        "missing_count": 0,
    }
    if status in {
        StatisticsMetricStatus.AVAILABLE,
        StatisticsMetricStatus.ZERO_DENOMINATOR,
    }:
        coverage_counts["included_trial_count"] = trial_count
        if status is StatisticsMetricStatus.ZERO_DENOMINATOR:
            coverage_counts["zero_denominator_count"] = trial_count
    else:
        coverage_counts[unavailable_counts[status]] = trial_count
    null_reason = {
        StatisticsMetricStatus.AVAILABLE: None,
        StatisticsMetricStatus.ZERO_DENOMINATOR: MetricNullReason.ZERO_DENOMINATOR,
        StatisticsMetricStatus.NOT_SCORABLE: MetricNullReason.NOT_SCORABLE,
        StatisticsMetricStatus.UNGRADED: MetricNullReason.UNGRADED,
        StatisticsMetricStatus.FAILURE_EXCLUDED: MetricNullReason.FAILURE_EXCLUDED,
        StatisticsMetricStatus.MISSING: MetricNullReason.MISSING,
    }[status]
    value = (
        statistics_module._ratio_ppm(numerator, denominator)
        if numerator is not None and denominator
        else None
    )
    return MetricAggregate(
        metric=metric,
        kind=MetricKind.RATE,
        numerator=numerator,
        denominator=denominator,
        value_ppm=value,
        null_reason=null_reason,
        coverage=MetricCoverage(**coverage_counts),
    )


@pytest.mark.parametrize(
    ("precision_fields", "recall_fields", "expected_status"),
    (
        (
            (StatisticsMetricStatus.ZERO_DENOMINATOR, 0, 0),
            (StatisticsMetricStatus.NOT_SCORABLE, None, None),
            StatisticsMetricStatus.ZERO_DENOMINATOR,
        ),
        (
            (StatisticsMetricStatus.NOT_SCORABLE, None, None),
            (StatisticsMetricStatus.ZERO_DENOMINATOR, 0, 0),
            StatisticsMetricStatus.NOT_SCORABLE,
        ),
        (
            (StatisticsMetricStatus.MISSING, None, None),
            (StatisticsMetricStatus.UNGRADED, None, None),
            StatisticsMetricStatus.MISSING,
        ),
        (
            (StatisticsMetricStatus.AVAILABLE, 1, 2),
            (StatisticsMetricStatus.FAILURE_EXCLUDED, None, None),
            StatisticsMetricStatus.FAILURE_EXCLUDED,
        ),
    ),
)
def test_statistics_f1_null_precedence_matches_metrics_aggregator(
    ratio_statistics: Any,
    precision_fields: tuple[StatisticsMetricStatus, int | None, int | None],
    recall_fields: tuple[StatisticsMetricStatus, int | None, int | None],
    expected_status: StatisticsMetricStatus,
) -> None:
    metric = ratio_statistics.metric(CoreMetric.ISSUE_F1)
    base_contribution = metric.case_contributions[0]
    trial_count = base_contribution.coverage.total_trial_count
    precision = _f1_source_aggregate(
        CoreMetric.ISSUE_PRECISION,
        *precision_fields,
        trial_count=trial_count,
    )
    recall = _f1_source_aggregate(
        CoreMetric.ISSUE_RECALL,
        *recall_fields,
        trial_count=trial_count,
    )
    expected = MetricsAggregator._f1(precision, recall)
    expected_value = statistics_module._metric_value(expected)
    expected_statistics_status = statistics_module._metric_status(expected)
    coverage = replace(
        base_contribution.coverage,
        metric_sources=(
            MetricSourceCoverageV1.from_metric_coverage(
                CoreMetric.ISSUE_PRECISION,
                precision.coverage,
            ),
            MetricSourceCoverageV1.from_metric_coverage(
                CoreMetric.ISSUE_RECALL,
                recall.coverage,
            ),
        ),
    )
    contribution = replace(
        base_contribution,
        status=expected_statistics_status,
        numerator=expected.numerator,
        denominator=expected.denominator,
        value=expected_value,
        coverage=coverage,
        derived_contributions=(
            DerivedCaseContributionV1(
                metric=CoreMetric.ISSUE_PRECISION,
                status=precision_fields[0],
                numerator=precision_fields[1],
                denominator=precision_fields[2],
            ),
            DerivedCaseContributionV1(
                metric=CoreMetric.ISSUE_RECALL,
                status=recall_fields[0],
                numerator=recall_fields[1],
                denominator=recall_fields[2],
            ),
        ),
    )
    aggregate = replace(
        metric,
        status=expected_statistics_status,
        numerator=expected.numerator,
        denominator=expected.denominator,
        value=expected_value,
        coverage=coverage,
        case_contributions=(contribution,),
    )

    assert expected_statistics_status is expected_status
    assert (
        contribution.numerator,
        contribution.denominator,
        contribution.value,
        contribution.status,
    ) == (
        expected.numerator,
        expected.denominator,
        expected_value,
        expected_status,
    )
    assert StatisticsMetricV1.from_dict(aggregate.to_dict()) == aggregate


def test_statistics_bootstrap_is_deterministic_for_fixed_policy(
    ratio_statistics: Any,
) -> None:
    source = ratio_statistics.metric(
        CoreMetric.JUDGE_FAILURE_RATE
    ).case_contributions[0]
    second = replace(
        source,
        task_id="bootstrap-case-2",
        canonical_case_digest="f" * 64,
        numerator=1,
        denominator=2,
        value=500_000,
    )
    first = paired_bootstrap_interval(
        (source, second),
        seed=POLICY.bootstrap_seed,
        iterations=POLICY.bootstrap_iterations,
        confidence_level_ppm=POLICY.confidence_level_ppm,
    )
    second_run = paired_bootstrap_interval(
        (source, second),
        seed=POLICY.bootstrap_seed,
        iterations=POLICY.bootstrap_iterations,
        confidence_level_ppm=POLICY.confidence_level_ppm,
    )
    insufficient = paired_bootstrap_interval(
        (source,),
        seed=POLICY.bootstrap_seed,
        iterations=POLICY.bootstrap_iterations,
        confidence_level_ppm=POLICY.confidence_level_ppm,
    )

    assert first == second_run
    assert first.digest() == second_run.digest()
    assert first.lower_bound is not None
    assert first.upper_bound is not None
    assert insufficient.status is ConfidenceIntervalStatus.INSUFFICIENT_CASE_POPULATION
    assert insufficient.lower_bound is None
    assert StatisticsPolicyV1.from_dict(POLICY.to_dict()) == POLICY
    assert StatisticsPolicyV1.from_json(POLICY.to_json()) == POLICY
    assert RunStatisticsV1.from_json(ratio_statistics.to_json()) == ratio_statistics


def test_statistics_run_bootstrap_budget_fails_before_rng_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered_rng = False

    class ForbiddenRandom:
        def __init__(self, _seed: int) -> None:
            nonlocal entered_rng
            entered_rng = True
            raise AssertionError("RNG loop must not start after total-budget rejection")

    monkeypatch.setattr(statistics_module.random, "Random", ForbiddenRandom)
    with pytest.raises(StatisticsError, match="total bootstrap.*budget"):
        statistics_module._validate_run_bootstrap_budget(
            case_count=500,
            iterations=100_000,
            metric_count=len(CoreMetric),
        )
    assert entered_rng is False


def test_statistics_compute_checks_total_budget_before_any_metric_bootstrap(
    ratio_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, bundle = ratio_source
    entered_bootstrap = False

    def reject_budget(**_kwargs: Any) -> int:
        raise StatisticsError("total bootstrap work exceeds the Run resource budget")

    def forbidden_bootstrap(*_args: Any, **_kwargs: Any):
        nonlocal entered_bootstrap
        entered_bootstrap = True
        raise AssertionError("per-metric bootstrap must not start")

    monkeypatch.setattr(
        statistics_module,
        "_validate_run_bootstrap_budget",
        reject_budget,
    )
    monkeypatch.setattr(
        statistics_module,
        "paired_bootstrap_interval",
        forbidden_bootstrap,
    )
    with pytest.raises(StatisticsError, match="total bootstrap.*budget"):
        compute_run_statistics(
            bundle,
            run_config=run.config,
            case_snapshot=run.snapshot,
            policy=POLICY,
        )
    assert entered_bootstrap is False


def _reseal_stable_id(payload: dict[str, Any], field: str, namespace: str) -> None:
    identity = dict(payload)
    identity.pop(field)
    payload[field] = stable_id(namespace, identity)


@pytest.mark.parametrize(
    ("metric_name", "replacement"),
    (
        (
            CoreMetric.JUDGE_FAILURE_RATE.value,
            {
                "status": "available",
                "numerator": 0,
                "denominator": 102,
                "value": 0,
            },
        ),
        (
            CoreMetric.ISSUE_F1.value,
            {
                "status": "available",
                "numerator": 0,
                "denominator": 1,
                "value": 0,
            },
        ),
        (
            CoreMetric.CRITICAL_HIGH_MISS_COUNT.value,
            {
                "status": "available",
                "numerator": 1,
                "denominator": 2,
                "value": 1,
            },
        ),
        (
            CoreMetric.FABRICATED_FINDINGS_PER_PR.value,
            {
                "status": "available",
                "numerator": 1,
                "denominator": 2,
                "value": 500_000,
            },
        ),
        (
            CoreMetric.LINE_RECALL.value,
            {
                "status": "missing",
                "numerator": None,
                "denominator": None,
                "value": None,
            },
        ),
    ),
)
def test_statistics_rejects_resealed_metric_derived_field_tamper(
    ratio_statistics: Any,
    metric_name: str,
    replacement: dict[str, Any],
) -> None:
    payload = deepcopy(ratio_statistics.to_dict())
    metric = next(item for item in payload["metrics"] if item["metric"] == metric_name)
    metric.update(replacement)
    _reseal_stable_id(metric, "metric_id", "statistics-metric-v1")
    _reseal_stable_id(payload, "statistics_id", "run-statistics-v1")

    with pytest.raises(StatisticsError, match="Case contributions|reaggregated"):
        RunStatisticsV1.from_dict(payload)


def test_statistics_rejects_resealed_trial_projection_derived_field_tamper(
    ratio_statistics: Any,
) -> None:
    payload = deepcopy(
        ratio_statistics.trial_metric(1, CoreMetric.JUDGE_FAILURE_RATE).to_dict()
    )
    payload.update({"numerator": 0, "denominator": 2, "value": 0})
    _reseal_stable_id(payload, "projection_id", "trial-metric-projection-v1")

    with pytest.raises(StatisticsError, match="Case contributions|reaggregated"):
        TrialMetricProjectionV1.from_dict(payload)


def test_statistics_rejects_resealed_metric_coverage_tamper(
    ratio_statistics: Any,
) -> None:
    payload = deepcopy(ratio_statistics.to_dict())
    metric = next(
        item
        for item in payload["metrics"]
        if item["metric"] == CoreMetric.JUDGE_FAILURE_RATE.value
    )
    metric["coverage"]["completed_trial_count"] = 1
    metric["coverage"]["agent_failure_count"] = 1
    _reseal_stable_id(metric, "metric_id", "statistics-metric-v1")
    _reseal_stable_id(payload, "statistics_id", "run-statistics-v1")

    with pytest.raises(StatisticsError, match="Case contributions|reaggregated"):
        RunStatisticsV1.from_dict(payload)


def _tamper_reseal_confidence_interval(
    payload: dict[str, Any],
    metric_name: str,
) -> None:
    metric = next(
        item for item in payload["metrics"] if item["metric"] == metric_name
    )
    interval = metric["confidence_interval"]
    coverage = interval["coverage"]
    coverage["available_case_count"] = 0
    coverage["missing_case_count"] = coverage["total_case_count"]
    interval.update(
        status=ConfidenceIntervalStatus.AVAILABLE.value,
        lower_bound=0,
        upper_bound=0,
    )
    _reseal_stable_id(
        interval,
        "interval_id",
        "confidence-interval-v1",
    )
    _reseal_stable_id(metric, "metric_id", "statistics-metric-v1")
    _reseal_stable_id(payload, "statistics_id", "run-statistics-v1")


def test_statistics_rejects_resealed_confidence_interval_tamper(
    ratio_statistics: Any,
) -> None:
    payload = deepcopy(ratio_statistics.to_dict())
    _tamper_reseal_confidence_interval(
        payload,
        CoreMetric.JUDGE_FAILURE_RATE.value,
    )

    with pytest.raises(
        StatisticsError,
        match="confidence interval|bootstrap|recomputed",
    ):
        RunStatisticsV1.from_dict(payload)


def test_statistics_hydration_checks_total_budget_before_interval_replay(
    ratio_statistics: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered_interval_replay = False

    def reject_budget(**_kwargs: Any) -> int:
        raise StatisticsError(
            "total bootstrap work exceeds the Run resource budget"
        )

    def forbidden_interval(*_args: Any, **_kwargs: Any):
        nonlocal entered_interval_replay
        entered_interval_replay = True
        raise AssertionError("interval replay must not start after budget rejection")

    monkeypatch.setattr(
        statistics_module,
        "_validate_run_bootstrap_budget",
        reject_budget,
    )
    monkeypatch.setattr(
        statistics_module,
        "paired_bootstrap_interval",
        forbidden_interval,
    )

    with pytest.raises(StatisticsError, match="total bootstrap.*budget"):
        RunStatisticsV1.from_dict(ratio_statistics.to_dict())
    assert entered_interval_replay is False


def test_statistics_hydration_preflights_every_metric_before_bootstrap(
    ratio_statistics: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_count = 19
    bootstrap_iterations = 100_000
    assert len(ratio_statistics.metrics) == 26
    assert (
        case_count * len(ratio_statistics.metrics) * bootstrap_iterations
        <= statistics_module.MAX_RUN_BOOTSTRAP_DRAWS
    )
    assert (
        (case_count * 25 + 500) * bootstrap_iterations
        > statistics_module.MAX_RUN_BOOTSTRAP_DRAWS
    )

    def expand_cases(source: Any, count: int) -> tuple[Any, ...]:
        return tuple(
            replace(
                source,
                task_id=f"preflight-case-{index:04d}",
                canonical_case_digest=canonical_sha256(
                    {"preflight-case": index}
                ),
            )
            for index in range(count)
        )

    expanded_projections = []
    for projection in ratio_statistics.trial_metrics:
        contributions = expand_cases(
            projection.case_contributions[0],
            case_count,
        )
        aggregate = statistics_module._reaggregate_case_contributions(
            contributions
        )
        expanded_projections.append(
            replace(
                projection,
                status=aggregate.status,
                numerator=aggregate.numerator,
                denominator=aggregate.denominator,
                value=aggregate.value,
                coverage=aggregate.coverage,
                case_contributions=contributions,
            )
        )

    expanded_metrics = []
    for index, metric in enumerate(ratio_statistics.metrics):
        metric_case_count = (
            500
            if index == len(ratio_statistics.metrics) - 1
            else case_count
        )
        contributions = expand_cases(
            metric.case_contributions[0],
            metric_case_count,
        )
        aggregate = statistics_module._reaggregate_case_contributions(
            contributions
        )
        metric_projections = tuple(
            projection
            for projection in expanded_projections
            if projection.metric is metric.metric
        )
        interval = ConfidenceIntervalV1(
            metric=metric.metric,
            kind=metric.kind,
            unit=metric.unit,
            status=ConfidenceIntervalStatus.NOT_SCORABLE,
            lower_bound=None,
            upper_bound=None,
            seed=ratio_statistics.bootstrap_policy.bootstrap_seed,
            iterations=bootstrap_iterations,
            confidence_level_ppm=(
                ratio_statistics.bootstrap_policy.confidence_level_ppm
            ),
            coverage=statistics_module._bootstrap_coverage(contributions),
        )
        expanded_metrics.append(
            replace(
                metric,
                status=aggregate.status,
                numerator=aggregate.numerator,
                denominator=aggregate.denominator,
                value=aggregate.value,
                coverage=aggregate.coverage,
                dispersion=statistics_module._dispersion(
                    metric.unit,
                    metric_projections,
                ),
                confidence_interval=interval,
                case_contributions=contributions,
            )
        )

    payload = deepcopy(ratio_statistics.to_dict())
    payload["source_binding"]["trial_score_digests"] = sorted(
        canonical_sha256({"preflight-score": index})
        for index in range(case_count * ratio_statistics.trial_count)
    )
    payload["bootstrap_policy"]["bootstrap_iterations"] = bootstrap_iterations
    payload["metrics"] = [metric.to_dict() for metric in expanded_metrics]
    payload["trial_metrics"] = [
        projection.to_dict() for projection in expanded_projections
    ]
    _reseal_stable_id(payload, "statistics_id", "run-statistics-v1")

    bootstrap_calls = 0

    def forbidden_bootstrap(*_args: Any, **_kwargs: Any):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        raise AssertionError("bootstrap replay must not start before preflight")

    monkeypatch.setattr(
        statistics_module,
        "paired_bootstrap_interval",
        forbidden_bootstrap,
    )

    with pytest.raises(StatisticsError):
        RunStatisticsV1.from_dict(payload)
    assert bootstrap_calls == 0


def test_statistics_rejects_post_binding_trial_score_digest_change(
    ratio_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, bundle = ratio_source
    original_bind = statistics_module.bind_analysis_source

    def bind_then_change_digest(*args: Any, **kwargs: Any):
        binding = original_bind(*args, **kwargs)
        monkeypatch.setattr(TrialScore, "digest", lambda _self: "0" * 64)
        return binding

    monkeypatch.setattr(statistics_module, "bind_analysis_source", bind_then_change_digest)
    with pytest.raises(StatisticsError, match="TrialScore digests"):
        compute_run_statistics(
            bundle,
            run_config=run.config,
            case_snapshot=run.snapshot,
            policy=POLICY,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("bootstrap_seed", -1),
        ("bootstrap_iterations", 0),
        ("confidence_level_ppm", 0),
        ("confidence_level_ppm", 1_000_000),
    ),
)
def test_statistics_policy_rejects_invalid_bounds(field: str, value: int) -> None:
    payload = POLICY.to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        StatisticsPolicyV1.from_dict(payload)

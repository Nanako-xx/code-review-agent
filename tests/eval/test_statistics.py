from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from review_agent_eval.analysis_artifacts import AnalysisSourceBinding
from review_agent_eval.config import EvalRunConfig
from review_agent_eval.intent_evaluator import IntentEvaluator, IntentJudgeRelation
from review_agent_eval.metrics import (
    CoreMetric,
    MetricsAggregator,
    TrialScorer,
)
from review_agent_eval.models import (
    IntentTruth,
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
    ConfidenceIntervalStatus,
    MetricDirection,
    RunStatisticsV1,
    StatisticsMetricStatus,
    StatisticsPolicyV1,
    compute_run_statistics,
    paired_bootstrap_interval,
)

from .test_intent_evaluator import judge_decision, judge_failure
from .test_judge import _execution
from .test_metrics import (
    TARGET_MATERIALIZATION_ID,
    _completed_submission,
    _failed_submission,
    _score_completed,
    _score_sources,
)
from .test_orchestrator_target_replay_v2 import (
    _CountingJudge,
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


def _repeated_bundle(
    root: Path,
    *,
    adapter: Any,
    trial_count: int,
    instance: str,
    intent_truth: IntentTruth,
    review_truth: ReviewTruth,
    judge: Any,
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
    evaluated = orchestrator.evaluate_run(
        config.run_id,
        evaluator_execution=_execution(),
        evaluation_revision="statistics-fixture-v1",
    )
    hydrated = orchestrator.load_run_evaluation(
        config.run_id,
        evaluated.evaluation_id,
    )
    return run, hydrated


@pytest.fixture(scope="module")
def ratio_statistics(tmp_path_factory: pytest.TempPathFactory):
    truth = IntentTruth(
        scorable=False,
        authority=None,
        expected_claims=(),
        forbidden_claims=(),
        clarification_policy=None,
    )
    review_truth = ReviewTruth(
        completeness=TruthCompleteness.HUMAN_OBSERVED,
        novel_finding_policy=NovelFindingPolicy.VERIFY,
        expected_findings=(),
        known_invalid_findings=(),
    )
    run, bundle = _repeated_bundle(
        tmp_path_factory.mktemp("statistics-ratio"),
        adapter=_FrozenSuccessAdapter(),
        trial_count=2,
        instance="statistics-ratio",
        intent_truth=truth,
        review_truth=review_truth,
        judge=_CountingJudge(),
    )
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case, replay, execution, config = _score_sources(trial_count=2)
    _, _, _, first = _score_completed(
        case,
        replay,
        execution,
        config,
        1,
        finding_count=2,
    )
    _, _, _, original_second = _score_completed(
        case,
        replay,
        execution,
        config,
        2,
        finding_count=100,
    )
    changed_contributions = tuple(
        replace(item, numerator=99)
        if item.metric is CoreMetric.ISSUE_PRECISION
        else item
        for item in original_second.contributions
    )
    second = object.__new__(type(original_second))
    for name in original_second.__dataclass_fields__:
        object.__setattr__(
            second,
            name,
            changed_contributions
            if name == "contributions"
            else getattr(original_second, name),
        )
    binding = AnalysisSourceBinding(
        run_id=config.run_id,
        evaluation_id=first.compatibility.evaluation_id,
        summary_id=stable_id("statistics-test-summary", {"ratio": True}),
        summary_digest=canonical_sha256({"summary": "ratio"}),
        run_config_digest=config.digest(),
        case_snapshot_digest=canonical_sha256({"snapshot": "ratio"}),
        trial_score_digests=tuple(sorted((first.digest(), second.digest()))),
    )
    bind_calls = []

    def bound(*args: Any, **kwargs: Any) -> AnalysisSourceBinding:
        bind_calls.append((args, kwargs))
        return binding

    monkeypatch.setattr(statistics_module, "bind_analysis_source", bound)
    result = compute_run_statistics(
        SimpleNamespace(
            trials=(
                SimpleNamespace(trial_score=first),
                SimpleNamespace(trial_score=second),
            )
        ),
        run_config=config,
        case_snapshot=SimpleNamespace(),
        policy=POLICY,
    )
    metric = result.metric(CoreMetric.ISSUE_PRECISION)

    assert len(bind_calls) == 1
    assert metric.numerator == 100
    assert metric.denominator == 102
    assert metric.value == 980_392
    assert metric.value != (500_000 + 990_000) // 2
    assert tuple(
        result.trial_metric(index, CoreMetric.ISSUE_PRECISION).value
        for index in (1, 2)
    ) == (500_000, 990_000)
    assert metric.direction is MetricDirection.HIGHER_IS_BETTER


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
        if item.metric is CoreMetric.AGENT_FAILURE_RATE
    )

    assert tuple(item.trial_index for item in projections) == (1, 2)
    assert tuple(item.value for item in projections) == (0, 0)
    assert ratio_statistics.trial_count == 2
    assert not hasattr(ratio_statistics, "best_trial_index")


def test_statistics_marks_missing_authority_not_scorable_not_zero(
    ratio_statistics: Any,
) -> None:
    metric = ratio_statistics.metric(CoreMetric.LINE_RECALL)

    assert metric.status is StatisticsMetricStatus.NOT_SCORABLE
    assert metric.numerator is None
    assert metric.denominator is None
    assert metric.value is None
    assert metric.coverage.metric_sources[0].not_scorable_count == 2


def test_statistics_bootstrap_is_deterministic_for_fixed_policy(
    ratio_statistics: Any,
) -> None:
    source = ratio_statistics.metric(
        CoreMetric.AGENT_FAILURE_RATE
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

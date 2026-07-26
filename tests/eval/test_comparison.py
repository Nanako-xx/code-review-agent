from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import review_agent_eval.comparison as comparison_module
from review_agent_eval.analysis_artifacts import AnalysisArtifactStore
from review_agent_eval.artifacts import ArtifactIntegrityError
from review_agent_eval.comparison import (
    COMPARISON_ALGORITHM_VERSION,
    REQUIRED_CASE_FIELDS,
    REQUIRED_EVALUATOR_FIELDS,
    CaseDeltaV1,
    ComparisonCompatibilityV1,
    ComparisonError,
    ComparisonPolicyV1,
    ComparisonStatus,
    DeltaClassification,
    DeltaIntervalStatus,
    RunComparisonV1,
    VerifiedRunEvaluation,
    compare_runs,
)
from review_agent_eval.config import EvalRunConfig
from review_agent_eval.metrics import CoreMetric
from review_agent_eval.models import (
    FindingSeverity,
    NovelFindingPolicy,
    ReviewTruth,
    SubmissionFinding,
    SubmissionReview,
    SubmissionStatus,
    TruthCompleteness,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)
from review_agent_eval.statistics import (
    STATISTICS_ALGORITHM_VERSION,
    ConfidenceIntervalStatus,
    ConfidenceIntervalV1,
    RunStatisticsV1,
    StatisticsPolicyV1,
)

from .test_orchestrator_target_replay_v2 import (
    _CountingJudge,
    _FrozenRun,
    _RecordingJudge,
    _case_bank,
    _execution,
    _frozen_orchestrator,
    _frozen_snapshot_and_case,
)
from .test_frozen_context import _prepared_bundle
from .test_target_runner import (
    _FrozenSuccessAdapter,
    _run_config,
    _runner as _frozen_runner,
)


STATISTICS_POLICY = StatisticsPolicyV1(
    algorithm_version=STATISTICS_ALGORITHM_VERSION,
    bootstrap_seed=20260726,
    bootstrap_iterations=128,
    confidence_level_ppm=950_000,
)
POLICY = ComparisonPolicyV1(
    schema_version="comparison_policy_v1",
    statistics_policy=STATISTICS_POLICY,
    required_case_fields=REQUIRED_CASE_FIELDS,
    required_evaluator_fields=REQUIRED_EVALUATOR_FIELDS,
)


class _MixedOutcomeAdapter(_FrozenSuccessAdapter):
    """Produce one failed, one Judge-ungraded, and one ordinary Trial."""

    def __init__(self) -> None:
        super().__init__()
        self.slot = 0

    def run(self, *args: Any, **kwargs: Any):
        self.slot += 1
        if self.slot == 1:
            raise TimeoutError("comparison fixture timeout")
        submission = super().run(*args, **kwargs)
        if self.slot != 2:
            return submission
        finding = SubmissionFinding(
            finding_id="comparison-novel-finding",
            claim="The frozen context contains a semantic comparison defect.",
            severity=FindingSeverity.HIGH,
            path=None,
            side=None,
            from_line=None,
            to_line=None,
            evidence_refs=(),
            suggested_action="Correct the behavior described by the context.",
        )
        return replace(
            submission,
            review=SubmissionReview(findings=(finding,), uncertainties=()),
        )


def _three_trial_config(
    snapshot: Any,
    *,
    instance: str,
    agent: Any | None = None,
    trial_count: int = 3,
) -> EvalRunConfig:
    base = _run_config(snapshot, current=False, instance=instance)
    return EvalRunConfig.create(
        run_instance_key=instance,
        agent=base.agent if agent is None else agent,
        clarification_matcher=base.clarification_matcher,
        evaluator=base.evaluator,
        suite=base.suite,
        adapter_capabilities=base.adapter_capabilities,
        trial_count=trial_count,
        resource_budgets=base.resource_budgets,
    )


def _evaluate_run(
    root: Path,
    *,
    prepared: Any,
    snapshot: Any,
    case: Any,
    config: EvalRunConfig,
    adapter: Any,
    execution: Any,
    judge: Any,
    revision: str = "comparison-fixture-v1",
) -> tuple[_FrozenRun, Any, Any]:
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
        evaluator_execution=execution,
        evaluation_revision=revision,
    )
    hydrated = orchestrator.load_run_evaluation(
        config.run_id,
        evaluated.evaluation_id,
    )
    return run, hydrated, orchestrator


@pytest.fixture(scope="module")
def paired_sources(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("strict-comparison")
    prepared_parent = root / "prepared"
    prepared_parent.mkdir()
    prepared = _prepared_bundle(prepared_parent)
    snapshot, case = _frozen_snapshot_and_case(prepared)
    execution = _execution()

    baseline_config = _three_trial_config(
        snapshot,
        instance="strict-comparison-baseline",
    )
    candidate_agent = replace(
        baseline_config.agent,
        agent_id="strict-comparison-candidate-agent",
        agent_name="Strict comparison candidate",
        agent_version="3",
        model="candidate-model",
        provider="candidate-provider",
        prompt_config_digest=canonical_sha256(
            {"prompt": "strict-comparison-candidate"}
        ),
    )
    candidate_config = _three_trial_config(
        snapshot,
        instance="strict-comparison-candidate",
        agent=candidate_agent,
    )

    baseline_run, baseline_bundle, _baseline_orchestrator = _evaluate_run(
        root / "baseline",
        prepared=prepared,
        snapshot=snapshot,
        case=case,
        config=baseline_config,
        adapter=_MixedOutcomeAdapter(),
        execution=execution,
        judge=_RecordingJudge(execution),
    )
    candidate_run, candidate_bundle, candidate_orchestrator = _evaluate_run(
        root / "candidate",
        prepared=prepared,
        snapshot=snapshot,
        case=case,
        config=candidate_config,
        adapter=_FrozenSuccessAdapter(),
        execution=execution,
        judge=_CountingJudge(),
    )
    baseline = VerifiedRunEvaluation.create(
        baseline_bundle,
        run_config=baseline_run.config,
        case_snapshot=baseline_run.snapshot,
    )
    candidate = VerifiedRunEvaluation.create(
        candidate_bundle,
        run_config=candidate_run.config,
        case_snapshot=candidate_run.snapshot,
    )
    two_trial_config = _three_trial_config(
        snapshot,
        instance="strict-comparison-trial-mismatch",
        trial_count=2,
    )
    two_run, two_bundle, _ = _evaluate_run(
        root / "trial-mismatch",
        prepared=prepared,
        snapshot=snapshot,
        case=case,
        config=two_trial_config,
        adapter=_FrozenSuccessAdapter(),
        execution=execution,
        judge=_CountingJudge(),
    )
    two_trial = VerifiedRunEvaluation.create(
        two_bundle,
        run_config=two_run.config,
        case_snapshot=two_run.snapshot,
    )
    return {
        "root": root,
        "prepared": prepared,
        "snapshot": snapshot,
        "case": case,
        "execution": execution,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_run": candidate_run,
        "candidate_orchestrator": candidate_orchestrator,
        "two_trial": two_trial,
    }


def _comparison(sources: dict[str, Any]) -> RunComparisonV1:
    return compare_runs(sources["baseline"], sources["candidate"], POLICY)


def test_compare_pairs_by_case_digest_and_trial_index(
    paired_sources: dict[str, Any],
) -> None:
    candidate = paired_sources["candidate"]
    reordered = VerifiedRunEvaluation(
        bundle=replace(
            candidate.bundle,
            trials=tuple(reversed(candidate.bundle.trials)),
        ),
        run_config=candidate.run_config,
        case_snapshot=candidate.case_snapshot,
        source_binding=candidate.source_binding,
    )

    result = compare_runs(paired_sources["baseline"], reordered, POLICY)

    assert result.status is ComparisonStatus.COMPARABLE
    assert len(result.case_deltas) == 1
    paired = result.case_deltas[0].paired_trials
    assert tuple(item.trial_index for item in paired) == (1, 2, 3)
    assert all(
        item.task_id == paired_sources["case"].task_id
        and item.case_version == paired_sources["case"].case_version
        and item.canonical_case_digest == paired_sources["case"].digest()
        for item in paired
    )
    assert tuple(item.baseline.trial_id for item in paired) == tuple(
        paired_sources["baseline"].run_config.trial_id(
            paired_sources["case"].task_id,
            index,
        )
        for index in (1, 2, 3)
    )


def test_compare_does_not_drop_failed_or_ungraded_trial_pairs(
    paired_sources: dict[str, Any],
) -> None:
    result = _comparison(paired_sources)
    paired = result.case_deltas[0].paired_trials

    assert len(paired) == 3
    assert paired[0].baseline.submission_status is SubmissionStatus.FAILED
    assert paired[0].metric_deltas
    assert any(
        item.baseline.judge_ungraded_count > 0
        for item in paired
    )
    assert result.judge_coverage_delta.baseline_ungraded_count > 0
    assert result.judge_coverage_delta.candidate_ungraded_count == 0


def test_compare_reports_metric_and_case_deltas_without_overall_score(
    paired_sources: dict[str, Any],
) -> None:
    result = _comparison(paired_sources)

    assert tuple(item.metric for item in result.metric_deltas) == tuple(
        sorted(CoreMetric, key=lambda item: item.value)
    )
    assert all(type(item) is CaseDeltaV1 for item in result.case_deltas)
    failure_delta = result.metric_delta(CoreMetric.AGENT_FAILURE_RATE)
    assert failure_delta.absolute_delta is not None
    assert failure_delta.classification is DeltaClassification.IMPROVED
    assert result.case_deltas[0].metric_delta(
        CoreMetric.AGENT_FAILURE_RATE
    ).classification is DeltaClassification.IMPROVED
    assert not hasattr(result, "overall_score")
    assert not hasattr(result, "case_pass")
    assert not hasattr(failure_delta, "judge_coverage_delta")
    assert result.judge_coverage_delta != result.case_deltas[0].metric_delta(
        CoreMetric.JUDGE_UNGRADED_RATE
    )


def test_compare_rejects_case_digest_trial_count_and_evaluator_mismatch(
    paired_sources: dict[str, Any],
) -> None:
    root = paired_sources["root"]
    prepared = paired_sources["prepared"]
    execution = paired_sources["execution"]

    changed_truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(),
        known_invalid_findings=(),
    )
    changed_snapshot, changed_case = _frozen_snapshot_and_case(
        prepared,
        review_truth=changed_truth,
    )
    changed_config = _three_trial_config(
        changed_snapshot,
        instance="strict-comparison-case-mismatch",
    )
    changed_run, changed_bundle, _ = _evaluate_run(
        root / "case-mismatch",
        prepared=prepared,
        snapshot=changed_snapshot,
        case=changed_case,
        config=changed_config,
        adapter=_FrozenSuccessAdapter(),
        execution=execution,
        judge=_CountingJudge(),
    )
    changed_verified = VerifiedRunEvaluation.create(
        changed_bundle,
        run_config=changed_run.config,
        case_snapshot=changed_run.snapshot,
    )

    changed_execution = replace(
        execution,
        cache_policy_version="semantic-judge-cache-comparison-mismatch-v1",
    )
    evaluated = paired_sources["candidate_orchestrator"].evaluate_run(
        paired_sources["candidate_run"].config.run_id,
        evaluator_execution=changed_execution,
        evaluation_revision="comparison-evaluator-mismatch-v1",
    )
    changed_eval_bundle = paired_sources[
        "candidate_orchestrator"
    ].load_run_evaluation(
        paired_sources["candidate_run"].config.run_id,
        evaluated.evaluation_id,
    )
    evaluator_verified = VerifiedRunEvaluation.create(
        changed_eval_bundle,
        run_config=paired_sources["candidate_run"].config,
        case_snapshot=paired_sources["candidate_run"].snapshot,
    )

    cases = (
        (changed_verified, "cases.canonical_case_digests"),
        (paired_sources["two_trial"], "trial.count"),
        (evaluator_verified, "evaluator.execution.digest"),
    )
    for candidate, expected_field in cases:
        result = compare_runs(paired_sources["baseline"], candidate, POLICY)
        assert result.status is ComparisonStatus.NOT_COMPARABLE
        assert expected_field in result.incompatibilities
        assert result.metric_deltas == ()
        assert result.case_deltas == ()


def test_compare_allows_agent_identity_change_and_records_agent_delta(
    paired_sources: dict[str, Any],
) -> None:
    result = _comparison(paired_sources)
    fields = tuple(item.field for item in result.compatibility.agent_delta.changes)

    assert result.status is ComparisonStatus.COMPARABLE
    assert "run_id" in fields
    assert "evaluation_id" in fields
    assert "agent_config_digest" in fields
    assert "agent.provider" in fields
    assert "agent.model" in fields
    assert "agent.prompt_config_digest" in fields
    assert result.compatibility.shared_projection is not None


def test_compare_reuses_fixed_bootstrap_policy(
    paired_sources: dict[str, Any],
) -> None:
    result = _comparison(paired_sources)

    for delta in result.metric_deltas:
        interval = delta.confidence_interval
        assert interval.seed == STATISTICS_POLICY.bootstrap_seed
        assert interval.iterations == STATISTICS_POLICY.bootstrap_iterations
        assert (
            interval.confidence_level_ppm
            == STATISTICS_POLICY.confidence_level_ppm
        )
        assert interval.status is DeltaIntervalStatus.INSUFFICIENT_CASE_POPULATION
    assert result.baseline_statistics.bootstrap_policy == STATISTICS_POLICY
    assert result.candidate_statistics.bootstrap_policy == STATISTICS_POLICY
    assert result.algorithm_digest == POLICY.algorithm_digest
    assert POLICY.algorithm_version == COMPARISON_ALGORITHM_VERSION
    assert ComparisonPolicyV1.from_json(POLICY.to_json()) == POLICY


def test_comparison_policy_paths_are_a_closed_non_deletable_set() -> None:
    missing = REQUIRED_CASE_FIELDS[:-1]
    with pytest.raises(ComparisonError, match="required_case_fields"):
        replace(POLICY, required_case_fields=missing)
    with pytest.raises(ComparisonError, match="required_evaluator_fields"):
        replace(
            POLICY,
            required_evaluator_fields=(
                *REQUIRED_EVALUATOR_FIELDS,
                "__class__.__mro__",
            ),
        )


def test_verified_evaluation_rebinds_supplied_binding(
    paired_sources: dict[str, Any],
) -> None:
    candidate = paired_sources["candidate"]
    forged = replace(
        candidate.source_binding,
        summary_digest="0" * 64,
    )
    with pytest.raises(ArtifactIntegrityError, match="binding|source"):
        VerifiedRunEvaluation(
            bundle=candidate.bundle,
            run_config=candidate.run_config,
            case_snapshot=candidate.case_snapshot,
            source_binding=forged,
        )


def test_comparison_artifact_publish_load_and_source_replay(
    paired_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    result = _comparison(paired_sources)
    store = AnalysisArtifactStore(tmp_path / ".eval-analyses")

    receipt = store.publish_comparison(result, policy=POLICY)
    loaded = store.load_comparison(receipt.artifact_id)
    replayed = store.load_verified_comparison(
        receipt.artifact_id,
        baseline=paired_sources["baseline"],
        candidate=paired_sources["candidate"],
        policy=POLICY,
    )

    assert receipt.kind == "comparison"
    assert tuple(item.to_dict() for item in receipt.source_bindings) == tuple(
        sorted(
            (
                result.baseline_binding.to_dict(),
                result.candidate_binding.to_dict(),
            ),
            key=canonical_json_bytes,
        )
    )
    assert loaded == result
    assert replayed == result
    assert store.load_json_bundle("comparison", receipt.artifact_id) == {
        "comparison_result.json": result.to_dict()
    }


def test_comparison_artifact_rejects_tamper_and_resealed_nested_contradiction(
    paired_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    result = _comparison(paired_sources)
    payload = deepcopy(result.to_dict())
    metric = next(
        item
        for item in payload["metric_deltas"]
        if item["metric"] == CoreMetric.AGENT_FAILURE_RATE.value
    )
    metric["absolute_delta"] = 0
    identity = dict(metric)
    identity.pop("delta_id")
    metric["delta_id"] = stable_id("metric-delta-v1", identity)
    comparison_identity = dict(payload)
    comparison_identity.pop("comparison_id")
    payload["comparison_id"] = stable_id(
        "run-comparison-v1",
        comparison_identity,
    )
    with pytest.raises(ComparisonError, match="recomputed|delta"):
        RunComparisonV1.from_dict(payload)

    store = AnalysisArtifactStore(tmp_path / ".eval-analyses")
    receipt = store.publish_comparison(result, policy=POLICY)
    path = (
        store.root
        / "comparison"
        / receipt.artifact_id
        / "comparison_result.json"
    )
    data = path.read_bytes()
    path.write_bytes(data[:-1] + (b" " if data[-1:] != b" " else b"\n"))
    with pytest.raises(ArtifactIntegrityError, match="digest|hash|size|canonical"):
        store.load_comparison(receipt.artifact_id)


def _tampered_projection_payload(
    result: RunComparisonV1,
    projection_name: str,
    field_path: str,
    field_value: Any,
) -> dict[str, Any]:
    payload = deepcopy(result.to_dict())
    compatibility = payload["compatibility"]
    projection = compatibility[projection_name]
    field = next(
        item
        for item in projection["fields"]
        if item["path"] == field_path
    )
    field["value"] = field_value
    projection["projection_digest"] = canonical_sha256(
        {"fields": projection["fields"]}
    )
    compatibility["shared_projection"] = None
    compatibility_identity = dict(compatibility)
    compatibility_identity.pop("compatibility_id")
    compatibility["compatibility_id"] = stable_id(
        "comparison-compatibility-v1",
        compatibility_identity,
    )
    payload["status"] = ComparisonStatus.NOT_COMPARABLE.value
    payload["metric_deltas"] = []
    payload["case_deltas"] = []
    payload["incompatibilities"] = [field_path]
    comparison_identity = dict(payload)
    comparison_identity.pop("comparison_id")
    payload["comparison_id"] = stable_id(
        "run-comparison-v1",
        comparison_identity,
    )
    return payload


def _forge_comparison_from_tampered_payload(
    source: RunComparisonV1,
    payload: dict[str, Any],
) -> RunComparisonV1:
    forged = object.__new__(RunComparisonV1)
    values = {
        name: getattr(source, name)
        for name in source.__dataclass_fields__
    }
    values.update(
        comparison_id=payload["comparison_id"],
        status=ComparisonStatus.NOT_COMPARABLE,
        compatibility=ComparisonCompatibilityV1.from_dict(
            payload["compatibility"]
        ),
        metric_deltas=(),
        case_deltas=(),
        incompatibilities=tuple(payload["incompatibilities"]),
    )
    for name, value in values.items():
        object.__setattr__(forged, name, value)
    return forged


@pytest.mark.parametrize(
    "projection_name",
    ("baseline_projection", "candidate_projection"),
)
def test_comparison_hydration_rejects_resealed_case_snapshot_projection_tamper(
    paired_sources: dict[str, Any],
    projection_name: str,
) -> None:
    payload = _tampered_projection_payload(
        _comparison(paired_sources),
        projection_name,
        "case_snapshot.digest",
        "0" * 64,
    )

    with pytest.raises(
        ComparisonError,
        match="case snapshot|case_snapshot|source binding",
    ):
        RunComparisonV1.from_dict(payload)


@pytest.mark.parametrize(
    "projection_name",
    ("baseline_projection", "candidate_projection"),
)
def test_comparison_publish_rejects_resealed_case_snapshot_projection_tamper(
    paired_sources: dict[str, Any],
    tmp_path: Path,
    projection_name: str,
) -> None:
    result = _comparison(paired_sources)
    payload = _tampered_projection_payload(
        result,
        projection_name,
        "case_snapshot.digest",
        "0" * 64,
    )
    forged = _forge_comparison_from_tampered_payload(result, payload)
    store = AnalysisArtifactStore(tmp_path / ".eval-analyses")

    with pytest.raises(
        ArtifactIntegrityError,
        match="canonical|hydration|binding|projection",
    ):
        store.publish_comparison(forged, policy=POLICY)


@pytest.mark.parametrize(
    "projection_name",
    ("baseline_projection", "candidate_projection"),
)
def test_comparison_hydration_rejects_resealed_evaluator_projection_tamper(
    paired_sources: dict[str, Any],
    projection_name: str,
) -> None:
    payload = _tampered_projection_payload(
        _comparison(paired_sources),
        projection_name,
        "evaluator.execution.digest",
        "0" * 64,
    )

    with pytest.raises(
        ComparisonError,
        match="evaluation|evaluator|execution|source binding",
    ):
        RunComparisonV1.from_dict(payload)


@pytest.mark.parametrize(
    "projection_name",
    ("baseline_projection", "candidate_projection"),
)
def test_comparison_publish_rejects_resealed_evaluator_projection_tamper(
    paired_sources: dict[str, Any],
    tmp_path: Path,
    projection_name: str,
) -> None:
    result = _comparison(paired_sources)
    payload = _tampered_projection_payload(
        result,
        projection_name,
        "evaluator.execution.digest",
        "0" * 64,
    )
    forged = _forge_comparison_from_tampered_payload(result, payload)
    store = AnalysisArtifactStore(tmp_path / ".eval-analyses")

    with pytest.raises(
        ArtifactIntegrityError,
        match="canonical|hydration|binding|evaluation|evaluator",
    ):
        store.publish_comparison(forged, policy=POLICY)


def _tampered_nested_statistics_ci_payload(
    result: RunComparisonV1,
) -> dict[str, Any]:
    payload = deepcopy(result.to_dict())
    statistics = payload["baseline_statistics"]
    metric = next(
        item
        for item in statistics["metrics"]
        if item["metric"] == CoreMetric.AGENT_FAILURE_RATE.value
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
    for value, field_name, namespace in (
        (interval, "interval_id", "confidence-interval-v1"),
        (metric, "metric_id", "statistics-metric-v1"),
        (statistics, "statistics_id", "run-statistics-v1"),
    ):
        identity = dict(value)
        identity.pop(field_name)
        value[field_name] = stable_id(namespace, identity)
    comparison_identity = dict(payload)
    comparison_identity.pop("comparison_id")
    payload["comparison_id"] = stable_id(
        "run-comparison-v1",
        comparison_identity,
    )
    return payload


def _forge_comparison_with_tampered_statistics(
    source: RunComparisonV1,
    payload: dict[str, Any],
) -> RunComparisonV1:
    source_statistics = source.baseline_statistics
    metric_payload = next(
        item
        for item in payload["baseline_statistics"]["metrics"]
        if item["metric"] == CoreMetric.AGENT_FAILURE_RATE.value
    )
    tampered_interval = ConfidenceIntervalV1.from_dict(
        metric_payload["confidence_interval"]
    )
    source_metric = source_statistics.metric(CoreMetric.AGENT_FAILURE_RATE)
    tampered_metric = replace(
        source_metric,
        confidence_interval=tampered_interval,
    )
    forged_statistics = object.__new__(RunStatisticsV1)
    statistics_values = {
        name: getattr(source_statistics, name)
        for name in source_statistics.__dataclass_fields__
    }
    statistics_values["metrics"] = tuple(
        tampered_metric if item.metric is CoreMetric.AGENT_FAILURE_RATE else item
        for item in source_statistics.metrics
    )
    for name, value in statistics_values.items():
        object.__setattr__(forged_statistics, name, value)

    forged = object.__new__(RunComparisonV1)
    comparison_values = {
        name: getattr(source, name)
        for name in source.__dataclass_fields__
    }
    comparison_values.update(
        comparison_id=payload["comparison_id"],
        baseline_statistics=forged_statistics,
    )
    for name, value in comparison_values.items():
        object.__setattr__(forged, name, value)
    return forged


def test_comparison_rejects_resealed_nested_statistics_ci_tamper(
    paired_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    result = _comparison(paired_sources)
    payload = _tampered_nested_statistics_ci_payload(result)

    with pytest.raises(
        ValueError,
        match="confidence interval|bootstrap|recomputed",
    ):
        RunComparisonV1.from_dict(payload)

    forged = _forge_comparison_with_tampered_statistics(result, payload)
    store = AnalysisArtifactStore(tmp_path / ".eval-analyses")
    with pytest.raises(
        ArtifactIntegrityError,
        match="canonical|hydration|confidence|bootstrap",
    ):
        store.publish_comparison(forged, policy=POLICY)


def test_compare_revalidates_real_sources_after_statistics(
    paired_sources: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = paired_sources["baseline"]
    attacked_bundle = replace(source.bundle)
    attacked = VerifiedRunEvaluation.create(
        attacked_bundle,
        run_config=source.run_config,
        case_snapshot=source.case_snapshot,
    )
    original_report = attacked_bundle.report
    original_compute = comparison_module.compute_run_statistics
    mutated = False

    def mutate_after_statistics(bundle: Any, **kwargs: Any):
        nonlocal mutated
        statistics = original_compute(bundle, **kwargs)
        if bundle is attacked_bundle:
            object.__setattr__(
                attacked_bundle,
                "report",
                original_report + "\nsource changed after statistics",
            )
            mutated = True
        return statistics

    monkeypatch.setattr(
        comparison_module,
        "compute_run_statistics",
        mutate_after_statistics,
    )
    try:
        with pytest.raises(
            ArtifactIntegrityError,
            match="report|source|changed|binding",
        ):
            compare_runs(attacked, paired_sources["candidate"], POLICY)
    finally:
        object.__setattr__(attacked_bundle, "report", original_report)
    assert mutated is True


def test_not_comparable_artifact_round_trip_has_no_partial_delta(
    paired_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    result = compare_runs(
        paired_sources["baseline"],
        paired_sources["two_trial"],
        POLICY,
    )

    assert result.status is ComparisonStatus.NOT_COMPARABLE
    assert result.incompatibilities == ("trial.count",)
    assert result.metric_deltas == ()
    assert result.case_deltas == ()
    assert RunComparisonV1.from_json(result.to_json()) == result
    store = AnalysisArtifactStore(tmp_path / ".eval-analyses")
    receipt = store.publish_comparison(result, policy=POLICY)
    assert store.load_comparison(receipt.artifact_id) == result

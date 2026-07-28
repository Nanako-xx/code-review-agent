from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from review_agent_eval.analysis_artifacts import AnalysisArtifactStore, AnalysisReceipt
from review_agent_eval.artifacts import ArtifactConflictError, ArtifactIntegrityError
from review_agent_eval.calibration import (
    CalibrationStatus,
    HumanLabelSetV1,
    ReviewerProvenanceKind,
    score_calibration,
)
from review_agent_eval.cases import CaseSplit, RunCaseSnapshot, SuiteManifest
from review_agent_eval.comparison import (
    ComparisonStatus,
    VerifiedRunEvaluation,
    compare_runs,
)
from review_agent_eval.gates import (
    GATE_POLICY_SCHEMA_VERSION,
    GateCheckReason,
    GateCheckStatus,
    GateCheckV1,
    GateConstraintScope,
    GateDecision,
    GateEligibility,
    GateError,
    GateFailureRefV1,
    GateOperator,
    FrozenGatePolicy,
    MetricConstraintV1,
    GatePolicyV1,
    GateReferenceKind,
    GateResultV1,
    _coverage_disposition,
    _ineligible_check,
    evaluate_gate,
    prepare_gate_policy,
)
from review_agent_eval.intent_evaluator import IntentTruth
from review_agent_eval.judge import JudgeTask
from review_agent_eval.metrics import CoreMetric
from review_agent_eval.models import (
    CaseOrigin,
    IntentAuthority,
    NovelFindingPolicy,
    ReviewTruth,
    TruthCompleteness,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)
from review_agent_eval.statistics import MetricUnit

from .test_calibration import (
    _export,
    _human_label,
    _matching_labels,
    calibration_source,
)
from .test_comparison import (
    POLICY as COMPARISON_POLICY,
    _MixedOutcomeAdapter,
    _comparison,
    _evaluate_run,
    _three_trial_config,
    paired_sources,
)
from .test_judge import _execution
from .test_orchestrator_target_replay_v2 import (
    _RecordingJudge,
    _expected,
    _frozen_snapshot_and_case,
    _prepared_bundle,
)
from .test_target_runner import _FrozenSuccessAdapter

def _constraint(
    metric: CoreMetric,
    scope: GateConstraintScope,
    operator: GateOperator,
    threshold: int,
    *,
    required: bool = True,
    min_coverage_ppm: int | None = None,
) -> MetricConstraintV1:
    return MetricConstraintV1(
        metric=metric,
        scope=scope,
        operator=operator,
        threshold=threshold,
        unit=MetricUnit.PPM,
        required=required,
        min_coverage_ppm=min_coverage_ppm,
    )


def _policy(
    source: VerifiedRunEvaluation,
    candidate_config: Any,
    *,
    constraints: tuple[MetricConstraintV1, ...],
    eligibility: GateEligibility = GateEligibility.DIAGNOSTIC_ONLY,
    calibration_result_digests: tuple[str, ...] = (),
) -> GatePolicyV1:
    return GatePolicyV1.create(
        baseline_binding=source.source_binding,
        candidate_run_id=candidate_config.run_id,
        candidate_run_config_digest=candidate_config.digest(),
        case_snapshot_digest=source.case_snapshot.digest(),
        trial_count=candidate_config.trial_count,
        comparison_policy_digest=COMPARISON_POLICY.policy_digest,
        calibration_result_digests=calibration_result_digests,
        eligibility=eligibility,
        constraints=constraints,
    )


@pytest.mark.parametrize(
    ("coverage_ppm", "minimum_ppm", "expected"),
    (
        (333_333, None, GateCheckStatus.INSUFFICIENT_COVERAGE),
        (666_666, None, GateCheckStatus.INSUFFICIENT_COVERAGE),
        (333_333, 300_000, None),
        (333_333, 400_000, GateCheckStatus.INSUFFICIENT_COVERAGE),
    ),
)
def test_gate_coverage_disposition_requires_full_coverage_by_default(
    coverage_ppm: int,
    minimum_ppm: int | None,
    expected: GateCheckStatus | None,
) -> None:
    assert _coverage_disposition(coverage_ppm, minimum_ppm) is expected


def test_gate_check_status_reason_pairs_and_legacy_hydration_are_closed() -> None:
    constraint = _constraint(
        CoreMetric.AGENT_FAILURE_RATE,
        GateConstraintScope.CANDIDATE_ABSOLUTE,
        GateOperator.AT_MOST,
        1_000_000,
    )
    pending = GateCheckV1(
        constraint.constraint_id,
        constraint.metric,
        constraint.scope,
        constraint.operator,
        constraint.required,
        GateCheckStatus.PENDING,
        None,
        constraint.threshold,
        constraint.unit,
        None,
        constraint.min_coverage_ppm,
        None,
        (),
        (),
        (GateCheckReason.CALIBRATION_PENDING_HUMAN_LABELS,),
    )
    partial_ref = GateFailureRefV1(
        GateReferenceKind.TRIAL,
        "gate-trial-ref-v1-" + "3" * 64,
        None,
        constraint.threshold,
        constraint.unit,
        GateCheckReason.NOT_SCORABLE,
    )
    partial_constraint = _constraint(
        CoreMetric.AGENT_FAILURE_RATE,
        GateConstraintScope.CANDIDATE_ABSOLUTE,
        GateOperator.AT_MOST,
        1_000_000,
        min_coverage_ppm=300_000,
    )
    partial = GateCheckV1(
        partial_constraint.constraint_id,
        partial_constraint.metric,
        partial_constraint.scope,
        partial_constraint.operator,
        partial_constraint.required,
        GateCheckStatus.PASS,
        0,
        partial_constraint.threshold,
        partial_constraint.unit,
        333_333,
        300_000,
        None,
        (partial_ref,),
        (),
        (),
    )
    assert partial.failure_refs == (partial_ref,)
    forged_ref = partial.to_dict()
    forged_ref["failure_refs"][0]["reason"] = GateCheckReason.THRESHOLD_FAILED.value
    forged_ref["check_id"] = stable_id("gate-check-v1", {
        key: value for key, value in forged_ref.items() if key != "check_id"
    })
    with pytest.raises(GateError, match="threshold_failed.*satisfies"):
        GateCheckV1.from_dict(forged_ref)
    invalid = pending.to_dict()
    invalid["reasons"] = [GateCheckReason.THRESHOLD_FAILED.value]
    invalid["check_id"] = stable_id("gate-check-v1", {
        key: value for key, value in invalid.items() if key != "check_id"
    })
    with pytest.raises(GateError, match="status|reason"):
        GateCheckV1.from_dict(invalid)

    legacy = pending.to_dict()
    legacy["status"] = GateCheckStatus.INELIGIBLE.value
    legacy["reasons"] = [GateCheckReason.NOT_SCORABLE.value]
    legacy["check_id"] = stable_id("gate-check-v1", {
        key: value for key, value in legacy.items() if key != "check_id"
    })
    hydrated_legacy = GateCheckV1.from_dict(legacy)
    assert hydrated_legacy.status is GateCheckStatus.INELIGIBLE
    hydrated_result = GateResultV1.create(
        policy_digest="0" * 64,
        policy_artifact_id="analysis-artifact-v1-" + "1" * 64,
        policy_receipt_digest="2" * 64,
        comparison_id="comparison-v1",
        decision=GateDecision.INELIGIBLE,
        checks=(pending,),
    )
    legacy_result = hydrated_result.to_dict()
    legacy_result["checks"] = [legacy]
    legacy_result["gate_result_id"] = stable_id("gate-result-v1", {
        key: value for key, value in legacy_result.items() if key != "gate_result_id"
    })
    assert GateResultV1.from_dict(legacy_result).checks[0] == hydrated_legacy
    with pytest.raises(GateError, match="legacy|ineligible"):
        GateCheckV1(
            constraint.constraint_id,
            constraint.metric,
            constraint.scope,
            constraint.operator,
            constraint.required,
            GateCheckStatus.INELIGIBLE,
            None,
            constraint.threshold,
            constraint.unit,
            None,
            constraint.min_coverage_ppm,
            None,
            (),
            (),
            (GateCheckReason.NOT_SCORABLE,),
        )
    with pytest.raises(GateError, match="legacy|ineligible"):
        GateResultV1.create(
            policy_digest="0" * 64,
            policy_artifact_id="analysis-artifact-v1-" + "1" * 64,
            policy_receipt_digest="2" * 64,
            comparison_id="comparison-v1",
            decision=GateDecision.INELIGIBLE,
            checks=(hydrated_legacy,),
        )


@pytest.mark.parametrize(
    ("reasons", "expected_status"),
    (
        (
            (
                GateCheckReason.CALIBRATION_PENDING_HUMAN_LABELS,
                GateCheckReason.CALIBRATION_FAILED_THRESHOLDS,
            ),
            GateCheckStatus.PENDING,
        ),
        (
            (
                GateCheckReason.FAILED_COVERAGE,
                GateCheckReason.ZERO_DENOMINATOR,
            ),
            GateCheckStatus.INSUFFICIENT_COVERAGE,
        ),
    ),
)
def test_gate_unavailable_reason_resolver_preserves_mixed_reasons(
    reasons: tuple[GateCheckReason, ...],
    expected_status: GateCheckStatus,
) -> None:
    constraint = _constraint(
        CoreMetric.AGENT_FAILURE_RATE,
        GateConstraintScope.CANDIDATE_ABSOLUTE,
        GateOperator.AT_MOST,
        1_000_000,
    )
    check = _ineligible_check(constraint, reasons)

    assert check.status is expected_status
    assert check.reasons == tuple(sorted(reasons, key=lambda item: item.value))
    assert check.reason is next(
        item for item in check.reasons if _ineligible_check(constraint, (item,)).status is expected_status
    )
    assert GateCheckV1.from_dict(check.to_dict()) == check
    replayed = GateResultV1.create(
        policy_digest="0" * 64,
        policy_artifact_id="analysis-artifact-v1-" + "1" * 64,
        policy_receipt_digest="2" * 64,
        comparison_id="comparison-v1",
        decision=GateDecision.INELIGIBLE,
        checks=(check,),
    )
    assert GateResultV1.from_dict(replayed.to_dict()) == replayed

    invalid = check.to_dict()
    invalid["status"] = (
        GateCheckStatus.NOT_SCORABLE.value
        if expected_status is not GateCheckStatus.NOT_SCORABLE
        else GateCheckStatus.PENDING.value
    )
    invalid["check_id"] = stable_id("gate-check-v1", {
        key: value for key, value in invalid.items() if key != "check_id"
    })
    with pytest.raises(GateError, match="status.*reasons"):
        GateCheckV1.from_dict(invalid)


@pytest.fixture(scope="module")
def gate_policy_store(
    tmp_path_factory: pytest.TempPathFactory,
) -> AnalysisArtifactStore:
    return AnalysisArtifactStore(
        tmp_path_factory.mktemp("regression-gate-policy-store") / "analysis"
    )


def _core_snapshot(
    prepared: Any,
    *,
    intent_truth: IntentTruth,
    review_truth: ReviewTruth,
) -> tuple[RunCaseSnapshot, Any]:
    public_snapshot, public_case = _frozen_snapshot_and_case(
        prepared,
        intent_truth=intent_truth,
        review_truth=review_truth,
    )
    case_payload = public_case.to_dict()
    case_payload["source"] = {
        "suite": "core-regression",
        "origin": CaseOrigin.HAND_AUTHORED.value,
        "source_id": "core-regression-fixture-case",
        "source_version": "core-fixture-v1",
        "source_uri": None,
        "license": None,
        "content_hash": canonical_sha256({"core": "fixture-case"}),
    }
    case = type(public_case).from_dict(case_payload)
    case_bytes = case.to_json().encode("utf-8")
    manifest_payload = public_snapshot.manifest.to_dict()
    manifest_payload["suite_id"] = "core-regression"
    manifest_payload["suite_version"] = "core-fixture-v1"
    manifest_payload["source"] = {
        "kind": "core",
        "source_id": "core-regression-fixture",
        "source_version": "core-fixture-v1",
        "source_uri": None,
        "license": None,
        "content_hash": canonical_sha256({"core": "fixture-suite"}),
        "preparation_binding": None,
    }
    manifest_payload["cases"] = [
        {
            **manifest_payload["cases"][0],
            "split": CaseSplit.REGRESSION.value,
            "raw_file_size_bytes": len(case_bytes),
            "raw_file_sha256": hashlib.sha256(case_bytes).hexdigest(),
            "canonical_case_digest": case.digest(),
            "eval_input_digest": case.eval_input().digest(),
            "truth_completeness": case.review_truth.completeness.value,
        }
    ]
    manifest = SuiteManifest.from_dict(manifest_payload)
    return RunCaseSnapshot.build(manifest, ((manifest.cases[0], case),)), case


@pytest.fixture(scope="module")
def core_gate_sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("regression-gate-core")
    prepared_root = root / "prepared"
    prepared_root.mkdir()
    prepared = _prepared_bundle(prepared_root)
    intent_truth = IntentTruth.from_dict(
        {
            "scorable": True,
            "authority": IntentAuthority.EXPLICIT_AUTHOR_METADATA.value,
            "expected_claims": [
                {
                    "truth_id": "expected-goal",
                    "dimension": "goal",
                    "text": "Preserve access control",
                    "required": True,
                }
            ],
            "forbidden_claims": [],
            "clarification_policy": "not_required",
        }
    )
    review_truth = ReviewTruth(
        completeness=TruthCompleteness.CLOSED_WORLD,
        novel_finding_policy=NovelFindingPolicy.FORBID,
        expected_findings=(
            _expected(
                "expected-gate-finding",
                "The frozen context demonstrates the expected gate defect.",
            ),
        ),
        known_invalid_findings=(),
    )
    snapshot, case = _core_snapshot(
        prepared,
        intent_truth=intent_truth,
        review_truth=review_truth,
    )
    execution = _execution()

    def make_source(instance: str, adapter: Any, *, candidate: bool = False):
        config = _three_trial_config(snapshot, instance=instance)
        run, bundle, _ = _evaluate_run(
            root / instance,
            prepared=prepared,
            snapshot=snapshot,
            case=case,
            config=config,
            adapter=adapter,
            execution=execution,
            judge=_RecordingJudge(execution),
        )
        return {
            "run": run,
            "config": config,
            "verified": VerifiedRunEvaluation.create(
                bundle,
                run_config=config,
                case_snapshot=snapshot,
            ),
            "candidate": candidate,
        }

    baseline = make_source("core-gate-baseline", _FrozenSuccessAdapter())
    promote = make_source(
        "core-gate-promote-candidate",
        _FrozenSuccessAdapter(),
        candidate=True,
    )
    block = make_source(
        "core-gate-block-candidate",
        _MixedOutcomeAdapter(),
        candidate=True,
    )
    return {
        "baseline": baseline["verified"],
        "promote": promote["verified"],
        "block": block["verified"],
        "promote_config": promote["config"],
        "block_config": block["config"],
    }


def _calibration_result(
    source: dict[str, Any],
    tmp_path: Path,
    profile: JudgeTask,
    *,
    labels: Any | None = None,
    reviewer_kind: Any | None = None,
):
    package = _export(source, tmp_path / profile.value, profile)
    if labels is None:
        labels = _matching_labels(package, reviewer_kind=reviewer_kind) if reviewer_kind else _matching_labels(package)
    from review_agent_eval.calibration import score_calibration

    return score_calibration(
        source["verified_by_profile"][profile],
        package=package,
        labels=labels,
    )


def _failing_constraint(
    metric: CoreMetric,
    scope: GateConstraintScope,
    actual: int,
) -> MetricConstraintV1:
    if actual < 1_000_000:
        return _constraint(metric, scope, GateOperator.AT_LEAST, actual + 1)
    return _constraint(metric, scope, GateOperator.AT_MOST, actual - 1)


def test_gate_policy_binds_candidate_run_plan_before_results_exist(
    paired_sources: dict[str, Any],
) -> None:
    baseline = paired_sources["baseline"]
    candidate_config = paired_sources["candidate_run"].config
    policy = _policy(
        baseline,
        candidate_config,
        constraints=(_constraint(CoreMetric.AGENT_FAILURE_RATE, GateConstraintScope.CANDIDATE_ABSOLUTE, GateOperator.AT_MOST, 0),),
    )

    prepared = prepare_gate_policy(
        baseline,
        candidate_config,
        policy=policy,
    )

    assert prepared.schema_version == GATE_POLICY_SCHEMA_VERSION
    assert prepared.baseline_binding == baseline.source_binding
    assert prepared.candidate_run_id == candidate_config.run_id
    assert prepared.candidate_run_config_digest == candidate_config.digest()
    assert prepared.case_snapshot_digest == baseline.case_snapshot.digest()
    assert prepared.trial_count == candidate_config.trial_count
    assert prepared.policy_id == policy.policy_id


def test_gate_policy_cannot_be_overwritten_or_rebound(
    paired_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    baseline = paired_sources["baseline"]
    candidate_config = paired_sources["candidate_run"].config
    policy = _policy(
        baseline,
        candidate_config,
        constraints=(_constraint(CoreMetric.AGENT_FAILURE_RATE, GateConstraintScope.CANDIDATE_ABSOLUTE, GateOperator.AT_MOST, 0),),
    )
    prepared = prepare_gate_policy(baseline, candidate_config, policy=policy)
    store = AnalysisArtifactStore(tmp_path / "analysis")
    frozen = store.publish_gate_policy(
        prepared,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    assert store.publish_gate_policy(
        prepared,
        baseline=baseline,
        candidate_run_config=candidate_config,
    ) == frozen
    loaded = store.load_verified_gate_policy(
        frozen.artifact_id,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    assert type(loaded) is FrozenGatePolicy
    assert loaded == frozen
    assert frozen.policy == prepared

    rebound = _policy(
        baseline,
        paired_sources["two_trial"].run_config,
        constraints=prepared.constraints,
    )
    with pytest.raises(GateError, match="trial_count|compatible"):
        prepare_gate_policy(
            baseline,
            paired_sources["two_trial"].run_config,
            policy=rebound,
        )

    policy_path = store.root / "gate-policy" / frozen.artifact_id / "gate_policy.json"
    policy_path.write_bytes(canonical_json_bytes({"tampered": True}))
    with pytest.raises((ArtifactIntegrityError, ArtifactConflictError)):
        store.publish_gate_policy(
            prepared,
            baseline=baseline,
            candidate_run_config=candidate_config,
        )


def test_gate_checks_absolute_and_baseline_delta_constraints(
    core_gate_sources: dict[str, Any],
    gate_policy_store: AnalysisArtifactStore,
) -> None:
    baseline = core_gate_sources["baseline"]
    candidate = core_gate_sources["block"]
    comparison = compare_runs(
        baseline,
        candidate,
        COMPARISON_POLICY,
    )
    metric_delta = comparison.metric_delta(CoreMetric.AGENT_FAILURE_RATE)
    assert metric_delta.candidate.value is not None
    assert metric_delta.absolute_delta is not None
    policy = _policy(
        baseline,
        core_gate_sources["block_config"],
        eligibility=GateEligibility.RELEASE_BLOCKING,
        constraints=(
            _failing_constraint(CoreMetric.AGENT_FAILURE_RATE, GateConstraintScope.CANDIDATE_ABSOLUTE, metric_delta.candidate.value),
            _failing_constraint(CoreMetric.AGENT_FAILURE_RATE, GateConstraintScope.BASELINE_DELTA, metric_delta.absolute_delta),
        ),
    )
    policy = prepare_gate_policy(
        baseline,
        core_gate_sources["block_config"],
        policy=policy,
    )
    frozen = gate_policy_store.publish_gate_policy(
        policy,
        baseline=baseline,
        candidate_run_config=core_gate_sources["block_config"],
    )
    result = evaluate_gate(gate_policy_store, frozen, comparison, {})

    assert result.decision is GateDecision.BLOCK
    by_scope = {check.scope: check for check in result.checks if check.required}
    assert all(check.status is GateCheckStatus.FAIL for check in by_scope.values())
    assert by_scope[GateConstraintScope.CANDIDATE_ABSOLUTE].actual == metric_delta.candidate.value
    assert by_scope[GateConstraintScope.BASELINE_DELTA].actual == metric_delta.absolute_delta


def test_gate_reports_case_and_trial_refs_for_each_failure(
    core_gate_sources: dict[str, Any],
    gate_policy_store: AnalysisArtifactStore,
) -> None:
    baseline = core_gate_sources["baseline"]
    candidate = core_gate_sources["block"]
    comparison = compare_runs(baseline, candidate, COMPARISON_POLICY)
    actual = comparison.metric_delta(CoreMetric.AGENT_FAILURE_RATE).candidate.value
    assert actual is not None
    prepared = prepare_gate_policy(
        baseline,
        core_gate_sources["block_config"],
        policy=_policy(
            baseline,
            core_gate_sources["block_config"],
            eligibility=GateEligibility.RELEASE_BLOCKING,
            constraints=(
                _failing_constraint(CoreMetric.AGENT_FAILURE_RATE, GateConstraintScope.CANDIDATE_ABSOLUTE, actual),
            ),
        ),
    )
    frozen = gate_policy_store.publish_gate_policy(
        prepared,
        baseline=baseline,
        candidate_run_config=core_gate_sources["block_config"],
    )
    result = evaluate_gate(gate_policy_store, frozen, comparison, {})
    check = result.checks[0]
    assert check.status is GateCheckStatus.FAIL
    assert check.case_refs
    assert check.trial_refs
    assert all("core-gate" not in ref for ref in (*check.case_refs, *check.trial_refs))
    assert all(ref.startswith("gate-") for ref in (*check.case_refs, *check.trial_refs))


def test_gate_requires_calibration_for_semantic_metrics(
    core_gate_sources: dict[str, Any],
    calibration_source: dict[str, Any],
    tmp_path: Path,
    gate_policy_store: AnalysisArtifactStore,
) -> None:
    baseline = core_gate_sources["baseline"]
    candidate_config = core_gate_sources["promote_config"]
    comparison = compare_runs(baseline, core_gate_sources["promote"], COMPARISON_POLICY)
    eligible = _calibration_result(
        calibration_source,
        tmp_path,
        JudgeTask.FINDING_EQUIVALENCE,
    )
    assert eligible.status.value == "gate_eligible"
    prepared = prepare_gate_policy(
        baseline,
        candidate_config,
        policy=_policy(
            baseline,
            candidate_config,
            eligibility=GateEligibility.RELEASE_BLOCKING,
            calibration_result_digests=(eligible.digest(),),
            constraints=(
                _constraint(CoreMetric.ISSUE_RECALL, GateConstraintScope.CANDIDATE_ABSOLUTE, GateOperator.AT_LEAST, 0),
            ),
        ),
    )
    frozen = gate_policy_store.publish_gate_policy(
        prepared,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    passed = evaluate_gate(
        gate_policy_store,
        frozen,
        comparison,
        {JudgeTask.FINDING_EQUIVALENCE.value: eligible},
    )
    assert passed.decision is GateDecision.PROMOTE

    missing = _policy(
        baseline,
        candidate_config,
        eligibility=GateEligibility.RELEASE_BLOCKING,
        constraints=(
            _constraint(CoreMetric.ISSUE_RECALL, GateConstraintScope.CANDIDATE_ABSOLUTE, GateOperator.AT_LEAST, 0),
        ),
    )
    missing = prepare_gate_policy(baseline, candidate_config, policy=missing)
    missing_frozen = gate_policy_store.publish_gate_policy(
        missing,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    missing_result = evaluate_gate(
        gate_policy_store,
        missing_frozen,
        comparison,
        {},
    )
    assert missing_result.decision is GateDecision.INELIGIBLE
    assert missing_result.checks[0].status is GateCheckStatus.NOT_SCORABLE

    profile = JudgeTask.FINDING_EQUIVALENCE
    pending_package = _export(
        calibration_source,
        tmp_path / "pending",
        profile,
    )
    pending = score_calibration(
        calibration_source["verified_by_profile"][profile],
        package=pending_package,
        labels=HumanLabelSetV1.create(package=pending_package, labels=()),
    )
    fixture_package = _export(
        calibration_source,
        tmp_path / "fixture",
        profile,
    )
    fixture = score_calibration(
        calibration_source["verified_by_profile"][profile],
        package=fixture_package,
        labels=_matching_labels(
            fixture_package,
            reviewer_kind=ReviewerProvenanceKind.FIXTURE,
        ),
    )
    failed_package = _export(
        calibration_source,
        tmp_path / "failed",
        profile,
    )
    matching = _matching_labels(failed_package)
    matching_by_item = {
        item.calibration_item_id: item.label for item in matching.labels
    }
    failed_labels = HumanLabelSetV1.create(
        package=failed_package,
        labels=tuple(
            _human_label(
                failed_package,
                item,
                (
                    "different"
                    if matching_by_item[item.calibration_item_id] == "equivalent"
                    else "equivalent"
                ),
            )
            for item in failed_package.items
        ),
    )
    failed = score_calibration(
        calibration_source["verified_by_profile"][profile],
        package=failed_package,
        labels=failed_labels,
    )
    assert pending.status is CalibrationStatus.PENDING_HUMAN_LABELS
    assert fixture.status is CalibrationStatus.PENDING_HUMAN_LABELS
    assert failed.status is CalibrationStatus.FAILED_THRESHOLDS
    for calibration, expected_status in (
        (pending, GateCheckStatus.PENDING),
        (fixture, GateCheckStatus.PENDING),
        (failed, GateCheckStatus.NOT_SCORABLE),
    ):
        untrusted_policy = prepare_gate_policy(
            baseline,
            candidate_config,
            policy=_policy(
                baseline,
                candidate_config,
                eligibility=GateEligibility.RELEASE_BLOCKING,
                calibration_result_digests=(calibration.digest(),),
                constraints=(
                    _constraint(
                        CoreMetric.ISSUE_RECALL,
                        GateConstraintScope.CANDIDATE_ABSOLUTE,
                        GateOperator.AT_LEAST,
                        0,
                    ),
                ),
            ),
        )
        untrusted_frozen = gate_policy_store.publish_gate_policy(
            untrusted_policy,
            baseline=baseline,
            candidate_run_config=candidate_config,
        )
        untrusted = evaluate_gate(
            gate_policy_store,
            untrusted_frozen,
            comparison,
            {profile.value: calibration},
        )
        assert untrusted.decision is GateDecision.INELIGIBLE
        assert untrusted.checks[0].status is expected_status


def test_gate_marks_public_or_unscorable_data_diagnostic_only(
    paired_sources: dict[str, Any],
    gate_policy_store: AnalysisArtifactStore,
) -> None:
    baseline = paired_sources["baseline"]
    candidate_config = paired_sources["candidate_run"].config
    requested = _policy(
        baseline,
        candidate_config,
        eligibility=GateEligibility.RELEASE_BLOCKING,
        constraints=(
            _constraint(CoreMetric.CLARIFICATION_ACCURACY, GateConstraintScope.CASE_ABSOLUTE, GateOperator.AT_LEAST, 0),
            _constraint(CoreMetric.CLARIFICATION_ACCURACY, GateConstraintScope.TRIAL_ABSOLUTE, GateOperator.AT_LEAST, 0),
        ),
    )
    prepared = prepare_gate_policy(baseline, candidate_config, policy=requested)
    assert prepared.eligibility is GateEligibility.DIAGNOSTIC_ONLY
    assert prepared.policy_id != requested.policy_id
    comparison = _comparison(paired_sources)
    frozen = gate_policy_store.publish_gate_policy(
        prepared,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    result = evaluate_gate(gate_policy_store, frozen, comparison, {})
    assert result.decision is GateDecision.INELIGIBLE
    scoped = {
        check.scope: check
        for check in result.checks
        if check.metric is CoreMetric.CLARIFICATION_ACCURACY
    }
    assert scoped[GateConstraintScope.CASE_ABSOLUTE].status is GateCheckStatus.NOT_SCORABLE
    assert scoped[GateConstraintScope.TRIAL_ABSOLUTE].status is GateCheckStatus.NOT_SCORABLE
    assert scoped[GateConstraintScope.CASE_ABSOLUTE].case_refs
    assert not scoped[GateConstraintScope.CASE_ABSOLUTE].trial_refs
    assert scoped[GateConstraintScope.TRIAL_ABSOLUTE].trial_refs
    assert not scoped[GateConstraintScope.TRIAL_ABSOLUTE].case_refs

    empty = _policy(
        baseline,
        candidate_config,
        constraints=(),
        eligibility=GateEligibility.DIAGNOSTIC_ONLY,
    )
    assert prepare_gate_policy(baseline, candidate_config, policy=empty).constraints == ()


def test_gate_returns_promote_block_and_ineligible_without_overall_score(
    core_gate_sources: dict[str, Any],
    paired_sources: dict[str, Any],
    gate_policy_store: AnalysisArtifactStore,
) -> None:
    baseline = core_gate_sources["baseline"]
    promote_config = core_gate_sources["promote_config"]
    promote_comparison = compare_runs(
        baseline,
        core_gate_sources["promote"],
        COMPARISON_POLICY,
    )
    promote_actual = promote_comparison.metric_delta(CoreMetric.AGENT_FAILURE_RATE).candidate.value
    assert promote_actual is not None
    promote_policy = prepare_gate_policy(
        baseline,
        promote_config,
        policy=_policy(
            baseline,
            promote_config,
            eligibility=GateEligibility.RELEASE_BLOCKING,
            constraints=(_constraint(CoreMetric.AGENT_FAILURE_RATE, GateConstraintScope.CANDIDATE_ABSOLUTE, GateOperator.AT_MOST, promote_actual),),
        ),
    )
    promote_frozen = gate_policy_store.publish_gate_policy(
        promote_policy,
        baseline=baseline,
        candidate_run_config=promote_config,
    )
    promoted = evaluate_gate(
        gate_policy_store,
        promote_frozen,
        promote_comparison,
        {},
    )
    assert promoted.decision is GateDecision.PROMOTE

    block_config = core_gate_sources["block_config"]
    block_comparison = compare_runs(
        baseline,
        core_gate_sources["block"],
        COMPARISON_POLICY,
    )
    block_actual = block_comparison.metric_delta(CoreMetric.AGENT_FAILURE_RATE).candidate.value
    assert block_actual is not None
    block_policy = prepare_gate_policy(
        baseline,
        block_config,
        policy=_policy(
            baseline,
            block_config,
            eligibility=GateEligibility.RELEASE_BLOCKING,
            constraints=(_failing_constraint(CoreMetric.AGENT_FAILURE_RATE, GateConstraintScope.CANDIDATE_ABSOLUTE, block_actual),),
        ),
    )
    block_frozen = gate_policy_store.publish_gate_policy(
        block_policy,
        baseline=baseline,
        candidate_run_config=block_config,
    )
    blocked = evaluate_gate(
        gate_policy_store,
        block_frozen,
        block_comparison,
        {},
    )
    assert blocked.decision is GateDecision.BLOCK

    public_baseline = paired_sources["baseline"]
    changed_execution = replace(
        paired_sources["execution"],
        cache_policy_version="semantic-judge-cache-gate-mismatch-v1",
    )
    evaluated = paired_sources["candidate_orchestrator"].evaluate_run(
        paired_sources["candidate_run"].config.run_id,
        evaluator_execution=changed_execution,
        evaluation_revision="gate-evaluator-mismatch-v1",
    )
    changed_bundle = paired_sources["candidate_orchestrator"].load_run_evaluation(
        paired_sources["candidate_run"].config.run_id,
        evaluated.evaluation_id,
    )
    public_candidate = VerifiedRunEvaluation.create(
        changed_bundle,
        run_config=paired_sources["candidate_run"].config,
        case_snapshot=paired_sources["candidate_run"].snapshot,
    )
    public_config = public_candidate.run_config
    incomparable = compare_runs(public_baseline, public_candidate, COMPARISON_POLICY)
    assert incomparable.status is ComparisonStatus.NOT_COMPARABLE
    ineligible_policy = prepare_gate_policy(
        public_baseline,
        public_config,
        policy=_policy(
            public_baseline,
            public_config,
            constraints=(_constraint(CoreMetric.AGENT_FAILURE_RATE, GateConstraintScope.CANDIDATE_ABSOLUTE, GateOperator.AT_MOST, 0),),
        ),
    )
    ineligible_frozen = gate_policy_store.publish_gate_policy(
        ineligible_policy,
        baseline=public_baseline,
        candidate_run_config=public_config,
    )
    ineligible = evaluate_gate(
        gate_policy_store,
        ineligible_frozen,
        incomparable,
        {},
    )
    assert ineligible.decision is GateDecision.INELIGIBLE
    assert set(promoted.to_dict()) == {
        "schema_version",
        "gate_result_id",
        "policy_digest",
        "policy_artifact_id",
        "policy_receipt_digest",
        "comparison_id",
        "decision",
        "checks",
    }
    assert all("overall" not in key.casefold() and "score" not in key.casefold() for key in promoted.to_dict())


@pytest.mark.parametrize(
    "kwargs",
    (
        {"unit": MetricUnit.COUNT},
        {"operator": "arbitrary"},
        {"threshold": True},
        {"threshold": -1},
        {"threshold": 1.5},
        {"threshold": float("nan")},
        {"threshold": float("inf")},
        {"min_coverage_ppm": -1},
    ),
)
def test_gate_constraint_rejects_noncanonical_threshold_metadata(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "metric": CoreMetric.AGENT_FAILURE_RATE,
        "scope": GateConstraintScope.CANDIDATE_ABSOLUTE,
        "operator": GateOperator.AT_MOST,
        "threshold": 0,
        "unit": MetricUnit.PPM,
        "required": True,
        "min_coverage_ppm": None,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        MetricConstraintV1(**values)


def test_gate_artifact_result_replay_rejects_resealed_result(
    core_gate_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    baseline = core_gate_sources["baseline"]
    candidate = core_gate_sources["promote"]
    candidate_config = core_gate_sources["promote_config"]
    comparison = compare_runs(baseline, candidate, COMPARISON_POLICY)
    actual = comparison.metric_delta(CoreMetric.AGENT_FAILURE_RATE).candidate.value
    assert actual is not None
    prepared = prepare_gate_policy(
        baseline,
        candidate_config,
        policy=_policy(
            baseline,
            candidate_config,
            eligibility=GateEligibility.RELEASE_BLOCKING,
            constraints=(_constraint(CoreMetric.AGENT_FAILURE_RATE, GateConstraintScope.CANDIDATE_ABSOLUTE, GateOperator.AT_MOST, actual),),
        ),
    )
    store = AnalysisArtifactStore(tmp_path / "analysis")
    policy = store.publish_gate_policy(
        prepared,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    result = evaluate_gate(store, policy, comparison, {})
    receipt = store.publish_gate_result(
        result,
        policy=policy,
        comparison=comparison,
        calibrations={},
    )
    assert store.load_verified_gate_result(
        receipt.artifact_id,
        policy=policy,
        comparison=comparison,
        calibrations={},
    ) == result
    forged_decision = (
        GateDecision.BLOCK
        if result.decision is not GateDecision.BLOCK
        else GateDecision.PROMOTE
    )
    forged_identity = result.to_dict()
    forged_identity.pop("gate_result_id")
    forged_identity["decision"] = forged_decision.value
    forged = object.__new__(type(result))
    object.__setattr__(forged, "schema_version", result.schema_version)
    object.__setattr__(
        forged,
        "gate_result_id",
        stable_id("gate-result-v1", forged_identity),
    )
    object.__setattr__(forged, "policy_digest", result.policy_digest)
    object.__setattr__(
        forged,
        "policy_artifact_id",
        result.policy_artifact_id,
    )
    object.__setattr__(
        forged,
        "policy_receipt_digest",
        result.policy_receipt_digest,
    )
    object.__setattr__(forged, "comparison_id", result.comparison_id)
    object.__setattr__(forged, "decision", forged_decision)
    object.__setattr__(forged, "checks", result.checks)
    with pytest.raises((ArtifactIntegrityError, ValueError)):
        store.publish_gate_result(
            forged,
            policy=policy,
            comparison=comparison,
            calibrations={},
        )


def test_gate_symbols_are_available_through_lazy_root_exports() -> None:
    import review_agent_eval

    assert review_agent_eval.GatePolicyV1 is GatePolicyV1
    assert review_agent_eval.FrozenGatePolicy is FrozenGatePolicy
    assert review_agent_eval.evaluate_gate is evaluate_gate


def test_gate_rejects_raw_policy_bypass_and_only_frozen_public_policy_is_ineligible(
    paired_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    baseline = paired_sources["baseline"]
    candidate = paired_sources["candidate"]
    candidate_config = paired_sources["candidate_run"].config
    comparison = compare_runs(baseline, candidate, COMPARISON_POLICY)
    raw_release = _policy(
        baseline,
        candidate_config,
        eligibility=GateEligibility.RELEASE_BLOCKING,
        constraints=(
            _constraint(
                CoreMetric.AGENT_FAILURE_RATE,
                GateConstraintScope.CANDIDATE_ABSOLUTE,
                GateOperator.AT_MOST,
                1_000_000,
            ),
        ),
    )

    store = AnalysisArtifactStore(tmp_path / "analysis")
    with pytest.raises(TypeError, match="FrozenGatePolicy|frozen"):
        evaluate_gate(store, raw_release, comparison, {})

    frozen = store.publish_gate_policy(
        raw_release,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    assert type(frozen) is FrozenGatePolicy
    assert frozen.policy.eligibility is GateEligibility.DIAGNOSTIC_ONLY
    result = evaluate_gate(store, frozen, comparison, {})
    assert result.decision is GateDecision.INELIGIBLE
    assert result.policy_artifact_id == frozen.artifact_id
    assert result.policy_receipt_digest == frozen.receipt_digest
    with pytest.raises(TypeError, match="store|AnalysisArtifactStore"):
        evaluate_gate(object(), frozen, comparison, {})
    with pytest.raises(TypeError, match="Store-issued|AnalysisArtifactStore"):
        FrozenGatePolicy()
    with pytest.raises((TypeError, ArtifactIntegrityError), match="FrozenGatePolicy|frozen"):
        store.publish_gate_result(
            result,
            policy=raw_release,
            comparison=comparison,
            calibrations={},
        )


def test_gate_live_store_check_rejects_copied_seal_forgery_without_policy_artifact(
    paired_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    baseline = paired_sources["baseline"]
    candidate = paired_sources["candidate"]
    candidate_config = paired_sources["candidate_run"].config
    comparison = compare_runs(baseline, candidate, COMPARISON_POLICY)
    raw_release = _policy(
        baseline,
        candidate_config,
        eligibility=GateEligibility.RELEASE_BLOCKING,
        constraints=(
            _constraint(
                CoreMetric.AGENT_FAILURE_RATE,
                GateConstraintScope.CANDIDATE_ABSOLUTE,
                GateOperator.AT_MOST,
                1_000_000,
            ),
        ),
    )
    store = AnalysisArtifactStore(tmp_path / "analysis")
    legitimate = store.publish_gate_policy(
        raw_release,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    assert legitimate.policy.eligibility is GateEligibility.DIAGNOSTIC_ONLY

    forged_receipt = AnalysisReceipt.create(
        kind="gate-policy",
        source_bindings=(raw_release.baseline_binding,),
        algorithm_digest=raw_release.algorithm_digest,
        files={"gate_policy.json": raw_release.to_dict()},
    )

    def copied_seal_forgery(receipt: AnalysisReceipt) -> FrozenGatePolicy:
        forged = object.__new__(FrozenGatePolicy)
        object.__setattr__(forged, "policy", raw_release)
        object.__setattr__(forged, "receipt", receipt)
        object.__setattr__(forged, "receipt_digest", receipt.digest())
        object.__setattr__(forged, "artifact_id", receipt.artifact_id)
        object.__setattr__(forged, "baseline_binding", raw_release.baseline_binding)
        object.__setattr__(
            forged,
            "candidate_run_config_digest",
            raw_release.candidate_run_config_digest,
        )
        object.__setattr__(
            forged,
            "_store_identity_digest",
            legitimate._store_identity_digest,
        )
        object.__setattr__(forged, "_seal", legitimate._seal)
        return forged

    absent = copied_seal_forgery(forged_receipt)
    assert absent.artifact_id != legitimate.artifact_id
    with pytest.raises(ArtifactIntegrityError):
        evaluate_gate(store, absent, comparison, {})

    aliases_other_policy = copied_seal_forgery(legitimate.receipt)
    assert aliases_other_policy.artifact_id == legitimate.artifact_id
    with pytest.raises(ArtifactIntegrityError):
        evaluate_gate(store, aliases_other_policy, comparison, {})

    policy_path = (
        store.root
        / "gate-policy"
        / legitimate.artifact_id
        / "gate_policy.json"
    )
    policy_path.write_bytes(canonical_json_bytes({"tampered": True}))
    with pytest.raises(ArtifactIntegrityError):
        evaluate_gate(store, legitimate, comparison, {})


def test_gate_rejects_frozen_policy_from_another_analysis_store_even_with_same_receipt_digest(
    paired_sources: dict[str, Any],
    tmp_path: Path,
) -> None:
    baseline = paired_sources["baseline"]
    candidate = paired_sources["candidate"]
    candidate_config = paired_sources["candidate_run"].config
    comparison = compare_runs(baseline, candidate, COMPARISON_POLICY)
    proposal = _policy(
        baseline,
        candidate_config,
        constraints=(
            _constraint(
                CoreMetric.AGENT_FAILURE_RATE,
                GateConstraintScope.CANDIDATE_ABSOLUTE,
                GateOperator.AT_MOST,
                1_000_000,
            ),
        ),
    )
    first_store = AnalysisArtifactStore(tmp_path / "analysis-a")
    second_store = AnalysisArtifactStore(tmp_path / "analysis-b")
    first = first_store.publish_gate_policy(
        proposal,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    second = second_store.publish_gate_policy(
        proposal,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    assert first.artifact_id == second.artifact_id
    assert first.receipt_digest == second.receipt_digest
    result = evaluate_gate(second_store, second, comparison, {})

    with pytest.raises(ArtifactIntegrityError, match="Store|store|receipt|frozen"):
        first_store.publish_gate_result(
            result,
            policy=second,
            comparison=comparison,
            calibrations={},
        )


def test_release_policy_rejects_all_optional_constraints(
    core_gate_sources: dict[str, Any],
    gate_policy_store: AnalysisArtifactStore,
) -> None:
    baseline = core_gate_sources["baseline"]
    candidate_config = core_gate_sources["promote_config"]
    optional = _constraint(
        CoreMetric.AGENT_FAILURE_RATE,
        GateConstraintScope.CANDIDATE_ABSOLUTE,
        GateOperator.AT_MOST,
        1_000_000,
        required=False,
    )

    with pytest.raises(GateError, match="required|release_blocking"):
        _policy(
            baseline,
            candidate_config,
            eligibility=GateEligibility.RELEASE_BLOCKING,
            constraints=(optional,),
        )

    diagnostic = _policy(
        baseline,
        candidate_config,
        eligibility=GateEligibility.DIAGNOSTIC_ONLY,
        constraints=(optional,),
    )
    assert diagnostic.constraints == (optional,)
    frozen = gate_policy_store.publish_gate_policy(
        diagnostic,
        baseline=baseline,
        candidate_run_config=candidate_config,
    )
    comparison = compare_runs(
        baseline,
        core_gate_sources["promote"],
        COMPARISON_POLICY,
    )
    result = evaluate_gate(gate_policy_store, frozen, comparison, {})
    assert result.decision is GateDecision.INELIGIBLE
    assert result.checks[0].status is GateCheckStatus.PASS
    unconfigured = tuple(
        check for check in result.checks if check.status is GateCheckStatus.NOT_CONFIGURED
    )
    assert {check.metric for check in unconfigured} == set(CoreMetric) - {
        CoreMetric.AGENT_FAILURE_RATE
    }
    assert all(not check.required for check in unconfigured)

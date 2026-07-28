from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from review_agent_eval.analysis_artifacts import AnalysisArtifactStore
from review_agent_eval.calibration import (
    CALIBRATION_ALGORITHM_VERSION,
    CALIBRATION_SELECTION_POLICY_SCHEMA_VERSION,
    CalibrationSelectionPolicyV1,
    HumanLabelSetV1,
    build_calibration_package,
    score_calibration,
)
from review_agent_eval.cli import EXIT_OK, _analysis_evaluation_loader, main
from review_agent_eval.comparison import (
    ComparisonPolicyV1,
    ComparisonStatus,
    REQUIRED_CASE_FIELDS,
    REQUIRED_EVALUATOR_FIELDS,
)
from review_agent_eval.gates import (
    GateDecision,
    GateEligibility,
    GateOperator,
    GatePolicyV1,
    GateConstraintScope,
    MetricConstraintV1,
)
from review_agent_eval.judge import JudgeTask
from review_agent_eval.metrics import CoreMetric
from review_agent_eval.models import canonical_json_bytes
from review_agent_eval.statistics import (
    STATISTICS_ALGORITHM_VERSION,
    MetricUnit,
    StatisticsPolicyV1,
)

from .test_cli import _root_arguments, _write_cli_suite


def _output(capsys: pytest.CaptureFixture[str]) -> dict:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def _write_json(path: Path, value: object) -> Path:
    path.write_bytes(canonical_json_bytes(value))
    return path


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = []
    for current, directories, names in os.walk(root):
        directories[:] = [name for name in directories if name != ".locks"]
        files.extend(Path(current) / name for name in names)
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _analysis_namespace(roots: dict[str, Path], analysis: Path) -> argparse.Namespace:
    return argparse.Namespace(
        suite_root=str(roots["suite"]),
        manifest="suite_manifest.json",
        expected_manifest_digest=None,
        runs_root=str(roots["runs"]),
        data_root=str(roots["data"]),
        workspace_root=str(roots["workspaces"]),
        analysis_root=str(analysis),
    )


def _prepare(
    roots: dict[str, Path],
    program: Path,
    *,
    instance: str,
    agent_id: str,
    mode: str,
    capsys: pytest.CaptureFixture[str],
) -> str:
    arguments = [
        "prepare",
        *_root_arguments(roots),
        "--agent-adapter",
        "subprocess",
        "--agent-id",
        agent_id,
        "--run-instance-key",
        instance,
        "--trial-count",
        "3",
        "--json",
    ]
    command = [
        str(Path(sys.executable).resolve()),
        str(program),
        mode,
        "{agent_id}",
        "{task_id}",
        "{trial_id}",
    ]
    if mode == "nonzero":
        command.append("scripted failure")
    for item in command:
        arguments.extend(("--agent-command", item))
    assert main(arguments) == EXIT_OK
    payload = _output(capsys)
    assert payload["trial_count"] == 3
    return payload["run_id"]


def _run_and_evaluate(
    roots: dict[str, Path],
    run_id: str,
    capsys: pytest.CaptureFixture[str],
) -> str:
    common = _root_arguments(roots)
    assert main(["run-agent", run_id, *common, "--json"]) == EXIT_OK
    run_payload = _output(capsys)
    assert len(run_payload["trials"]) == 3
    assert main(
        [
            "evaluate",
            run_id,
            *common,
            "--revision",
            "task15-e2e-v1",
            "--judge-provider",
            "fake",
            "--json",
        ]
    ) == EXIT_OK
    evaluated = _output(capsys)
    assert evaluated["trial_count"] == 3
    return evaluated["evaluation_id"]


def test_task15_scripted_analysis_lifecycle_is_source_bound_and_write_separated(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots, program = _write_cli_suite(tmp_path)
    analysis = tmp_path / ".eval-analyses"
    external = tmp_path / "external-calibration"
    common = [
        *_root_arguments(roots),
        "--analysis-root",
        str(analysis),
        "--json",
    ]

    statistics_policy = StatisticsPolicyV1(
        algorithm_version=STATISTICS_ALGORITHM_VERSION,
        bootstrap_seed=20260728,
        bootstrap_iterations=64,
        confidence_level_ppm=950_000,
    )
    comparison_policy = ComparisonPolicyV1(
        schema_version="comparison_policy_v1",
        statistics_policy=statistics_policy,
        required_case_fields=REQUIRED_CASE_FIELDS,
        required_evaluator_fields=REQUIRED_EVALUATOR_FIELDS,
    )
    comparison_path = _write_json(
        tmp_path / "comparison-policy.json",
        comparison_policy.to_dict(),
    )
    calibration_policy = CalibrationSelectionPolicyV1(
        schema_version=CALIBRATION_SELECTION_POLICY_SCHEMA_VERSION,
        algorithm_version=CALIBRATION_ALGORITHM_VERSION,
        selection_seed=20260728,
        max_items_per_profile=16,
        max_normal_items_per_stratum=8,
        minimum_human_labels=1,
        minimum_human_coverage_ppm=1_000_000,
        minimum_labels_per_class=0,
        minimum_exact_agreement_ppm=0,
        minimum_cohen_kappa_ppm=-1_000_000,
        minimum_auxiliary_human_coverage_ppm=1_000_000,
        minimum_auxiliary_labels_per_class=0,
        minimum_auxiliary_exact_agreement_ppm=0,
        minimum_auxiliary_cohen_kappa_ppm=None,
        trusted_reviewer_provenance_digests=(),
        trusted_adjudicator_provenance_digests=(),
    )
    calibration_path = _write_json(
        tmp_path / "calibration-policy.json",
        calibration_policy.to_dict(),
    )

    baseline_run = _prepare(
        roots,
        program,
        instance="task15-baseline",
        agent_id="task15-baseline-agent",
        mode="nonzero",
        capsys=capsys,
    )
    baseline_evaluation = _run_and_evaluate(roots, baseline_run, capsys)
    candidate_run = _prepare(
        roots,
        program,
        instance="task15-candidate",
        agent_id="task15-candidate-agent",
        mode="success",
        capsys=capsys,
    )

    namespace = _analysis_namespace(roots, analysis)
    with _analysis_evaluation_loader(namespace) as (load, run_store, _root):
        baseline = load(baseline_run, baseline_evaluation)
        candidate_config = run_store.load_run_config(candidate_run)
        assert len(baseline.trials) == 3
        package = build_calibration_package(
            baseline,
            profile=JudgeTask.INTENT_EQUIVALENCE,
            policy=calibration_policy,
        )
        labels = HumanLabelSetV1.create(package=package, labels=())
        expected_calibration = score_calibration(
            baseline,
            package=package,
            labels=labels,
        )
        gate_policy = GatePolicyV1.create(
            baseline_binding=baseline.source_binding,
            candidate_run_id=candidate_config.run_id,
            candidate_run_config_digest=candidate_config.digest(),
            case_snapshot_digest=baseline.case_snapshot.digest(),
            trial_count=candidate_config.trial_count,
            comparison_policy_digest=comparison_policy.policy_digest,
            calibration_result_digests=(expected_calibration.digest(),),
            eligibility=GateEligibility.RELEASE_BLOCKING,
            constraints=(
                MetricConstraintV1(
                    metric=CoreMetric.AGENT_FAILURE_RATE,
                    scope=GateConstraintScope.CANDIDATE_ABSOLUTE,
                    operator=GateOperator.AT_MOST,
                    threshold=0,
                    unit=MetricUnit.PPM,
                    required=True,
                    min_coverage_ppm=1_000_000,
                ),
            ),
        )
    gate_policy_path = _write_json(
        tmp_path / "gate-policy.json",
        gate_policy.to_dict(),
    )

    # The policy artifact is committed before the candidate has any attempt.
    assert main(
        [
            "gate",
            "prepare",
            *common,
            "--baseline-run-id",
            baseline_run,
            "--baseline-evaluation-id",
            baseline_evaluation,
            "--candidate-run-id",
            candidate_run,
            "--policy",
            str(gate_policy_path),
        ]
    ) == EXIT_OK
    gate_prepare = _output(capsys)
    gate_policy_artifact = gate_prepare["artifact_id"]
    policy_mtime = (
        analysis
        / "gate-policy"
        / gate_policy_artifact
        / "receipt.json"
    ).stat().st_mtime_ns

    candidate_evaluation = _run_and_evaluate(roots, candidate_run, capsys)
    runs_before_analysis = _tree_digest(roots["runs"])

    assert main(
        [
            "compare",
            *common,
            "--baseline-run-id",
            baseline_run,
            "--baseline-evaluation-id",
            baseline_evaluation,
            "--candidate-run-id",
            candidate_run,
            "--candidate-evaluation-id",
            candidate_evaluation,
            "--policy",
            str(comparison_path),
        ]
    ) == EXIT_OK
    compared = _output(capsys)
    assert compared["comparison_status"] == ComparisonStatus.COMPARABLE.value

    assert main(
        [
            "calibrate",
            "export",
            *common,
            "--run-id",
            baseline_run,
            "--evaluation-id",
            baseline_evaluation,
            "--profile",
            JudgeTask.INTENT_EQUIVALENCE.value,
            "--selection-policy",
            str(calibration_path),
            "--output-root",
            str(external),
        ]
    ) == EXIT_OK
    exported = _output(capsys)
    assert exported["calibration_status"] == "pending_human_labels"
    assert exported["selected_count"] == 0

    labels_path = _write_json(tmp_path / "labels.json", labels.to_dict())
    assert main(
        [
            "calibrate",
            "import-labels",
            *common,
            "--run-id",
            baseline_run,
            "--evaluation-id",
            baseline_evaluation,
            "--profile",
            JudgeTask.INTENT_EQUIVALENCE.value,
            "--selection-policy",
            str(calibration_path),
            "--package-id",
            exported["artifact_id"],
            "--labels",
            str(labels_path),
        ]
    ) == EXIT_OK
    imported = _output(capsys)
    assert imported["label_count"] == 0

    assert main(
        [
            "calibrate",
            "score",
            *common,
            "--run-id",
            baseline_run,
            "--evaluation-id",
            baseline_evaluation,
            "--profile",
            JudgeTask.INTENT_EQUIVALENCE.value,
            "--selection-policy",
            str(calibration_path),
            "--package-id",
            exported["artifact_id"],
            "--label-set-id",
            imported["artifact_id"],
        ]
    ) == EXIT_OK
    scored = _output(capsys)
    assert scored["calibration_status"] == "pending_human_labels"

    binding_path = _write_json(
        tmp_path / "calibration-binding.json",
        {
            "profile": JudgeTask.INTENT_EQUIVALENCE.value,
            "artifact_id": scored["artifact_id"],
            "run_id": baseline_run,
            "evaluation_id": baseline_evaluation,
            "selection_policy": str(calibration_path),
            "package_id": exported["artifact_id"],
            "label_set_id": imported["artifact_id"],
        },
    )
    assert main(
        [
            "gate",
            "evaluate",
            *common,
            "--baseline-run-id",
            baseline_run,
            "--baseline-evaluation-id",
            baseline_evaluation,
            "--candidate-run-id",
            candidate_run,
            "--candidate-evaluation-id",
            candidate_evaluation,
            "--comparison-id",
            compared["artifact_id"],
            "--comparison-policy",
            str(comparison_path),
            "--gate-policy-id",
            gate_policy_artifact,
            "--calibration-binding",
            str(binding_path),
        ]
    ) == EXIT_OK
    gated = _output(capsys)
    assert gated["decision"] == GateDecision.PROMOTE.value

    store = AnalysisArtifactStore(analysis, create_root=False)
    with _analysis_evaluation_loader(namespace) as (load, _run_store, _root):
        baseline = load(baseline_run, baseline_evaluation)
        candidate = load(candidate_run, candidate_evaluation)
        reloaded = store.load_verified_comparison(
            compared["artifact_id"],
            baseline=baseline,
            candidate=candidate,
            policy=comparison_policy,
        )
    assert reloaded.baseline_statistics.trial_count == 3
    assert reloaded.candidate_statistics.trial_count == 3
    assert len(
        [
            item
            for item in reloaded.baseline_statistics.trial_metrics
            if item.metric is CoreMetric.AGENT_FAILURE_RATE
        ]
    ) == 3
    assert (
        reloaded.baseline_statistics.metric(
            CoreMetric.AGENT_FAILURE_RATE
        ).coverage.agent_failure_count
        == 3
    )
    assert (
        reloaded.candidate_statistics.metric(
            CoreMetric.AGENT_FAILURE_RATE
        ).coverage.judge_ungraded_count
        >= 0
    )
    assert {
        (item.canonical_case_digest, item.trial_index)
        for case in reloaded.case_deltas
        for item in case.paired_trials
    } == {
        (baseline.trials[0].eval_case.digest(), 1),
        (baseline.trials[0].eval_case.digest(), 2),
        (baseline.trials[0].eval_case.digest(), 3),
    }
    comparison_bytes = (
        analysis
        / "comparison"
        / compared["artifact_id"]
        / "comparison_result.json"
    ).read_bytes()
    assert comparison_bytes == canonical_json_bytes(reloaded.to_dict())
    assert _tree_digest(roots["runs"]) == runs_before_analysis
    assert (
        analysis
        / "gate-policy"
        / gate_policy_artifact
        / "receipt.json"
    ).stat().st_mtime_ns == policy_mtime
    assert not any(
        key in reloaded.to_dict()
        for key in ("overall_score", "case_pass")
    )

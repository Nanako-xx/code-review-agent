from __future__ import annotations

import json
import hashlib
import shutil
import sys
from pathlib import Path

import pytest

import review_agent_eval.cli as cli_module
from review_agent_eval.adapters.subprocess_agent import SubprocessAgentAdapter
from review_agent_eval.artifacts import ArtifactStore
from review_agent_eval.cli import EXIT_OK, main
from review_agent_eval.datasets import CaseBank
from review_agent_eval.models import (
    RepositoryReviewTarget,
    ReviewTargetKind,
    canonical_json_bytes,
)
from review_agent_eval.orchestrator import EvaluationOrchestrator

from .test_agent_adapter import _AGENT_PROGRAM


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CORE_SUITE_ROOT = REPOSITORY_ROOT / "eval"
CORE_MANIFEST = Path("suites/core-regression/manifest.json")
TASK_ID = "core-py-001"


def _write_single_case_core_suite(tmp_path: Path) -> tuple[Path, Path]:
    """Copy canonical Core Case/Repository bytes into a one-Case Suite."""

    source_manifest = json.loads(
        (CORE_SUITE_ROOT / CORE_MANIFEST).read_text(encoding="utf-8")
    )
    case_entry = next(
        item for item in source_manifest["cases"] if item["task_id"] == TASK_ID
    )
    source_manifest["cases"] = [case_entry]

    suite_root = tmp_path / "core-suite"
    case_source = CORE_SUITE_ROOT / case_entry["path"]
    case_destination = suite_root / case_entry["path"]
    case_destination.parent.mkdir(parents=True)
    shutil.copy2(case_source, case_destination)

    case_payload = json.loads(case_source.read_text(encoding="utf-8"))
    repository_path = case_payload["input"]["review_target"]["repository"]["path"]
    shutil.copytree(
        CORE_SUITE_ROOT / repository_path,
        suite_root / repository_path,
        dirs_exist_ok=True,
    )
    (suite_root / "manifest.json").write_bytes(
        canonical_json_bytes(source_manifest)
    )

    agent_program = tmp_path / "subprocess-agent.py"
    agent_program.write_text(_AGENT_PROGRAM, encoding="utf-8")
    return suite_root, agent_program


def _common_arguments(suite_root: Path, tmp_path: Path) -> tuple[list[str], Path]:
    runs_root = tmp_path / ".eval-runs"
    return (
        [
            "--suite-root",
            str(suite_root),
            "--runs-root",
            str(runs_root),
            "--data-root",
            str(tmp_path / ".eval-data"),
            "--workspace-root",
            str(tmp_path / ".eval-workspaces"),
        ],
        runs_root,
    )


def _invoke_json(
    capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> tuple[int, dict]:
    code = main(arguments)
    captured = capsys.readouterr()
    assert captured.err == ""
    return code, json.loads(captured.out)


def _without_message(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "message"}


def test_repository_core_fixture_cli_lifecycle_resumes_committed_evaluation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_root, agent_program = _write_single_case_core_suite(tmp_path)
    common, runs_root = _common_arguments(suite_root, tmp_path)
    agent_command = [
        str(Path(sys.executable).resolve()),
        str(agent_program),
        "success",
        "{agent_id}",
        "{task_id}",
        "{trial_id}",
    ]
    prepare_arguments = [
        "prepare",
        *common,
        "--manifest",
        "manifest.json",
        "--run-instance-key",
        "task-5a-core-e2e",
        "--agent-adapter",
        "subprocess",
        "--json",
    ]
    for argument in agent_command:
        prepare_arguments.extend(("--agent-command", argument))

    code, prepared = _invoke_json(capsys, prepare_arguments)
    assert code == EXIT_OK
    assert prepared["status"] == "ok"
    assert prepared["suite_id"] == "core-regression"
    assert prepared["case_count"] == prepared["trial_count"] == 1
    run_id = prepared["run_id"]

    store = ArtifactStore(runs_root, create_root=False)
    bank = CaseBank.open(suite_root, "manifest.json")
    core_case = bank.evaluator_case(TASK_ID)
    target = core_case.input.review_target
    assert isinstance(target, RepositoryReviewTarget)
    assert target.kind is ReviewTargetKind.REPOSITORY

    run_config = store.load_run_config(run_id)
    run_manifest = store.load_run_manifest(run_id)
    case_snapshot = store.load_case_snapshot(run_id)
    assert run_config.suite.manifest_digest == bank.manifest_digest
    assert run_config.suite.case(TASK_ID) == bank.manifest.case(TASK_ID)
    assert case_snapshot.eval_input(TASK_ID) == core_case.eval_input()
    assert run_manifest.run_id == run_config.run_id == run_id
    assert len(run_manifest.trials) == 1
    plan = run_manifest.trials[0]
    trial_manifest = store.load_trial_manifest(run_id, TASK_ID, plan.trial_id)
    assert (
        plan.task_id
        == trial_manifest.task_id
        == core_case.task_id
        == TASK_ID
    )
    assert plan.canonical_case_digest == trial_manifest.canonical_case_digest
    assert plan.eval_input_digest == trial_manifest.eval_input_digest
    assert plan.eval_input_digest == core_case.eval_input().digest()
    assert plan.trial_id == trial_manifest.trial_id
    assert trial_manifest.target_kind is ReviewTargetKind.REPOSITORY

    code, agent_result = _invoke_json(
        capsys,
        [
            "run-agent",
            run_id,
            *common,
            "--manifest",
            "manifest.json",
            "--json",
        ],
    )
    assert code == EXIT_OK
    assert agent_result["run_status"] == "completed"
    assert agent_result["trials"] == [
        {
            "skipped": False,
            "status": "completed",
            "submission_status": "completed",
            "task_id": TASK_ID,
            "trial_id": plan.trial_id,
            "trial_index": 1,
        }
    ]

    materialization = store.load_trial_materialization(
        run_id, TASK_ID, plan.trial_id
    )
    submission = store.load_existing_submission(run_id, TASK_ID, plan.trial_id)
    terminal_state = store.load_trial_state(run_id, TASK_ID, plan.trial_id)
    assert materialization.eval_input.review_target == target
    materialized_target = materialization.eval_input.review_target
    assert isinstance(materialized_target, RepositoryReviewTarget)
    assert materialized_target.repository.base_revision == (
        target.repository.base_revision
    )
    assert materialized_target.repository.head_revision == (
        target.repository.head_revision
    )
    assert materialized_target.repository.base_revision != (
        materialized_target.repository.head_revision
    )
    assert materialization.manifest.run_id == run_id
    assert materialization.manifest.task_id == TASK_ID
    assert materialization.manifest.trial_id == plan.trial_id
    assert materialization.manifest.review_target_digest == target.digest()
    assert materialization.manifest.eval_input_digest == plan.eval_input_digest
    assert materialization.manifest.target_access.target_materialization_id == (
        materialization.manifest.materialization_id
    )
    assert target.repository.path is not None
    fixture_head = suite_root / target.repository.path / "head"
    expected_head_bindings = {
        path.relative_to(fixture_head).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in fixture_head.rglob("*")
        if path.is_file()
    }
    materialized_paths = [
        item.relative_path for item in materialization.manifest.files
    ]
    assert len(materialized_paths) == len(set(materialized_paths))
    materialized_head_bindings = {
        item.relative_path: (item.size_bytes, item.sha256)
        for item in materialization.manifest.files
    }
    assert materialized_head_bindings == expected_head_bindings
    assert materialization.manifest.target_access.readable_relative_paths == tuple(
        sorted(expected_head_bindings)
    )
    assert submission.target_materialization_id == (
        materialization.manifest.materialization_id
    )
    assert submission.eval_input_digest == plan.eval_input_digest
    assert submission.intent is not None
    assert submission.review is not None
    assert isinstance(submission.evidence, tuple)
    assert terminal_state.terminal_receipt is not None

    evaluate_arguments = [
        "evaluate",
        run_id,
        *common,
        "--manifest",
        "manifest.json",
        "--revision",
        "task-5a-v2",
        "--judge-provider",
        "fake",
        "--judge-model",
        "task-5a-fake-judge",
        "--json",
    ]
    code, first_evaluation = _invoke_json(capsys, evaluate_arguments)
    assert code == EXIT_OK
    assert first_evaluation["status"] == "ok"
    assert first_evaluation["trial_count"] == 1
    evaluation_id = first_evaluation["evaluation_id"]
    assert first_evaluation["trials"][0]["trial_id"] == plan.trial_id
    assert first_evaluation["trials"][0]["evaluation_id"] == evaluation_id

    first_bundle = store.load_evaluation_bundle(
        run_id, TASK_ID, plan.trial_id, evaluation_id
    )
    first_run_bundle = store.load_run_evaluation(run_id, evaluation_id)
    artifact_prefix = "cases/%s/trials/%s/evaluations/%s/" % (
        plan.case_path_id,
        plan.trial_id,
        evaluation_id,
    )
    artifact_paths = {
        item.relative_path for item in first_bundle.namespace.artifacts
    }
    assert artifact_paths == {
        artifact_prefix + "evaluator_execution_config.json",
        artifact_prefix + "intent_matches.json",
        artifact_prefix + "review_matches.json",
        artifact_prefix + "judge_input.json",
        artifact_prefix + "judge_output.json",
        artifact_prefix + "score.json",
        artifact_prefix + "report.md",
    }
    assert isinstance(first_bundle.intent_matches, dict)
    assert isinstance(first_bundle.review_matches, dict)
    assert isinstance(
        first_bundle.review_matches["evidence_integrity_results"], list
    )
    assert isinstance(first_bundle.judge_input, dict)
    assert isinstance(first_bundle.judge_output, dict)
    assert isinstance(first_bundle.score, dict)
    assert isinstance(first_bundle.report, str) and first_bundle.report
    assert isinstance(first_run_bundle.summary, dict)
    assert isinstance(first_run_bundle.report, str) and first_run_bundle.report
    assert first_bundle.submission_digest == submission.digest()
    assert first_bundle.canonical_case_digest == plan.canonical_case_digest
    assert first_bundle.trial_manifest_digest == trial_manifest.digest()

    before_trial_namespaces = store.list_evaluations(
        run_id, TASK_ID, plan.trial_id
    )
    before_run_namespaces = store.list_run_evaluations(run_id)
    before_score = first_bundle.score
    before_report = first_bundle.report
    terminal_receipt_id = terminal_state.terminal_receipt.receipt_id

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("resume re-executed Agent, evaluator, or Judge")

    with monkeypatch.context() as resume_guard:
        resume_guard.setattr(SubprocessAgentAdapter, "run", forbidden)
        resume_guard.setattr(EvaluationOrchestrator, "_evaluate_intent", forbidden)
        resume_guard.setattr(EvaluationOrchestrator, "_evaluate_review", forbidden)
        resume_guard.setattr(cli_module, "_judge_for", forbidden)
        code, resumed_evaluation = _invoke_json(capsys, evaluate_arguments)
    assert code == EXIT_OK
    assert _without_message(resumed_evaluation) == _without_message(
        first_evaluation
    )
    assert resumed_evaluation["evaluation_id"] == evaluation_id
    assert store.list_evaluations(
        run_id, TASK_ID, plan.trial_id
    ) == before_trial_namespaces
    assert store.list_run_evaluations(run_id) == before_run_namespaces
    resumed_bundle = store.load_evaluation_bundle(
        run_id, TASK_ID, plan.trial_id, evaluation_id
    )
    assert resumed_bundle.score == before_score
    assert resumed_bundle.report == before_report
    assert resumed_bundle.submission_digest == submission.digest()
    assert (
        store.load_trial_state(run_id, TASK_ID, plan.trial_id)
        .terminal_receipt.receipt_id
        == terminal_receipt_id
    )

    rejudge_arguments = list(evaluate_arguments)
    rejudge_arguments[rejudge_arguments.index("task-5a-fake-judge")] = (
        "task-5a-fake-judge-v2"
    )
    original_judge_for = cli_module._judge_for
    original_evaluate_intent = EvaluationOrchestrator._evaluate_intent
    original_evaluate_review = EvaluationOrchestrator._evaluate_review
    rejudge_spy = {
        "judge_for": 0,
        "intent": 0,
        "review": 0,
        "intent_requests": 0,
        "review_requests": 0,
    }

    def spy_judge_for(args: object, execution: object) -> object:
        rejudge_spy["judge_for"] += 1
        assert getattr(args, "judge_provider") == "fake"
        assert getattr(args, "judge_model") == "task-5a-fake-judge-v2"
        return original_judge_for(args, execution)

    def spy_evaluate_intent(
        orchestrator: EvaluationOrchestrator, *args: object, **kwargs: object
    ) -> object:
        rejudge_spy["intent"] += 1
        result = original_evaluate_intent(orchestrator, *args, **kwargs)
        rejudge_spy["intent_requests"] += len(result[1])
        return result

    def spy_evaluate_review(
        orchestrator: EvaluationOrchestrator, *args: object, **kwargs: object
    ) -> object:
        rejudge_spy["review"] += 1
        result = original_evaluate_review(orchestrator, *args, **kwargs)
        rejudge_spy["review_requests"] += len(result[1])
        return result

    with monkeypatch.context() as rejudge_guard:
        rejudge_guard.setattr(SubprocessAgentAdapter, "run", forbidden)
        rejudge_guard.setattr(cli_module, "_judge_for", spy_judge_for)
        rejudge_guard.setattr(
            EvaluationOrchestrator, "_evaluate_intent", spy_evaluate_intent
        )
        rejudge_guard.setattr(
            EvaluationOrchestrator, "_evaluate_review", spy_evaluate_review
        )
        code, rejudged_evaluation = _invoke_json(capsys, rejudge_arguments)
    assert code == EXIT_OK
    assert rejudge_spy["judge_for"] == 1
    assert rejudge_spy["intent"] == 1
    assert rejudge_spy["review"] == 1
    rejudged_evaluation_id = rejudged_evaluation["evaluation_id"]
    assert rejudged_evaluation_id != evaluation_id
    assert rejudged_evaluation["evaluation_revision"] == "task-5a-v2"
    assert rejudged_evaluation["trials"][0]["evaluation_id"] == (
        rejudged_evaluation_id
    )

    after_trial_namespaces = store.list_evaluations(
        run_id, TASK_ID, plan.trial_id
    )
    after_run_namespaces = store.list_run_evaluations(run_id)
    assert len(after_trial_namespaces) == len(before_trial_namespaces) + 1
    assert len(after_run_namespaces) == len(before_run_namespaces) + 1
    assert {item.evaluation_id for item in after_trial_namespaces} == {
        evaluation_id,
        rejudged_evaluation_id,
    }
    assert {item.evaluation_id for item in after_run_namespaces} == {
        evaluation_id,
        rejudged_evaluation_id,
    }

    old_bundle_after_rejudge = store.load_evaluation_bundle(
        run_id, TASK_ID, plan.trial_id, evaluation_id
    )
    assert old_bundle_after_rejudge.score == before_score
    assert old_bundle_after_rejudge.report == before_report
    assert old_bundle_after_rejudge.namespace == first_bundle.namespace
    old_run_bundle_after_rejudge = store.load_run_evaluation(run_id, evaluation_id)
    assert old_run_bundle_after_rejudge.namespace == first_run_bundle.namespace
    assert old_run_bundle_after_rejudge.summary == first_run_bundle.summary
    assert old_run_bundle_after_rejudge.report == first_run_bundle.report

    rejudged_bundle = store.load_evaluation_bundle(
        run_id, TASK_ID, plan.trial_id, rejudged_evaluation_id
    )
    assert rejudged_bundle.evaluation_revision == first_bundle.evaluation_revision
    assert rejudged_bundle.evaluator_execution.digest() != (
        first_bundle.evaluator_execution.digest()
    )
    assert rejudged_bundle.submission_digest == submission.digest()
    assert rejudged_bundle.canonical_case_digest == plan.canonical_case_digest
    assert rejudged_bundle.trial_manifest_digest == trial_manifest.digest()
    assert isinstance(rejudged_bundle.score, dict)
    assert isinstance(rejudged_bundle.report, str) and rejudged_bundle.report
    persisted_requests = rejudged_bundle.judge_input["requests"]
    persisted_results = rejudged_bundle.judge_output["results"]
    persisted_failure_count = sum(
        item["failure"] is not None for item in persisted_results
    )
    assert len(persisted_requests) == (
        rejudge_spy["intent_requests"] + rejudge_spy["review_requests"]
    )
    assert len(persisted_requests) > 0
    assert len(persisted_results) == len(persisted_requests)
    assert 0 <= persisted_failure_count <= len(persisted_requests)
    assert store.load_trial_materialization(
        run_id, TASK_ID, plan.trial_id
    ) == materialization
    assert store.load_existing_submission(
        run_id, TASK_ID, plan.trial_id
    ) == submission
    assert (
        store.load_trial_state(run_id, TASK_ID, plan.trial_id)
        .terminal_receipt.receipt_id
        == terminal_receipt_id
    )

    code, inspected = _invoke_json(
        capsys,
        [
            "inspect",
            run_id,
            *common,
            "--task-id",
            TASK_ID,
            "--trial-id",
            plan.trial_id,
            "--evaluation-id",
            rejudged_evaluation_id,
            "--format",
            "json",
        ],
    )
    assert code == EXIT_OK
    inspection = inspected["inspection"]
    assert inspection["source_bindings"] == {
        "canonical_case_digest": plan.canonical_case_digest,
        "eval_input_digest": plan.eval_input_digest,
        "evaluation_id": rejudged_evaluation_id,
        "evaluation_revision": "task-5a-v2",
        "evaluator_execution_digest": rejudged_bundle.evaluator_execution.digest(),
        "run_id": run_id,
        "submission_digest": submission.digest(),
        "task_id": TASK_ID,
        "trial_id": plan.trial_id,
        "trial_manifest_digest": trial_manifest.digest(),
    }
    assert inspection["submission"]["intent_present"] is True
    assert inspection["submission"]["review_present"] is True
    assert inspection["submission"]["evidence_count"] == len(
        submission.evidence
    )
    assert inspection["intent_evaluation"]["status"] == (
        rejudged_bundle.intent_matches["status"]
    )
    assert inspection["review_evaluation"]["status"] == (
        rejudged_bundle.review_matches["status"]
    )
    assert inspection["review_evaluation"][
        "evidence_integrity_result_count"
    ] == len(rejudged_bundle.review_matches["evidence_integrity_results"])
    assert inspection["score"]["artifact_digest"] == (
        rejudged_evaluation["trials"][0]["score_digest"]
    )
    assert inspection["report"]["available"] is True

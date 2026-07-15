from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import review_agent_eval.artifacts as artifact_module
from review_agent_eval.artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactRef,
    ArtifactSecurityError,
    ArtifactStateError,
    ArtifactStore,
    MAX_RUN_MANIFEST_BYTES,
    RunStatus,
    StageName,
    StageReceipt,
    TrialManifest,
    derive_receipt_id,
    load_existing_submission,
)
from review_agent_eval.cases import RunCaseSnapshot, SuiteManifest
from review_agent_eval.config import (
    EvalRunConfig,
    EvaluatorExecutionConfig,
    ResourceBudgets,
    SuiteRunConfig,
    derive_case_path_id,
    derive_evaluation_id,
)
from review_agent_eval.models import (
    EvalCase,
    EvalInput,
    EvalSubmission,
    FailureCode,
    SchemaError,
    SubmissionStatus,
    TrialStatus,
    canonical_json_bytes,
)

from .test_config import agent_config, budgets, evaluator_config


TASK_ID = "../../private/case:with/slashes"
BASE = "1" * 40
HEAD = "2" * 40


def make_case_snapshot(*, suite_version: str = "suite-v1") -> RunCaseSnapshot:
    eval_input = make_input()
    input_payload = eval_input.to_dict()
    case = EvalCase.from_dict(
        {
            "schema_version": "eval_case_v1",
            "task_id": TASK_ID,
            "case_version": 1,
            "source": {
                "suite": "artifact-suite",
                "origin": "hand_authored",
                "source_id": "artifact-case-source",
                "source_version": "source-v1",
                "source_uri": None,
                "license": None,
                "content_hash": "3" * 64,
            },
            "input": {
                "repository": input_payload["repository"],
                "review_request": input_payload["review_request"],
            },
            "clarification_script": {"max_rounds": 1, "answers": []},
            "intent_truth": {
                "scorable": True,
                "authority": "explicit_author_metadata",
                "expected_claims": [
                    {
                        "truth_id": "intent-preserve-access-control",
                        "dimension": "goal",
                        "text": "Preserve access control",
                        "required": True,
                    }
                ],
                "forbidden_claims": [],
                "clarification_policy": "not_required",
            },
            "review_truth": {
                "completeness": "closed_world",
                "novel_finding_policy": "forbid",
                "expected_findings": [],
                "known_invalid_findings": [],
            },
        }
    )
    raw = case.to_json().encode("utf-8")
    manifest = SuiteManifest.from_dict(
        {
            "schema_version": "suite_manifest_v1",
            "suite_id": "artifact-suite",
            "suite_version": suite_version,
            "source": {
                "kind": "core",
                "source_id": "artifact-suite-source",
                "source_version": "source-v1",
                "source_uri": None,
                "license": None,
                "content_hash": "4" * 64,
            },
            "cases": [
                {
                    "task_id": TASK_ID,
                    "case_version": 1,
                    "path": "cases/artifact-case.json",
                    "split": "regression",
                    "protocol_id": "native_repository",
                    "dimensions": [
                        {"name": "language", "value": "python"}
                    ],
                    "raw_file_size_bytes": len(raw),
                    "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
                    "canonical_case_digest": case.digest(),
                    "eval_input_digest": eval_input.digest(),
                    "truth_completeness": "closed_world",
                }
            ],
        }
    )
    return RunCaseSnapshot.build(
        manifest, ((manifest.cases[0], case),)
    )


def make_config(
    *,
    instance: str = "artifact-instance-001",
    evaluator=None,
    case_snapshot: RunCaseSnapshot | None = None,
    resource_budgets: ResourceBudgets | None = None,
):
    snapshot = case_snapshot or make_case_snapshot()
    return EvalRunConfig.create(
        run_instance_key=instance,
        agent=agent_config(),
        evaluator=evaluator or evaluator_config(),
        suite=SuiteRunConfig.from_case_snapshot(snapshot),
        trial_count=1,
        resource_budgets=resource_budgets or budgets(parallel=1),
    )


def make_input() -> EvalInput:
    return EvalInput.from_dict(
        {
            "schema_version": "eval_input_v1",
            "task_id": TASK_ID,
            "repository": {
                "source": "fixture",
                "path": "fixtures/repository",
                "url": None,
                "base_revision": BASE,
                "head_revision": HEAD,
            },
            "review_request": {
                "title": "Review the authorization change",
                "description": None,
                "user_intent": "Preserve access control",
                "review_focus": None,
                "linked_requirements": [],
                "project_rules": [],
                "existing_ci_evidence": [],
            },
        }
    )


def completed_submission(trial_id: str) -> EvalSubmission:
    return EvalSubmission.from_dict(
        {
            "schema_version": "eval_submission_v1",
            "task_id": TASK_ID,
            "agent_id": "agent-current",
            "trial_id": trial_id,
            "status": "completed",
            "intent": {
                "status": "sufficient",
                "goal": "Preserve access control",
                "acceptance_criteria": [],
                "scope": [],
                "constraints": [],
                "claims": [],
                "clarification_questions": [],
                "uncertainties": [],
            },
            "review": {"findings": [], "uncertainties": []},
            "evidence": [],
            "usage": {
                "elapsed_seconds": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "tool_calls": None,
                "cost_amount": None,
                "cost_currency": None,
            },
            "trace_ref": None,
            "failure": None,
        }
    )


def failed_submission(trial_id: str, message: str) -> EvalSubmission:
    return EvalSubmission.from_dict(
        {
            "schema_version": "eval_submission_v1",
            "task_id": TASK_ID,
            "agent_id": "agent-current",
            "trial_id": trial_id,
            "status": "failed",
            "intent": None,
            "review": None,
            "evidence": [],
            "usage": {
                "elapsed_seconds": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "tool_calls": None,
                "cost_amount": None,
                "cost_currency": None,
            },
            "trace_ref": None,
            "failure": {
                "code": "process_killed",
                "message": message,
                "retryable": False,
            },
        }
    )


def make_store(tmp_path: Path):
    store = ArtifactStore(tmp_path / ".eval-runs")
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot)
    manifest = store.create_run(config, snapshot)
    plan = manifest.trials[0]
    trial = store.load_trial_manifest(config.run_id, TASK_ID, plan.trial_id)
    return store, config, manifest, plan, trial


def complete_trial(store, config, plan):
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt is not None
    store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        attempt=running.active_attempt,
    )
    submission = completed_submission(plan.trial_id)
    state = store.finalize_submission(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        submission,
        attempt=running.active_attempt,
    )
    return submission, state


def test_final_layout_uses_immutable_plan_manifests_and_hashed_case_path(tmp_path):
    store, config, manifest, plan, trial = make_store(tmp_path)
    run_dir = store.root / config.run_id
    trial_dir = (
        run_dir / "cases" / plan.case_path_id / "trials" / plan.trial_id
    )

    assert manifest.schema_version == "eval_run_manifest_v1"
    assert trial.schema_version == "eval_trial_manifest_v1"
    expected_evaluator_execution_digest = (
        EvaluatorExecutionConfig.from_resource_budgets(
            config.evaluator, config.resource_budgets
        ).digest()
    )
    assert (
        manifest.initial_evaluator_execution_digest
        == expected_evaluator_execution_digest
    )
    assert (
        trial.initial_evaluator_execution_digest
        == expected_evaluator_execution_digest
    )
    assert plan.case_path_id == derive_case_path_id(TASK_ID)
    assert TASK_ID not in plan.manifest.relative_path
    assert (run_dir / "run_config.json").is_file()
    assert (run_dir / "case_snapshot.json").is_file()
    assert (run_dir / "run_manifest.json").is_file()
    assert (trial_dir / "trial_manifest.json").is_file()
    assert "status" not in manifest.to_dict()
    assert "status" not in trial.to_dict()
    assert "submission" not in trial.to_dict()
    assert "receipts" not in trial.to_dict()
    assert "os.replace" not in inspect.getsource(artifact_module)

    run_bytes = (run_dir / "run_manifest.json").read_bytes()
    trial_bytes = (trial_dir / "trial_manifest.json").read_bytes()
    assert run_bytes == canonical_json_bytes(manifest)
    assert trial_bytes == canonical_json_bytes(trial)
    assert hashlib.sha256((run_dir / "run_config.json").read_bytes()).hexdigest() == (
        manifest.run_config.sha256
    )
    assert hashlib.sha256((run_dir / "case_snapshot.json").read_bytes()).hexdigest() == (
        manifest.case_snapshot.sha256
    )
    assert store.load_case_snapshot(config.run_id) == make_case_snapshot()

    # The complete plan already exists; create_trial is read-only and a second
    # run writer fails without changing either plan manifest.
    assert store.create_trial(config.run_id, TASK_ID, 1) == trial
    assert (run_dir / "run_manifest.json").read_bytes() == run_bytes
    assert (trial_dir / "trial_manifest.json").read_bytes() == trial_bytes
    with pytest.raises(ArtifactConflictError):
        store.create_run(config, make_case_snapshot())
    assert (run_dir / "run_manifest.json").read_bytes() == run_bytes

    with pytest.raises(FrozenInstanceError):
        trial.seed = 0


def test_run_creation_requires_the_exact_verified_case_snapshot(tmp_path):
    store = ArtifactStore(tmp_path / ".eval-runs")
    expected = make_case_snapshot()
    config = make_config(case_snapshot=expected)
    different = make_case_snapshot(suite_version="suite-v2")

    with pytest.raises(SchemaError, match="Run Config suite"):
        store.create_run(config, different)
    assert not (store.root / config.run_id).exists()


def test_loading_run_cross_validates_snapshot_config_and_manifest(tmp_path):
    store, config, manifest, plan, _ = make_store(tmp_path)
    snapshot_path = store.root / config.run_id / "case_snapshot.json"
    snapshot_path.write_bytes(snapshot_path.read_bytes() + b" ")

    with pytest.raises(ArtifactIntegrityError, match="size|hash|canonical"):
        store.load_run_config(config.run_id)
    with pytest.raises(ArtifactIntegrityError, match="size|hash|canonical"):
        store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    with pytest.raises(ArtifactIntegrityError, match="size|hash|canonical"):
        store.load_trial_state(config.run_id, TASK_ID, plan.trial_id)
    assert manifest.case_snapshot.relative_path == "case_snapshot.json"


@pytest.mark.parametrize(
    "field",
    ["agent_config_digest", "initial_evaluator_execution_digest"],
)
def test_trial_manifest_execution_digests_must_match_verified_run(
    tmp_path, field
):
    store, config, _, plan, _ = make_store(tmp_path)
    run_dir = store.root / config.run_id
    trial_path = run_dir / Path(*plan.manifest.relative_path.split("/"))
    trial_payload = json.loads(trial_path.read_text(encoding="utf-8"))
    trial_payload[field] = "0" * 64
    trial_bytes = canonical_json_bytes(trial_payload)
    trial_path.write_bytes(trial_bytes)

    run_manifest_path = run_dir / "run_manifest.json"
    run_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_payload["trials"][0]["manifest"]["sha256"] = hashlib.sha256(
        trial_bytes
    ).hexdigest()
    run_payload["trials"][0]["manifest"]["size_bytes"] = len(trial_bytes)
    run_manifest_path.write_bytes(canonical_json_bytes(run_payload))

    with pytest.raises(ArtifactIntegrityError, match="TrialManifest"):
        store.load_trial_manifest(config.run_id, TASK_ID, plan.trial_id)


def test_artifact_api_has_no_untyped_write_or_speculative_aliases(tmp_path):
    store, _, _, _, _ = make_store(tmp_path)

    assert not hasattr(store, "write_json_artifact")
    assert not hasattr(store, "resume_trial")
    assert not hasattr(store, "write_evaluation_artifacts")
    assert not hasattr(store, "write_run_evaluation_outputs")
    assert not hasattr(store, "for_workspace")
    assert not hasattr(artifact_module, "ArtifactDescriptor")
    assert not hasattr(artifact_module, "RunTrialEntry")


def test_directory_fsync_capability_is_explicit(tmp_path):
    store, _, _, _, _ = make_store(tmp_path)
    assert store.directory_fsync_supported is (os.name != "nt")
    assert artifact_module.DIRECTORY_FSYNC_SUPPORTED is (os.name != "nt")


def test_control_plane_snapshot_is_not_limited_by_agent_artifact_budget(tmp_path):
    store = ArtifactStore(tmp_path / ".eval-runs")
    snapshot = make_case_snapshot()
    tiny_agent_artifacts = ResourceBudgets(
        agent_timeout_seconds=900,
        evaluator_timeout_seconds=300,
        max_agent_output_bytes=1024,
        max_trace_bytes=1024,
        max_execution_artifact_file_bytes=1,
        max_execution_artifact_total_bytes=4096,
        max_parallel_trials=1,
    )
    config = make_config(
        case_snapshot=snapshot,
        resource_budgets=tiny_agent_artifacts,
    )

    manifest = store.create_run(config, snapshot)
    assert manifest.case_snapshot.size_bytes > 1
    assert store.load_case_snapshot(config.run_id) == snapshot


def test_atomic_create_if_absent_fsync_hash_and_parallel_conflict(tmp_path, monkeypatch):
    store, config, _, _, _ = make_store(tmp_path)
    fsync_calls = []
    real_fsync = artifact_module.os.fsync

    def recording_fsync(descriptor):
        fsync_calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(artifact_module.os, "fsync", recording_fsync)
    first = store._write_json(config.run_id, "auxiliary/one.json", {"v": 1})
    path = store.root / config.run_id / Path(*first.relative_path.split("/"))
    original = path.read_bytes()
    assert first.sha256 == hashlib.sha256(original).hexdigest()
    assert first.size_bytes == len(original)
    assert fsync_calls
    with pytest.raises(ArtifactConflictError):
        store._write_json(config.run_id, "auxiliary/one.json", {"v": 2})
    assert path.read_bytes() == original

    def writer(value):
        try:
            return store._write_json(
                config.run_id, "auxiliary/concurrent.json", {"writer": value}
            )
        except ArtifactConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(writer, (1, 2)))
    assert sum(isinstance(item, ArtifactRef) for item in results) == 1
    assert sum(isinstance(item, ArtifactConflictError) for item in results) == 1


def test_receipts_derive_pending_running_incomplete_and_terminal_state(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    run_manifest_before = (store.root / config.run_id / "run_manifest.json").read_bytes()
    trial_manifest_path = store.root / config.run_id / Path(
        *plan.manifest.relative_path.split("/")
    )
    trial_manifest_before = trial_manifest_path.read_bytes()

    assert store.load_trial_state(config.run_id, TASK_ID, plan.trial_id).status is TrialStatus.PENDING
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.status is TrialStatus.RUNNING
    incomplete = store.mark_trial_incomplete(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        attempt=running.active_attempt,
    )
    assert incomplete.status is TrialStatus.INCOMPLETE
    assert not (
        store.root
        / config.run_id
        / "cases"
        / trial.case_path_id
        / "trials"
        / plan.trial_id
        / "submission.json"
    ).exists()
    with pytest.raises(ArtifactStateError):
        store.load_existing_submission(config.run_id, TASK_ID, plan.trial_id)

    resumed = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert resumed.status is TrialStatus.RUNNING
    assert resumed.active_attempt == 2
    prepare = store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        attempt=resumed.active_attempt,
    )
    assert prepare.stage is StageName.PREPARE
    submission = completed_submission(plan.trial_id)
    terminal = store.finalize_submission(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        submission,
        attempt=resumed.active_attempt,
    )
    assert terminal.status is TrialStatus.COMPLETED
    assert terminal.terminal_receipt is not None
    assert terminal.terminal_receipt.stage is StageName.AGENT
    assert terminal.terminal_receipt.config_digest == config.agent_config_digest
    assert store.load_existing_submission(config.run_id, TASK_ID, plan.trial_id) == submission
    assert store.load_run_state(config.run_id).status is RunStatus.COMPLETED

    # Plan manifests never participate in status transitions.
    assert (store.root / config.run_id / "run_manifest.json").read_bytes() == run_manifest_before
    assert trial_manifest_path.read_bytes() == trial_manifest_before
    with pytest.raises(ArtifactConflictError):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=resumed.active_attempt,
        )


def test_stale_attempt_cannot_commit_as_the_current_retry(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    first = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    store.mark_trial_incomplete(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        attempt=first.active_attempt,
    )
    second = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert second.active_attempt == 2

    with pytest.raises(ArtifactConflictError, match="stale"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            completed_submission(plan.trial_id),
            attempt=first.active_attempt,
        )
    assert store.load_trial_state(
        config.run_id, TASK_ID, plan.trial_id
    ).active_attempt == 2


def test_disk_replay_rejects_completed_terminal_without_prepare(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    base = "cases/%s/trials/%s" % (trial.case_path_id, plan.trial_id)
    submission_ref = store._write_json(
        config.run_id,
        "%s/submission.json" % base,
        completed_submission(plan.trial_id),
    )
    receipt = StageReceipt.create(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        stage=StageName.AGENT,
        config_digest=trial.agent_config_digest,
        artifacts=(submission_ref,),
        attempt=1,
        terminal_status=TrialStatus.COMPLETED,
    )
    store._write_json(
        config.run_id,
        "%s/receipts/terminal.json" % base,
        receipt,
    )

    with pytest.raises(ArtifactIntegrityError, match="prepare"):
        store.load_run_state(config.run_id)


def test_disk_replay_hydrates_and_binds_terminal_submission(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        attempt=running.active_attempt,
    )
    base = "cases/%s/trials/%s" % (trial.case_path_id, plan.trial_id)
    failed = failed_submission(plan.trial_id, "process stopped")
    submission_ref = store._write_json(
        config.run_id, "%s/submission.json" % base, failed
    )
    mismatched = StageReceipt.create(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        stage=StageName.AGENT,
        config_digest=trial.agent_config_digest,
        artifacts=(submission_ref,),
        attempt=1,
        terminal_status=TrialStatus.COMPLETED,
    )
    store._write_json(
        config.run_id,
        "%s/receipts/terminal.json" % base,
        mismatched,
    )

    with pytest.raises(ArtifactIntegrityError, match="Submission"):
        store.load_existing_submission(config.run_id, TASK_ID, plan.trial_id)


def test_attempt_receipt_directories_must_be_contiguous(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    start = StageReceipt.create(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        stage=StageName.START,
        config_digest=trial.agent_config_digest,
        attempt=2,
    )
    base = "cases/%s/trials/%s/receipts" % (
        trial.case_path_id,
        plan.trial_id,
    )
    store._write_json(
        config.run_id,
        "%s/attempt-0002/start.json" % base,
        start,
    )

    with pytest.raises(ArtifactIntegrityError, match="contiguous"):
        store.load_trial_state(config.run_id, TASK_ID, plan.trial_id)


def test_resume_only_commits_missing_legal_receipts_and_never_rewrites_submission(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    base = "cases/%s/trials/%s" % (trial.case_path_id, plan.trial_id)

    input_ref = store._write_json(
        config.run_id, "%s/input.json" % base, make_input()
    )
    input_path = store.root / config.run_id / Path(*input_ref.relative_path.split("/"))
    input_bytes = input_path.read_bytes()
    first_plan = store.recover_trial(config.run_id, TASK_ID, plan.trial_id)
    assert first_plan.status is TrialStatus.INCOMPLETE
    assert first_plan.completed_stages == (StageName.PREPARE,)
    assert first_plan.missing_stages == (StageName.AGENT,)
    assert input_path.read_bytes() == input_bytes

    store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    submission = completed_submission(plan.trial_id)
    submission_ref = store._write_json(
        config.run_id, "%s/submission.json" % base, submission
    )
    submission_path = store.root / config.run_id / Path(
        *submission_ref.relative_path.split("/")
    )
    submission_bytes = submission_path.read_bytes()
    second_plan = store.recover_trial(config.run_id, TASK_ID, plan.trial_id)

    assert second_plan.terminal is True
    assert second_plan.status is TrialStatus.COMPLETED
    assert submission_path.read_bytes() == submission_bytes
    terminal_path = submission_path.parent / "receipts" / "terminal.json"
    assert terminal_path.is_file()
    assert store.load_existing_submission(config.run_id, TASK_ID, plan.trial_id) == submission


def test_abandonment_atomically_commits_failed_process_killed_submission(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    recovery = store.recover_trial(config.run_id, TASK_ID, plan.trial_id)
    assert recovery.status is TrialStatus.INCOMPLETE

    state = store.abandon_trial(config.run_id, TASK_ID, plan.trial_id)
    submission = store.load_existing_submission(config.run_id, TASK_ID, plan.trial_id)
    assert state.status is TrialStatus.FAILED
    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.PROCESS_KILLED
    assert state.terminal_receipt is not None
    assert state.terminal_receipt.failure_code is FailureCode.PROCESS_KILLED


def test_load_existing_submission_is_strictly_read_only(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    submission, _ = complete_trial(store, config, plan)
    run_dir = store.root / config.run_id
    before = {
        path.relative_to(run_dir).as_posix(): path.stat().st_mtime_ns
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    assert store.load_existing_submission(config.run_id, TASK_ID, plan.trial_id) == submission
    assert load_existing_submission(
        store.root, config.run_id, TASK_ID, plan.trial_id
    ) == submission
    after = {
        path.relative_to(run_dir).as_posix(): path.stat().st_mtime_ns
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_read_only_submission_loader_never_creates_a_missing_root(tmp_path):
    missing_root = tmp_path / "missing" / ".eval-runs"
    with pytest.raises(ArtifactIntegrityError, match="does not exist"):
        load_existing_submission(
            missing_root,
            "run-" + "0" * 64,
            TASK_ID,
            "trial-" + "0" * 64,
        )
    assert not missing_root.exists()


def test_evaluator_outputs_and_reports_are_versioned_without_mutating_submission(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    _, _ = complete_trial(store, config, plan)
    submission_path = (
        store.root
        / config.run_id
        / "cases"
        / trial.case_path_id
        / "trials"
        / plan.trial_id
        / "submission.json"
    )
    original_submission = submission_path.read_bytes()
    changed_evaluator = evaluator_config(judge_version="judge-v2", model="judge-2")
    changed_execution = EvaluatorExecutionConfig.from_resource_budgets(
        changed_evaluator, config.resource_budgets
    )

    first = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=changed_execution,
        revision="revision-1",
        intent_matches={"matches": []},
        review_matches={"matches": []},
        judge_input={"claims": []},
        judge_output={"status": "graded"},
        score={"total": 0},
        report="# Evaluation\n",
    )
    second = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=changed_execution,
        revision="revision-2",
        intent_matches={"matches": []},
        review_matches={"matches": []},
        judge_input={"claims": []},
        judge_output={"status": "graded"},
        score={"total": 1},
        report="# Evaluation 2\n",
    )
    changed_timeout_execution = EvaluatorExecutionConfig.create(
        evaluator=changed_evaluator,
        evaluator_timeout_seconds=(
            changed_execution.evaluator_timeout_seconds + 1
        ),
        max_execution_artifact_file_bytes=(
            changed_execution.max_execution_artifact_file_bytes
        ),
        max_execution_artifact_total_bytes=(
            changed_execution.max_execution_artifact_total_bytes
        ),
    )
    changed_timeout = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=changed_timeout_execution,
        revision="revision-1",
        intent_matches={"matches": []},
        review_matches={"matches": []},
        judge_input={"claims": []},
        judge_output={"status": "graded"},
        score={"total": 0},
    )

    assert first.stage is StageName.EVALUATOR
    assert first.evaluation_revision == "revision-1"
    evaluator_execution_digest = changed_execution.digest()
    assert first.config_digest == evaluator_execution_digest
    assert first.config_digest != config.agent_config_digest
    assert first.evaluation_id != second.evaluation_id
    assert first.evaluation_id != changed_timeout.evaluation_id
    assert submission_path.read_bytes() == original_submission
    for receipt in (first, second):
        evaluation_dir = submission_path.parent / "evaluations" / receipt.evaluation_id
        for name in (
            "evaluator_execution_config.json",
            "intent_matches.json",
            "review_matches.json",
            "judge_input.json",
            "judge_output.json",
            "score.json",
            "report.md",
            "receipt.json",
        ):
            assert (evaluation_dir / name).is_file()

    tampered = first.to_dict()
    tampered["evaluation_revision"] = "different-revision"
    tampered["receipt_id"] = derive_receipt_id(
        first.run_id,
        first.task_id,
        first.trial_id,
        first.stage,
        first.config_digest,
        attempt=first.attempt,
        evaluation_id=first.evaluation_id,
        evaluation_revision="different-revision",
    )
    with pytest.raises(SchemaError, match="evaluation_id"):
        StageReceipt.from_dict(tampered)

    expected_execution_bytes = sum(
        path.stat().st_size
        for path in submission_path.parent.rglob("*")
        if path.is_file()
        and "evaluations" in path.parts
        and path.name != "receipt.json"
        and ".locks" not in path.parts
    )
    assert (
        store._execution_artifact_total_bytes(config.run_id)
        == expected_execution_bytes
    )

    with pytest.raises(ArtifactConflictError):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=changed_execution,
            revision="revision-1",
            intent_matches={},
            review_matches={},
            judge_input={},
            judge_output={},
            score={},
        )


def test_evaluation_total_budget_is_preflighted_before_artifacts_publish(tmp_path):
    constrained = ResourceBudgets(
        agent_timeout_seconds=900,
        evaluator_timeout_seconds=300,
        max_agent_output_bytes=256,
        max_trace_bytes=256,
        max_execution_artifact_file_bytes=650,
        max_execution_artifact_total_bytes=650,
        max_parallel_trials=1,
    )
    snapshot = make_case_snapshot()
    config = make_config(
        case_snapshot=snapshot,
        resource_budgets=constrained,
    )
    store = ArtifactStore(tmp_path / ".eval-runs")
    manifest = store.create_run(config, snapshot)
    plan = manifest.trials[0]
    complete_trial(store, config, plan)
    evaluator = evaluator_config()
    evaluator_execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator, constrained
    )
    evaluation_id = derive_evaluation_id(
        config.run_id, evaluator_execution.digest(), "budgeted"
    )

    with pytest.raises(ArtifactIntegrityError, match="cumulative"):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=evaluator_execution,
            revision="budgeted",
            intent_matches={"matches": []},
            review_matches={"matches": []},
            judge_input={"claims": []},
            judge_output={"status": "graded"},
            score={"total": 0},
            report="# Evaluation\n",
        )

    evaluation_dir = (
        store.root
        / config.run_id
        / "cases"
        / plan.case_path_id
        / "trials"
        / plan.trial_id
        / "evaluations"
        / evaluation_id
    )
    assert not (evaluation_dir / "evaluator_execution_config.json").exists()
    assert not (evaluation_dir / "receipt.json").exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "sk-test-secret-value"},
        {"url": "https://user:password@example.test"},
        {"raw_reasoning": "hidden"},
        {"environment": {"HOME": "/private"}},
        {"error": "HOME=/private PATH=/bin"},
        {"error": "raw reasoning: hidden internal trace"},
        {"token": "ordinary-secret-value"},
        {"error": "AWS_SECRET_ACCESS_KEY=ordinary-secret"},
        {"error": "Authorization: Basic dXNlcjpwYXNz"},
        {"reasoning_content": "private intermediate steps"},
        {"error": "<think>private intermediate steps</think>"},
        {
            "error": (
                "Path=C:\\Windows\n"
                "UserProfile=C:\\Users\\private\n"
                "ComSpec=C:\\Windows\\System32\\cmd.exe"
            )
        },
        {
            "diagnostic": {
                "HOME": "/private",
                "PATH": "/bin",
                "USER": "private-user",
            }
        },
    ],
)
def test_artifacts_reject_secrets_userinfo_env_and_hidden_reasoning(tmp_path, payload):
    store, config, _, _, _ = make_store(tmp_path)
    with pytest.raises(SchemaError):
        store._write_json(config.run_id, "auxiliary/unsafe.json", payload)
    assert not (store.root / config.run_id / "auxiliary" / "unsafe.json").exists()


@pytest.mark.parametrize(
    "text",
    [
        "AWS_SECRET_ACCESS_KEY=ordinary-secret",
        "Authorization: Basic dXNlcjpwYXNz",
        "<think>private intermediate steps</think>",
        "Path=C:\\Windows\nUserProfile=C:\\Users\\private",
    ],
)
def test_text_artifacts_reject_credentials_env_and_raw_reasoning(tmp_path, text):
    store, config, _, _, _ = make_store(tmp_path)
    with pytest.raises(SchemaError):
        store._write_text(
            config.run_id,
            "evaluations/unsafe/report.md",
            text,
            maximum=4096,
        )
    assert not (
        store.root / config.run_id / "evaluations" / "unsafe" / "report.md"
    ).exists()


def test_submission_failure_and_report_cannot_persist_secret_values(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    with pytest.raises(SchemaError):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            failed_submission(plan.trial_id, "Bearer abcdefghijklmnop"),
            attempt=running.active_attempt,
        )

    # Complete safely, then reject an unsafe evaluator report before its bytes
    # can be persisted in report.md.
    store.abandon_trial(config.run_id, TASK_ID, plan.trial_id)
    unsafe_execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(), config.resource_budgets
    )
    with pytest.raises(SchemaError):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=unsafe_execution,
            revision="unsafe-report",
            intent_matches={},
            review_matches={},
            judge_input={},
            judge_output={},
            score={},
            report="https://user:password@example.test/report",
        )
    evaluation_id = derive_evaluation_id(
        config.run_id,
        EvaluatorExecutionConfig.from_resource_budgets(
            evaluator_config(), config.resource_budgets
        ).digest(),
        "unsafe-report",
    )
    report_path = (
        store.root
        / config.run_id
        / "cases"
        / derive_case_path_id(TASK_ID)
        / "trials"
        / plan.trial_id
        / "evaluations"
        / evaluation_id
        / "report.md"
    )
    assert not report_path.exists()


def test_hash_mismatch_is_rejected_for_committed_submission(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    complete_trial(store, config, plan)
    submission_path = (
        store.root
        / config.run_id
        / "cases"
        / trial.case_path_id
        / "trials"
        / plan.trial_id
        / "submission.json"
    )
    submission_path.write_bytes(submission_path.read_bytes() + b" ")

    with pytest.raises(ArtifactIntegrityError, match="size|hash|canonical"):
        store.load_existing_submission(config.run_id, TASK_ID, plan.trial_id)


def test_single_file_and_cumulative_read_limits_fail_closed(tmp_path):
    store, config, _, _, _ = make_store(tmp_path)
    first = store._write_json(
        config.run_id, "auxiliary/large-1.json", {"blob": "x" * 80}
    )
    second = store._write_json(
        config.run_id, "auxiliary/large-2.json", {"blob": "y" * 80}
    )

    small_file_reader = ArtifactStore(
        store.root, max_file_bytes=64, max_total_read_bytes=256
    )
    with pytest.raises(ArtifactIntegrityError, match="single-file"):
        small_file_reader.read_json_artifact(config.run_id, first)

    cumulative_reader = ArtifactStore(
        store.root, max_file_bytes=128, max_total_read_bytes=150
    )
    with pytest.raises(ArtifactIntegrityError, match="cumulative"):
        cumulative_reader.read_json_artifacts(config.run_id, (first, second))


def test_noncanonical_json_is_rejected_even_when_descriptor_hash_matches(tmp_path):
    store, config, _, _, _ = make_store(tmp_path)
    path = store.root / config.run_id / "auxiliary" / "noncanonical.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b'{"a": 1}'
    path.write_bytes(raw)
    ref = ArtifactRef(
        relative_path="auxiliary/noncanonical.json",
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )
    with pytest.raises(ArtifactIntegrityError, match="canonical"):
        store.read_json_artifact(config.run_id, ref)


def test_schema_specific_limit_rejects_before_reading_file_bytes(tmp_path, monkeypatch):
    store, config, _, _, _ = make_store(tmp_path)
    target = store.root / config.run_id / "run_manifest.json"
    target_key = os.path.normcase(os.path.abspath(target))
    real_lstat = artifact_module.os.lstat
    read_calls = []

    class OversizedStat:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_size = MAX_RUN_MANIFEST_BYTES + 1

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def fake_lstat(path):
        result = real_lstat(path)
        if os.path.normcase(os.path.abspath(os.fspath(path))) == target_key:
            return OversizedStat(result)
        return result

    real_read = artifact_module.os.read

    def recording_read(descriptor, amount):
        read_calls.append((descriptor, amount))
        return real_read(descriptor, amount)

    monkeypatch.setattr(artifact_module.os, "lstat", fake_lstat)
    monkeypatch.setattr(artifact_module.os, "read", recording_read)
    with pytest.raises(ArtifactIntegrityError, match="single-file"):
        store.load_run_manifest(config.run_id)
    assert read_calls == []


def test_symlink_and_special_file_reads_are_rejected(tmp_path, monkeypatch):
    store, config, _, _, _ = make_store(tmp_path)
    ref = store._write_json(config.run_id, "auxiliary/link.json", {"ok": True})
    path = store.root / config.run_id / Path(*ref.relative_path.split("/"))
    outside = tmp_path / "outside.json"
    outside.write_bytes(canonical_json_bytes({"ok": True}))
    original = path.read_bytes()
    path.unlink()
    try:
        os.symlink(outside, path)
    except (OSError, NotImplementedError):
        # Windows may require Developer Mode for creating a real symlink.  Keep
        # the test deterministic by presenting the same lstat metadata shape.
        path.write_bytes(original)
        target_key = os.path.normcase(os.path.abspath(path))
        real_lstat = artifact_module.os.lstat

        class SymlinkStat:
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self.st_mode = stat.S_IFLNK | 0o777

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

        def fake_lstat(value):
            result = real_lstat(value)
            if os.path.normcase(os.path.abspath(os.fspath(value))) == target_key:
                return SymlinkStat(result)
            return result

        monkeypatch.setattr(artifact_module.os, "lstat", fake_lstat)
    with pytest.raises(ArtifactSecurityError):
        store.read_json_artifact(config.run_id, ref)

    if os.name != "nt" and hasattr(os, "mkfifo"):
        fifo = store.root / config.run_id / "auxiliary" / "fifo.json"
        os.mkfifo(fifo)
        fifo_ref = ArtifactRef(
            relative_path="auxiliary/fifo.json", sha256="0" * 64, size_bytes=0
        )
        with pytest.raises(ArtifactSecurityError):
            store.read_json_artifact(config.run_id, fifo_ref)


def test_windows_reparse_attribute_is_rejected_explicitly(tmp_path, monkeypatch):
    store, config, _, _, _ = make_store(tmp_path)
    ref = store._write_json(
        config.run_id, "auxiliary/reparse.json", {"ok": True}
    )
    target = store.root / config.run_id / Path(*ref.relative_path.split("/"))
    target_key = os.path.normcase(os.path.abspath(target))
    real_lstat = artifact_module.os.lstat

    class ReparseStat:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.st_file_attributes = getattr(wrapped, "st_file_attributes", 0) | 0x400

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def fake_lstat(path):
        result = real_lstat(path)
        if os.path.normcase(os.path.abspath(os.fspath(path))) == target_key:
            return ReparseStat(result)
        return result

    monkeypatch.setattr(artifact_module.os, "lstat", fake_lstat)
    with pytest.raises(ArtifactSecurityError, match="reparse"):
        store.read_json_artifact(config.run_id, ref)


def test_opened_handle_path_must_resolve_to_the_exact_artifact(tmp_path, monkeypatch):
    store, config, _, _, _ = make_store(tmp_path)
    ref = store._write_json(
        config.run_id, "auxiliary/opened-path.json", {"ok": True}
    )
    outside = tmp_path / "outside.json"
    outside.write_bytes(canonical_json_bytes({"ok": True}))
    monkeypatch.setattr(
        artifact_module, "_windows_descriptor_path", lambda _descriptor: outside
    )

    with pytest.raises(ArtifactSecurityError, match="outside|unexpected"):
        store.read_json_artifact(config.run_id, ref)


def test_missing_inode_is_not_treated_as_same_file():
    class Identity:
        st_dev = 0
        st_ino = 0

    assert artifact_module._same_file(Identity(), Identity()) is False


def test_parallel_trial_start_writers_fail_closed(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)

    def start():
        try:
            return store.start_trial(config.run_id, TASK_ID, plan.trial_id)
        except (ArtifactConflictError, ArtifactStateError) as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: start(), (1, 2)))
    assert sum(hasattr(item, "status") for item in results) == 1
    assert sum(isinstance(item, (ArtifactConflictError, ArtifactStateError)) for item in results) == 1
    assert store.load_trial_state(config.run_id, TASK_ID, plan.trial_id).status is TrialStatus.RUNNING


def test_trial_manifest_rejects_status_or_submission_fields_as_non_plan_data(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    payload = trial.to_dict()
    assert "canonical_case_digest" in payload
    assert "case_digest" not in payload
    assert plan.canonical_case_digest == payload["canonical_case_digest"]
    payload["status"] = "incomplete"
    with pytest.raises(SchemaError, match="unknown field"):
        TrialManifest.from_dict(payload)

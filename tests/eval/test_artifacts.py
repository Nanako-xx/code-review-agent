from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import review_agent_eval.artifacts as artifact_module
from review_agent_eval.artifacts import (
    AgentVisibleFileBinding,
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactRef,
    ArtifactSecurityError,
    ArtifactStateError,
    ArtifactStore,
    MAX_RUN_MANIFEST_BYTES,
    RunManifest,
    RunStatus,
    StageName,
    StageReceipt,
    TrialMaterializationManifest,
    TrialManifest,
    derive_receipt_id,
    load_existing_submission,
)
from review_agent_eval.cases import RunCaseSnapshot, SuiteManifest, WireContractV2
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
    TraceType,
    TrialStatus,
    UnsupportedProtocolVersionError,
    canonical_json_bytes,
)

from .test_config import (
    _judge_input_context_payload,
    _model_turn_context_block_payload,
    _review_context_payload,
    adapter_capabilities,
    agent_config,
    budgets,
    evaluator_config,
    matcher_config,
)


TASK_ID = "../../private/case:with/slashes"
BASE = "1" * 40
HEAD = "2" * 40


def make_case_snapshot(*, suite_version: str = "suite-v1") -> RunCaseSnapshot:
    eval_input = make_input()
    input_payload = eval_input.to_dict()
    case = EvalCase.from_dict(
        {
            "schema_version": "eval_case_v2",
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
                "review_target": input_payload["review_target"],
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
            "review_evaluator_context": {"truth_contexts": []},
        }
    )
    raw = case.to_json().encode("utf-8")
    manifest = SuiteManifest.from_dict(
        {
            "schema_version": "suite_manifest_v2",
            "suite_id": "artifact-suite",
            "suite_version": suite_version,
            "wire_contract": {
                "case_schema_version": "eval_case_v2",
                "input_schema_version": "eval_input_v2",
                "submission_schema_version": "eval_submission_v2",
                "review_target_kind": "repository",
                "materializer_protocol": "repository-materializer-v2",
            },
            "source": {
                "kind": "core",
                "source_id": "artifact-suite-source",
                "source_version": "source-v1",
                "source_uri": None,
                "license": None,
                "content_hash": "4" * 64,
                "preparation_binding": None,
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
    trial_count: int = 1,
):
    snapshot = case_snapshot or make_case_snapshot()
    return EvalRunConfig.create(
        run_instance_key=instance,
        agent=agent_config(),
        clarification_matcher=matcher_config(),
        evaluator=evaluator or evaluator_config(),
        suite=SuiteRunConfig.from_case_snapshot(snapshot),
        adapter_capabilities=adapter_capabilities(),
        trial_count=trial_count,
        resource_budgets=resource_budgets or budgets(parallel=1),
    )


def make_input() -> EvalInput:
    return EvalInput.from_dict(
        {
            "schema_version": "eval_input_v2",
            "task_id": TASK_ID,
            "review_target": {
                "kind": "repository",
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
            },
        }
    )


def make_materialization(
    config: EvalRunConfig,
    plan,
    *,
    attempt: int,
    eval_input: EvalInput | None = None,
) -> TrialMaterializationManifest:
    bound_input = eval_input or make_input()
    visible = AgentVisibleFileBinding(
        role="repository_file",
        relative_path="target/repository/app.py",
        size_bytes=17,
        sha256=hashlib.sha256(b"print('review')\n").hexdigest(),
    )
    return TrialMaterializationManifest.create(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        attempt=attempt,
        eval_input_digest=bound_input.digest(),
        review_target_digest=bound_input.review_target.digest(),
        wire_contract=config.wire_contract,
        suite_preparation_binding_digest=(
            config.suite_preparation_binding_digest
        ),
        prepared_source_id="prepared-repository-001",
        adapter_capabilities_digest=config.adapter_capabilities_digest,
        readable_relative_paths=(visible.relative_path,),
        files=(visible,),
        replay_binding_digest="7" * 64,
    )


def test_materialization_accepts_benign_token_like_repository_paths(tmp_path) -> None:
    store, config, _manifest, plan, _trial = make_store(tmp_path)
    path = "samples/sk_service_configurator.py"
    body = b"def configure():\n    return None\n"
    visible = AgentVisibleFileBinding(
        role="repository_file",
        relative_path=path,
        size_bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )

    materialization = TrialMaterializationManifest.create(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        attempt=1,
        eval_input_digest=make_input().digest(),
        review_target_digest=make_input().review_target.digest(),
        wire_contract=config.wire_contract,
        suite_preparation_binding_digest=config.suite_preparation_binding_digest,
        prepared_source_id="prepared-repository-001",
        adapter_capabilities_digest=config.adapter_capabilities_digest,
        readable_relative_paths=(path,),
        files=(visible,),
        replay_binding_digest="7" * 64,
    )

    assert materialization.target_access.readable_relative_paths == (path,)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt == materialization.attempt
    receipt = store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        materialization,
        attempt=materialization.attempt,
    )
    assert receipt.target_access is not None
    assert receipt.target_access.readable_relative_paths == (path,)


def write_orphan_materialization(
    store: ArtifactStore,
    config: EvalRunConfig,
    plan,
    trial: TrialManifest,
    *,
    attempt: int,
) -> TrialMaterializationManifest:
    materialization = make_materialization(
        config,
        plan,
        attempt=attempt,
    )
    relative_path = (
        "cases/%s/trials/%s/materializations/attempt-%04d/"
        "materialization_manifest.json"
        % (trial.case_path_id, plan.trial_id, attempt)
    )
    target = store.root / config.run_id / Path(*relative_path.split("/"))
    store._ensure_directory(target.parent)
    store._write_json(
        config.run_id,
        relative_path,
        materialization,
    )
    return materialization


def completed_submission(
    trial_id: str,
    *,
    target_materialization_id: str = "materialization-unbound",
) -> EvalSubmission:
    return EvalSubmission.from_dict(
        {
            "schema_version": "eval_submission_v2",
            "task_id": TASK_ID,
            "agent_id": "agent-current",
            "trial_id": trial_id,
            "eval_input_digest": make_input().digest(),
            "target_materialization_id": target_materialization_id,
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


def failed_submission(
    trial_id: str,
    message: str,
    *,
    target_materialization_id: str = "materialization-unbound",
) -> EvalSubmission:
    return EvalSubmission.from_dict(
        {
            "schema_version": "eval_submission_v2",
            "task_id": TASK_ID,
            "agent_id": "agent-current",
            "trial_id": trial_id,
            "eval_input_digest": make_input().digest(),
            "target_materialization_id": target_materialization_id,
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


TERMINAL_SUBMISSION_STATUSES = (
    SubmissionStatus.COMPLETED,
    SubmissionStatus.FAILED,
    SubmissionStatus.BLOCKED,
    SubmissionStatus.INVALID_OUTPUT,
)


def terminal_submission(
    trial_id: str,
    status: SubmissionStatus,
    *,
    target_materialization_id: str,
    eval_input_digest: str | None = None,
    failure_code: FailureCode | None = None,
    include_trace: bool = False,
) -> EvalSubmission:
    if status is SubmissionStatus.COMPLETED:
        payload = completed_submission(
            trial_id,
            target_materialization_id=target_materialization_id,
        ).to_dict()
    else:
        payload = failed_submission(
            trial_id,
            "terminal failure",
            target_materialization_id=target_materialization_id,
        ).to_dict()
        default_code = {
            SubmissionStatus.FAILED: FailureCode.PROCESS_KILLED,
            SubmissionStatus.BLOCKED: FailureCode.AGENT_BLOCKED,
            SubmissionStatus.INVALID_OUTPUT: FailureCode.INVALID_JSON,
        }[status]
        payload["status"] = status.value
        payload["failure"]["code"] = (failure_code or default_code).value
    if eval_input_digest is not None:
        payload["eval_input_digest"] = eval_input_digest
    if include_trace:
        payload["trace_ref"] = {
            "type": "opaque_id",
            "value": "trace-terminal-binding",
        }
    return EvalSubmission.from_dict(payload)


def local_trace_submission(
    trial_id: str,
    *,
    target_materialization_id: str,
) -> EvalSubmission:
    payload = completed_submission(
        trial_id,
        target_materialization_id=target_materialization_id,
    ).to_dict()
    payload["trace_ref"] = {
        "type": "local_path",
        "value": "traces/attempt.jsonl",
    }
    return EvalSubmission.from_dict(payload)


def make_store(tmp_path: Path):
    store = ArtifactStore(tmp_path / ".eval-runs")
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot)
    manifest = store.create_run(config, snapshot)
    plan = manifest.trials[0]
    trial = store.load_trial_manifest(config.run_id, TASK_ID, plan.trial_id)
    return store, config, manifest, plan, trial


def required_runner_artifacts(submission: EvalSubmission):
    """Minimal control-plane artifacts required by a terminal Agent receipt."""

    artifacts = {
        "clarification_match_receipts.json": {"receipts": []},
        "terminal_summary.json": {"status": submission.status.value},
    }
    if (
        submission.trace_ref is not None
        and submission.trace_ref.type is TraceType.LOCAL_PATH
    ):
        artifacts["trace_capture.json"] = {"events": []}
    return artifacts


def write_required_runner_artifacts(
    store: ArtifactStore,
    config,
    plan,
    submission: EvalSubmission,
    *,
    attempt: int,
):
    base = "cases/%s/trials/%s/runner/attempt-%04d" % (
        plan.case_path_id,
        plan.trial_id,
        attempt,
    )
    refs = []
    for name, value in required_runner_artifacts(submission).items():
        refs.append(store._write_json(config.run_id, "%s/%s" % (base, name), value))
    return tuple(refs)


def write_orphan_submission_artifacts(
    store,
    config,
    plan,
    submission: EvalSubmission,
    *,
    attempt: int,
):
    base = "cases/%s/trials/%s" % (plan.case_path_id, plan.trial_id)
    store._write_json(
        config.run_id,
        base + "/submission.json",
        submission,
    )
    write_required_runner_artifacts(
        store,
        config,
        plan,
        submission,
        attempt=attempt,
    )
    return store._target(
        config.run_id,
        base + "/receipts/terminal.json",
    )


def trial_artifact_snapshot(store, config, plan):
    root = (
        store.root
        / config.run_id
        / "cases"
        / plan.case_path_id
        / "trials"
        / plan.trial_id
    )
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".locks" not in path.relative_to(root).parts
    }


def prepare_active_trial(store, config, plan):
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt is not None
    materialization = make_materialization(
        config,
        plan,
        attempt=running.active_attempt,
    )
    store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        materialization,
        attempt=running.active_attempt,
    )
    return running, materialization


def complete_trial(store, config, plan):
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt is not None
    materialization = make_materialization(
        config,
        plan,
        attempt=running.active_attempt,
    )
    store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        materialization,
        attempt=running.active_attempt,
    )
    submission = completed_submission(
        plan.trial_id,
        target_materialization_id=materialization.materialization_id,
    )
    state = store.finalize_submission(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        submission,
        attempt=running.active_attempt,
        runner_artifacts=required_runner_artifacts(submission),
    )
    return submission, state


def test_final_layout_uses_immutable_plan_manifests_and_hashed_case_path(tmp_path):
    store, config, manifest, plan, trial = make_store(tmp_path)
    run_dir = store.root / config.run_id
    trial_dir = (
        run_dir / "cases" / plan.case_path_id / "trials" / plan.trial_id
    )

    assert manifest.schema_version == "eval_run_manifest_v2"
    assert trial.schema_version == "eval_trial_manifest_v2"
    assert manifest.wire_contract == config.wire_contract
    assert manifest.suite_preparation_binding_digest is None
    assert manifest.adapter_capabilities_digest == (
        config.adapter_capabilities_digest
    )
    assert trial.wire_contract == config.wire_contract
    assert trial.target_kind is config.wire_contract.review_target_kind
    assert trial.materializer_protocol == config.materializer_protocol
    assert trial.suite_preparation_binding_digest is None
    assert trial.adapter_capabilities_digest == (
        config.adapter_capabilities_digest
    )
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


def test_v2_artifact_parents_reject_v1_wire_before_deeper_hydration(tmp_path):
    store, config, manifest, plan, trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt is not None
    materialization = make_materialization(
        config,
        plan,
        attempt=running.active_attempt,
    )

    for model, value in (
        (RunManifest, manifest),
        (TrialManifest, trial),
        (TrialMaterializationManifest, materialization),
    ):
        payload = value.to_dict()
        payload["wire_contract"]["case_schema_version"] = "eval_case_v1"
        payload["run_id"] = "malformed-but-deeper"
        with pytest.raises(UnsupportedProtocolVersionError):
            model.from_dict(payload)


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


def test_capability_preflight_v2_binds_wire_and_rejects_v1_root(tmp_path):
    store, config, manifest, _plan, _trial = make_store(tmp_path)
    ref = store.write_run_preflight(
        config.run_id,
        {"compatible": True, "checked_cases": [TASK_ID]},
    )
    path = store.root / config.run_id / Path(*ref.relative_path.split("/"))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "eval_capability_preflight_v2"
    assert payload["run_manifest_digest"] == manifest.digest()
    assert payload["wire_contract"] == config.wire_contract.to_dict()
    assert payload["adapter_capabilities_digest"] == (
        config.adapter_capabilities_digest
    )
    assert payload["target_kinds"] == ["repository"]
    assert payload["materializer_protocol"] == config.materializer_protocol
    assert store.load_run_preflight(config.run_id) == payload["preflight"]

    payload["schema_version"] = "eval_run_capability_preflight_v1"
    payload["legacy_unknown"] = True
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(UnsupportedProtocolVersionError):
        store.load_run_preflight(config.run_id)


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


def test_execution_budget_does_not_reduce_control_plane_capacity(tmp_path):
    store = ArtifactStore(tmp_path / ".eval-runs")
    snapshot = make_case_snapshot()
    large_execution_budget = ResourceBudgets(
        agent_timeout_seconds=900,
        evaluator_timeout_seconds=300,
        max_agent_output_bytes=1024,
        max_trace_bytes=1024,
        max_execution_artifact_file_bytes=16 * 1024 * 1024,
        max_execution_artifact_total_bytes=1 << 40,
        max_parallel_trials=1,
    )
    config = make_config(
        case_snapshot=snapshot,
        resource_budgets=large_execution_budget,
    )

    manifest = store.create_run(config, snapshot)
    assert manifest.run_config.relative_path == "run_config.json"
    assert store.load_run_config(config.run_id) == config


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
    materialization = make_materialization(
        config,
        plan,
        attempt=resumed.active_attempt,
    )
    prepare = store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        materialization,
        attempt=resumed.active_attempt,
    )
    assert prepare.stage is StageName.PREPARE
    assert prepare.attempt == resumed.active_attempt
    assert prepare.materialization_id == materialization.materialization_id
    assert prepare.eval_input_digest == make_input().digest()
    assert prepare.review_target_digest == make_input().review_target.digest()
    assert prepare.prepared_source_id == materialization.prepared_source_id
    assert prepare.agent_visible_files == materialization.files
    assert prepare.adapter_capabilities_digest == (
        config.adapter_capabilities_digest
    )
    assert prepare.target_access == materialization.target_access
    assert prepare.materialization_manifest is not None
    assert prepare.materialization_manifest_digest == (
        prepare.materialization_manifest.sha256
    )
    assert len(prepare.artifacts) == 2
    submission = completed_submission(
        plan.trial_id,
        target_materialization_id=materialization.materialization_id,
    )
    terminal = store.finalize_submission(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        submission,
        attempt=resumed.active_attempt,
        runner_artifacts=required_runner_artifacts(submission),
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

    with pytest.raises(ArtifactStateError, match="running Trial|active"):
        store.write_prepare_stage(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            make_input(),
            make_materialization(
                config,
                plan,
                attempt=first.active_attempt,
            ),
            attempt=first.active_attempt,
        )

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


@pytest.mark.parametrize(
    "status",
    TERMINAL_SUBMISSION_STATUSES,
    ids=lambda status: status.value,
)
def test_finalize_rejects_wrong_input_digest_without_terminal_mutation(
    tmp_path, status
):
    store, config, _, plan, _ = make_store(tmp_path)
    running, materialization = prepare_active_trial(store, config, plan)
    submission = terminal_submission(
        plan.trial_id,
        status,
        target_materialization_id=materialization.materialization_id,
        eval_input_digest="0" * 64,
        include_trace=True,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises((SchemaError, ArtifactIntegrityError, ArtifactStateError)):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(submission),
        )

    assert trial_artifact_snapshot(store, config, plan) == before


@pytest.mark.parametrize(
    "status",
    TERMINAL_SUBMISSION_STATUSES,
    ids=lambda status: status.value,
)
def test_finalize_rejects_wrong_materialization_without_terminal_mutation(
    tmp_path, status
):
    store, config, _, plan, _ = make_store(tmp_path)
    running, _materialization = prepare_active_trial(store, config, plan)
    submission = terminal_submission(
        plan.trial_id,
        status,
        target_materialization_id="materialization-wrong",
        include_trace=True,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises((SchemaError, ArtifactIntegrityError, ArtifactStateError)):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(submission),
        )

    assert trial_artifact_snapshot(store, config, plan) == before


@pytest.mark.parametrize(
    "missing_name",
    (
        "clarification_match_receipts.json",
        "terminal_summary.json",
    ),
)
def test_finalize_missing_control_artifact_rejects_before_any_publication(
    tmp_path, missing_name
):
    store, config, _, plan, _ = make_store(tmp_path)
    running, materialization = prepare_active_trial(store, config, plan)
    submission = completed_submission(
        plan.trial_id,
        target_materialization_id=materialization.materialization_id,
    )
    runner_artifacts = required_runner_artifacts(submission)
    runner_artifacts.pop(missing_name)
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="required Runner"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=runner_artifacts,
        )

    assert trial_artifact_snapshot(store, config, plan) == before


def test_finalize_does_not_adopt_unpassed_orphan_required_runner_artifact(
    tmp_path,
):
    store, config, _, plan, _ = make_store(tmp_path)
    running, materialization = prepare_active_trial(store, config, plan)
    submission = completed_submission(
        plan.trial_id,
        target_materialization_id=materialization.materialization_id,
    )
    artifacts = required_runner_artifacts(submission)
    orphan_name = "terminal_summary.json"
    orphan_value = artifacts.pop(orphan_name)
    runner_base = "cases/%s/trials/%s/runner/attempt-%04d" % (
        plan.case_path_id,
        plan.trial_id,
        running.active_attempt,
    )
    store._write_json(
        config.run_id,
        "%s/%s" % (runner_base, orphan_name),
        orphan_value,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="required Runner"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=artifacts,
        )

    assert trial_artifact_snapshot(store, config, plan) == before


def test_finalize_local_trace_requires_capture_before_any_publication(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    running, materialization = prepare_active_trial(store, config, plan)
    submission = local_trace_submission(
        plan.trial_id,
        target_materialization_id=materialization.materialization_id,
    )
    artifacts = required_runner_artifacts(submission)
    artifacts.pop("trace_capture.json")
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="required Runner"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=artifacts,
        )

    assert trial_artifact_snapshot(store, config, plan) == before


def test_finalize_rejects_malformed_mandatory_trace_without_mutation(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    running, materialization = prepare_active_trial(store, config, plan)
    submission = local_trace_submission(
        plan.trial_id,
        target_materialization_id=materialization.materialization_id,
    )
    artifacts = required_runner_artifacts(submission)
    artifacts["trace_capture.json"] = {"raw_reasoning": "hidden"}
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(SchemaError):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=artifacts,
        )

    assert trial_artifact_snapshot(store, config, plan) == before


def test_finalize_preflights_mandatory_trace_file_budget_without_mutation(
    tmp_path,
):
    constrained = ResourceBudgets(
        agent_timeout_seconds=900,
        evaluator_timeout_seconds=300,
        max_agent_output_bytes=64,
        max_trace_bytes=64,
        max_execution_artifact_file_bytes=64,
        max_execution_artifact_total_bytes=512,
        max_parallel_trials=1,
    )
    store = ArtifactStore(tmp_path / ".eval-runs")
    snapshot = make_case_snapshot()
    config = make_config(
        case_snapshot=snapshot,
        resource_budgets=constrained,
    )
    manifest = store.create_run(config, snapshot)
    plan = manifest.trials[0]
    running, materialization = prepare_active_trial(store, config, plan)
    submission = local_trace_submission(
        plan.trial_id,
        target_materialization_id=materialization.materialization_id,
    )
    artifacts = required_runner_artifacts(submission)
    artifacts["trace_capture.json"] = {"events": ["x" * 128]}
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="required trace"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=artifacts,
        )

    assert trial_artifact_snapshot(store, config, plan) == before


def test_finalize_preflights_trace_ref_create_only_conflict_without_mutation(
    tmp_path,
):
    store, config, _, plan, _ = make_store(tmp_path)
    running, materialization = prepare_active_trial(store, config, plan)
    submission = local_trace_submission(
        plan.trial_id,
        target_materialization_id=materialization.materialization_id,
    )
    base = "cases/%s/trials/%s" % (plan.case_path_id, plan.trial_id)
    store._write_json(
        config.run_id,
        "%s/trace_ref.json" % base,
        submission.trace_ref,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactConflictError, match="trace_ref"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(submission),
        )

    assert trial_artifact_snapshot(store, config, plan) == before


@pytest.mark.parametrize(
    "status",
    TERMINAL_SUBMISSION_STATUSES,
    ids=lambda status: status.value,
)
def test_finalize_without_prepare_rejects_ordinary_terminal_without_mutation(
    tmp_path, status
):
    store, config, _, plan, _ = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    submission = terminal_submission(
        plan.trial_id,
        status,
        target_materialization_id="materialization-unbound",
        include_trace=True,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises((SchemaError, ArtifactIntegrityError, ArtifactStateError)):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(submission),
        )

    assert trial_artifact_snapshot(store, config, plan) == before


def test_finalize_without_prepare_accepts_only_canonical_harness_binding(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    expected_binding = artifact_module.derive_pre_materialization_failure_binding(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        attempt=running.active_attempt,
        eval_input_digest=plan.eval_input_digest,
        review_target_digest=make_input().review_target.digest(),
    )
    submission = terminal_submission(
        plan.trial_id,
        SubmissionStatus.FAILED,
        target_materialization_id=expected_binding,
        failure_code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
    )

    state = store.finalize_submission(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        submission,
        attempt=running.active_attempt,
        runner_artifacts=required_runner_artifacts(submission),
    )

    assert state.status is TrialStatus.FAILED
    assert store.load_existing_submission(
        config.run_id, TASK_ID, plan.trial_id
    ) == submission


def test_pre_materialization_harness_failure_rejects_evidence_without_mutation(
    tmp_path,
):
    store, config, _, plan, _ = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    eval_input = make_input()
    expected_binding = artifact_module.derive_pre_materialization_failure_binding(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        attempt=running.active_attempt,
        eval_input_digest=plan.eval_input_digest,
        review_target_digest=eval_input.review_target.digest(),
    )
    payload = terminal_submission(
        plan.trial_id,
        SubmissionStatus.FAILED,
        target_materialization_id=expected_binding,
        failure_code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
    ).to_dict()
    excerpt = "Agent-supplied evidence"
    payload["evidence"] = [
        {
            "evidence_id": "evidence-forged-harness",
            "source": {
                "kind": "repository_file",
                "target_materialization_id": expected_binding,
                "revision": eval_input.review_target.repository.head_revision,
                "path": "app.py",
                "from_line": 1,
                "to_line": 1,
            },
            "content_hash": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
            "excerpt": excerpt,
        }
    ]
    submission = EvalSubmission.from_dict(payload)
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(SchemaError, match="requires a committed prepare receipt"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(submission),
        )

    assert trial_artifact_snapshot(store, config, plan) == before


def test_finalize_rejects_arbitrary_pre_materialization_binding_without_mutation(
    tmp_path,
):
    store, config, _, plan, _ = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    submission = terminal_submission(
        plan.trial_id,
        SubmissionStatus.FAILED,
        target_materialization_id="materialization-arbitrary",
        failure_code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
        include_trace=True,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises((SchemaError, ArtifactIntegrityError, ArtifactStateError)):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(submission),
        )

    assert trial_artifact_snapshot(store, config, plan) == before


def test_finalize_rejects_loose_materialization_without_prepare(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    write_orphan_materialization(
        store,
        config,
        plan,
        trial,
        attempt=running.active_attempt,
    )
    expected_binding = artifact_module.derive_pre_materialization_failure_binding(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        attempt=running.active_attempt,
        eval_input_digest=plan.eval_input_digest,
        review_target_digest=make_input().review_target.digest(),
    )
    submission = terminal_submission(
        plan.trial_id,
        SubmissionStatus.FAILED,
        target_materialization_id=expected_binding,
        failure_code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="without"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(submission),
        )

    assert trial_artifact_snapshot(store, config, plan) == before


def test_finalize_rejects_drifted_pre_prepare_input_without_mutation(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    drifted_payload = make_input().to_dict()
    drifted_payload["review_target"]["repository"]["head_revision"] = "3" * 40
    drifted_input = EvalInput.from_dict(drifted_payload)
    input_path = (
        "cases/%s/trials/%s/input.json" % (trial.case_path_id, plan.trial_id)
    )
    store._write_json(config.run_id, input_path, drifted_input)
    expected_binding = artifact_module.derive_pre_materialization_failure_binding(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        attempt=running.active_attempt,
        eval_input_digest=plan.eval_input_digest,
        review_target_digest=make_input().review_target.digest(),
    )
    submission = terminal_submission(
        plan.trial_id,
        SubmissionStatus.FAILED,
        target_materialization_id=expected_binding,
        failure_code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="EvalInput"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(submission),
        )

    assert trial_artifact_snapshot(store, config, plan) == before


@pytest.mark.parametrize(
    "drift",
    ("wire_target", "preparation", "capability", "target_content"),
)
def test_prepare_rejects_materialization_contract_drift(tmp_path, drift):
    store, config, _, plan, _trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt is not None
    baseline = make_materialization(
        config,
        plan,
        attempt=running.active_attempt,
    )
    wire_contract = baseline.wire_contract
    preparation_digest = baseline.suite_preparation_binding_digest
    capability_digest = baseline.adapter_capabilities_digest
    target_digest = baseline.review_target_digest
    if drift == "wire_target":
        wire_contract = WireContractV2.from_dict(
            {
                "case_schema_version": "eval_case_v2",
                "input_schema_version": "eval_input_v2",
                "submission_schema_version": "eval_submission_v2",
                "review_target_kind": "frozen_context",
                "materializer_protocol": "frozen-context-materializer-v2",
            }
        )
    elif drift == "preparation":
        preparation_digest = "0" * 64
    elif drift == "capability":
        capability_digest = "0" * 64
    else:
        target_digest = "0" * 64
    changed = TrialMaterializationManifest.create(
        run_id=baseline.run_id,
        task_id=baseline.task_id,
        trial_id=baseline.trial_id,
        attempt=baseline.attempt,
        eval_input_digest=baseline.eval_input_digest,
        review_target_digest=target_digest,
        wire_contract=wire_contract,
        suite_preparation_binding_digest=preparation_digest,
        prepared_source_id=baseline.prepared_source_id,
        adapter_capabilities_digest=capability_digest,
        readable_relative_paths=(
            baseline.target_access.readable_relative_paths
        ),
        files=baseline.files,
        replay_binding_digest=baseline.replay_binding_digest,
    )

    with pytest.raises(SchemaError, match="drift"):
        store.write_prepare_stage(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            make_input(),
            changed,
            attempt=running.active_attempt,
        )


@pytest.mark.parametrize("unbounded_field", ("readable_relative_paths", "files"))
def test_materialization_create_bounds_input_iterables(
    tmp_path, monkeypatch, unbounded_field
):
    _store, config, _manifest, plan, _trial = make_store(tmp_path)
    baseline = make_materialization(config, plan, attempt=1)
    monkeypatch.setattr(artifact_module, "MAX_AGENT_VISIBLE_FILES", 2)
    consumed = []

    def over_limit_values():
        for index in range(3):
            consumed.append(index)
            if unbounded_field == "readable_relative_paths":
                yield "target/repository/file-%d.py" % index
            else:
                body = ("file-%d" % index).encode("utf-8")
                yield AgentVisibleFileBinding(
                    role="repository_file",
                    relative_path="target/repository/file-%d.py" % index,
                    size_bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(),
                )
        raise AssertionError("materialization consumed beyond its item limit")

    values = {
        "readable_relative_paths": (
            over_limit_values()
            if unbounded_field == "readable_relative_paths"
            else baseline.target_access.readable_relative_paths
        ),
        "files": (
            over_limit_values()
            if unbounded_field == "files"
            else baseline.files
        ),
    }
    with pytest.raises(SchemaError, match="item limit"):
        TrialMaterializationManifest.create(
            run_id=baseline.run_id,
            task_id=baseline.task_id,
            trial_id=baseline.trial_id,
            attempt=baseline.attempt,
            eval_input_digest=baseline.eval_input_digest,
            review_target_digest=baseline.review_target_digest,
            wire_contract=baseline.wire_contract,
            suite_preparation_binding_digest=(
                baseline.suite_preparation_binding_digest
            ),
            prepared_source_id=baseline.prepared_source_id,
            adapter_capabilities_digest=(
                baseline.adapter_capabilities_digest
            ),
            readable_relative_paths=values["readable_relative_paths"],
            files=values["files"],
            replay_binding_digest=baseline.replay_binding_digest,
        )
    assert consumed == [0, 1, 2]


def test_materialization_create_size_gate_precedes_sort_and_identity(
    tmp_path, monkeypatch
):
    _store, config, _manifest, plan, _trial = make_store(tmp_path)
    baseline = make_materialization(config, plan, attempt=1)
    monkeypatch.setattr(
        artifact_module,
        "MAX_TRIAL_MATERIALIZATION_BYTES",
        len(canonical_json_bytes(baseline)) - 1,
    )

    def forbidden_identity(cls, **_kwargs):
        raise AssertionError("identity derivation ran before the size gate")

    monkeypatch.setattr(
        TrialMaterializationManifest,
        "derive_materialization_id",
        classmethod(forbidden_identity),
    )

    with pytest.raises(SchemaError, match="canonical byte limit"):
        TrialMaterializationManifest.create(
            run_id=baseline.run_id,
            task_id=baseline.task_id,
            trial_id=baseline.trial_id,
            attempt=baseline.attempt,
            eval_input_digest=baseline.eval_input_digest,
            review_target_digest=baseline.review_target_digest,
            wire_contract=baseline.wire_contract,
            suite_preparation_binding_digest=(
                baseline.suite_preparation_binding_digest
            ),
            prepared_source_id=baseline.prepared_source_id,
            adapter_capabilities_digest=(
                baseline.adapter_capabilities_digest
            ),
            readable_relative_paths=(
                baseline.target_access.readable_relative_paths
            ),
            files=baseline.files,
            replay_binding_digest=baseline.replay_binding_digest,
        )


def test_bounded_canonical_size_preflight_matches_exact_json_bytes():
    payload = {
        "unicode": ["café", "line\nfeed", "quote\"slash\\"],
        "scalars": {"bool": True, "float": 1.25, "none": None},
    }
    exact_size = len(canonical_json_bytes(payload))

    artifact_module._check_bounded_canonical_payload_size(
        payload,
        exact_size,
        "test payload",
    )
    with pytest.raises(SchemaError, match="canonical byte limit"):
        artifact_module._check_bounded_canonical_payload_size(
            payload,
            exact_size - 1,
            "test payload",
        )


def test_materialization_direct_size_gate_precedes_file_sort(
    tmp_path, monkeypatch
):
    _store, config, _manifest, plan, _trial = make_store(tmp_path)
    baseline = make_materialization(config, plan, attempt=1)
    monkeypatch.setattr(
        artifact_module,
        "MAX_TRIAL_MATERIALIZATION_BYTES",
        len(canonical_json_bytes(baseline)) - 1,
    )

    def forbidden_sort(*_args, **_kwargs):
        raise AssertionError("sorting ran before the size gate")

    monkeypatch.setattr(
        artifact_module,
        "sorted",
        forbidden_sort,
        raising=False,
    )

    with pytest.raises(SchemaError, match="canonical byte limit"):
        TrialMaterializationManifest(**vars(baseline))


def test_materialization_from_dict_size_gate_precedes_nested_hydration(
    tmp_path, monkeypatch
):
    _store, config, _manifest, plan, _trial = make_store(tmp_path)
    baseline = make_materialization(config, plan, attempt=1)
    payload = baseline.to_dict()
    monkeypatch.setattr(
        artifact_module,
        "MAX_TRIAL_MATERIALIZATION_BYTES",
        len(canonical_json_bytes(payload)) - 1,
    )

    def forbidden_target_access(cls, _value):
        raise AssertionError("nested hydration ran before the size gate")

    monkeypatch.setattr(
        artifact_module.TargetAccess,
        "from_dict",
        classmethod(forbidden_target_access),
    )

    with pytest.raises(SchemaError, match="canonical byte limit"):
        TrialMaterializationManifest.from_dict(payload)


def test_materialization_path_coverage_uses_indexed_component_walk():
    count = 20_000
    readable = tuple(
        "target/repository/roots/%05d" % index for index in range(count)
    )
    files = tuple("%s/file.py" % path for path in readable)

    artifact_module._validate_materialization_path_coverage(files, readable)


def test_materialization_path_coverage_rejects_unauthorized_file():
    with pytest.raises(SchemaError, match="outside TargetAccess"):
        artifact_module._validate_materialization_path_coverage(
            ("target/private/secret.py",),
            ("target/public",),
        )


def test_materialization_path_coverage_rejects_uncovered_readable_path():
    with pytest.raises(SchemaError, match="no Agent-visible file binding"):
        artifact_module._validate_materialization_path_coverage(
            ("target/public/app.py",),
            ("target/public", "target/empty"),
        )


@pytest.mark.parametrize("unbounded_field", ("artifacts", "agent_visible_files"))
def test_stage_receipt_create_bounds_input_iterables(
    monkeypatch, unbounded_field
):
    maximum_name = (
        "MAX_RECEIPT_ARTIFACTS"
        if unbounded_field == "artifacts"
        else "MAX_AGENT_VISIBLE_FILES"
    )
    monkeypatch.setattr(artifact_module, maximum_name, 2)
    consumed = []

    def over_limit_values():
        for index in range(3):
            consumed.append(index)
            body = ("receipt-%d" % index).encode("utf-8")
            if unbounded_field == "artifacts":
                yield ArtifactRef(
                    relative_path="receipt/file-%d.json" % index,
                    size_bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(),
                )
            else:
                yield AgentVisibleFileBinding(
                    role="repository_file",
                    relative_path="target/repository/file-%d.py" % index,
                    size_bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(),
                )
        raise AssertionError("StageReceipt consumed beyond its item limit")

    values = {
        "artifacts": (
            over_limit_values() if unbounded_field == "artifacts" else ()
        ),
        "agent_visible_files": (
            over_limit_values()
            if unbounded_field == "agent_visible_files"
            else ()
        ),
    }
    with pytest.raises(SchemaError, match="item limit"):
        StageReceipt.create(
            run_id="run-" + "0" * 64,
            task_id="task-001",
            trial_id="trial-" + "0" * 64,
            stage=StageName.START,
            config_digest="1" * 64,
            attempt=1,
            artifacts=values["artifacts"],
            agent_visible_files=values["agent_visible_files"],
        )
    assert consumed == [0, 1, 2]


def test_materialization_schema_and_persisted_content_drift_fail_closed(tmp_path):
    store, config, _, plan, _trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt is not None
    materialization = make_materialization(
        config,
        plan,
        attempt=running.active_attempt,
    )
    v1_payload = materialization.to_dict()
    v1_payload["schema_version"] = "eval_trial_materialization_v1"
    v1_payload["legacy_unknown"] = True
    with pytest.raises(UnsupportedProtocolVersionError):
        TrialMaterializationManifest.from_dict(v1_payload)

    prepare = store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        materialization,
        attempt=running.active_attempt,
    )
    assert prepare.materialization_manifest is not None
    path = (
        store.root
        / config.run_id
        / Path(*prepare.materialization_manifest.relative_path.split("/"))
    )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ArtifactIntegrityError, match="size|hash|canonical"):
        store.load_trial_state(config.run_id, TASK_ID, plan.trial_id)


def test_v2_prepare_receipt_rejects_v1_materialization_before_hydration(tmp_path):
    store, config, _, plan, trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt is not None
    materialization = make_materialization(
        config,
        plan,
        attempt=running.active_attempt,
    )
    prepare = store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        materialization,
        attempt=running.active_attempt,
    )
    assert prepare.materialization_manifest is not None
    materialization_path = (
        store.root
        / config.run_id
        / Path(*prepare.materialization_manifest.relative_path.split("/"))
    )
    materialization_payload = json.loads(
        materialization_path.read_text(encoding="utf-8")
    )
    materialization_payload["schema_version"] = (
        "eval_trial_materialization_v1"
    )
    materialization_payload["legacy_unknown"] = True
    materialization_bytes = canonical_json_bytes(materialization_payload)
    materialization_path.write_bytes(materialization_bytes)
    changed_hash = hashlib.sha256(materialization_bytes).hexdigest()

    receipt_path = (
        store.root
        / config.run_id
        / "cases"
        / trial.case_path_id
        / "trials"
        / plan.trial_id
        / "receipts"
        / ("attempt-%04d" % running.active_attempt)
        / "prepare.json"
    )
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_payload["materialization_manifest"]["sha256"] = changed_hash
    receipt_payload["materialization_manifest"]["size_bytes"] = len(
        materialization_bytes
    )
    receipt_payload["materialization_manifest_digest"] = changed_hash
    for artifact in receipt_payload["artifacts"]:
        if artifact["relative_path"] == prepare.materialization_manifest.relative_path:
            artifact["sha256"] = changed_hash
            artifact["size_bytes"] = len(materialization_bytes)
    receipt_path.write_bytes(canonical_json_bytes(receipt_payload))

    with pytest.raises(UnsupportedProtocolVersionError):
        store.load_trial_state(config.run_id, TASK_ID, plan.trial_id)


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
    materialization = make_materialization(
        config,
        plan,
        attempt=running.active_attempt,
    )
    store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        materialization,
        attempt=running.active_attempt,
    )
    base = "cases/%s/trials/%s" % (trial.case_path_id, plan.trial_id)
    failed = failed_submission(
        plan.trial_id,
        "process stopped",
        target_materialization_id=materialization.materialization_id,
    )
    submission_ref = store._write_json(
        config.run_id, "%s/submission.json" % base, failed
    )
    runner_refs = write_required_runner_artifacts(
        store,
        config,
        plan,
        failed,
        attempt=1,
    )
    mismatched = StageReceipt.create(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        stage=StageName.AGENT,
        config_digest=trial.agent_config_digest,
        artifacts=(submission_ref,) + runner_refs,
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
    first_materialization = write_orphan_materialization(
        store,
        config,
        plan,
        trial,
        attempt=running.active_attempt,
    )
    first_plan = store.recover_trial(config.run_id, TASK_ID, plan.trial_id)
    assert first_plan.status is TrialStatus.INCOMPLETE
    assert first_plan.completed_stages == (StageName.PREPARE,)
    assert first_plan.missing_stages == (StageName.AGENT,)
    assert input_path.read_bytes() == input_bytes

    retry = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert retry.active_attempt == 2
    second_materialization = write_orphan_materialization(
        store,
        config,
        plan,
        trial,
        attempt=retry.active_attempt,
    )
    assert first_materialization.materialization_id != (
        second_materialization.materialization_id
    )
    submission = completed_submission(
        plan.trial_id,
        target_materialization_id=second_materialization.materialization_id,
    )
    submission_ref = store._write_json(
        config.run_id, "%s/submission.json" % base, submission
    )
    write_required_runner_artifacts(
        store,
        config,
        plan,
        submission,
        attempt=2,
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


def test_recovery_checks_cumulative_execution_budget_under_lock_before_mutation(
    tmp_path, monkeypatch
):
    constrained = ResourceBudgets(
        agent_timeout_seconds=900,
        evaluator_timeout_seconds=300,
        max_agent_output_bytes=64,
        max_trace_bytes=64,
        max_execution_artifact_file_bytes=256,
        max_execution_artifact_total_bytes=300,
        max_parallel_trials=1,
    )
    store = ArtifactStore(tmp_path / ".eval-runs")
    snapshot = make_case_snapshot()
    config = make_config(
        case_snapshot=snapshot,
        resource_budgets=constrained,
    )
    manifest = store.create_run(config, snapshot)
    plan = manifest.trials[0]
    running, materialization = prepare_active_trial(store, config, plan)
    submission = completed_submission(
        plan.trial_id,
        target_materialization_id=materialization.materialization_id,
    )
    terminal_path = write_orphan_submission_artifacts(
        store,
        config,
        plan,
        submission,
        attempt=running.active_attempt,
    )
    runner_base = "cases/%s/trials/%s/runner/attempt-%04d" % (
        plan.case_path_id,
        plan.trial_id,
        running.active_attempt,
    )
    for name in ("workspace_manifest.json", "command_attestations.json"):
        ref = store._write_json(
            config.run_id,
            "%s/%s" % (runner_base, name),
            {"payload": "x" * 180},
        )
        assert ref.size_bytes <= constrained.max_execution_artifact_file_bytes
    assert store._execution_artifact_total_bytes(config.run_id) > (
        constrained.max_execution_artifact_total_bytes
    )

    real_lock = store._lock
    real_total = store._execution_artifact_total_bytes
    budget_lock_held = False
    total_checked_under_lock = False

    @contextmanager
    def tracked_lock(path):
        nonlocal budget_lock_held
        with real_lock(path):
            is_budget_lock = path.name == "execution-budget.lock"
            if is_budget_lock:
                budget_lock_held = True
            try:
                yield
            finally:
                if is_budget_lock:
                    budget_lock_held = False

    def checked_total(run_id):
        nonlocal total_checked_under_lock
        assert budget_lock_held
        total_checked_under_lock = True
        return real_total(run_id)

    monkeypatch.setattr(store, "_lock", tracked_lock)
    monkeypatch.setattr(store, "_execution_artifact_total_bytes", checked_total)
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="cumulative"):
        store.recover_trial(config.run_id, TASK_ID, plan.trial_id)

    assert total_checked_under_lock is True
    assert not terminal_path.exists()
    assert trial_artifact_snapshot(store, config, plan) == before


@pytest.mark.parametrize("orphan_kind", ("input", "materialization", "submission"))
def test_recovery_rejects_v1_orphans_before_committing_incomplete(
    tmp_path, orphan_kind
):
    store, config, _manifest, plan, trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt == 1
    base = "cases/%s/trials/%s" % (trial.case_path_id, plan.trial_id)

    if orphan_kind == "input":
        payload = make_input().to_dict()
        payload["schema_version"] = "eval_input_v1"
        payload["legacy_unknown"] = True
        relative_path = base + "/input.json"
    elif orphan_kind == "materialization":
        materialization = make_materialization(
            config,
            plan,
            attempt=running.active_attempt,
        )
        payload = materialization.to_dict()
        payload["schema_version"] = "eval_trial_materialization_v1"
        payload["legacy_unknown"] = True
        relative_path = (
            base
            + "/materializations/attempt-%04d/materialization_manifest.json"
            % running.active_attempt
        )
        store._ensure_directory(
            store._target(config.run_id, relative_path).parent
        )
    else:
        payload = failed_submission(plan.trial_id, "orphan").to_dict()
        payload["schema_version"] = "eval_submission_v1"
        payload["legacy_unknown"] = True
        relative_path = base + "/submission.json"

    store._write_json(config.run_id, relative_path, payload)
    incomplete_path = store._target(
        config.run_id,
        base + "/receipts/attempt-%04d/incomplete.json" % running.active_attempt,
    )

    with pytest.raises(UnsupportedProtocolVersionError):
        store.recover_trial(config.run_id, TASK_ID, plan.trial_id)
    assert not incomplete_path.exists()


@pytest.mark.parametrize(
    "status",
    TERMINAL_SUBMISSION_STATUSES,
    ids=lambda status: status.value,
)
def test_recovery_revalidates_submission_input_digest_before_commit(
    tmp_path, status
):
    store, config, _manifest, plan, _trial = make_store(tmp_path)
    running, materialization = prepare_active_trial(store, config, plan)
    submission = terminal_submission(
        plan.trial_id,
        status,
        target_materialization_id=materialization.materialization_id,
        eval_input_digest="0" * 64,
    )
    terminal_path = write_orphan_submission_artifacts(
        store,
        config,
        plan,
        submission,
        attempt=running.active_attempt,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="EvalInput digest"):
        store.recover_trial(config.run_id, TASK_ID, plan.trial_id)

    assert not terminal_path.exists()
    assert trial_artifact_snapshot(store, config, plan) == before


@pytest.mark.parametrize(
    "status",
    TERMINAL_SUBMISSION_STATUSES,
    ids=lambda status: status.value,
)
def test_recovery_rejects_wrong_materialization_before_terminal_commit(
    tmp_path, status
):
    store, config, _manifest, plan, _trial = make_store(tmp_path)
    running, _materialization = prepare_active_trial(store, config, plan)
    submission = terminal_submission(
        plan.trial_id,
        status,
        target_materialization_id="materialization-wrong",
    )
    terminal_path = write_orphan_submission_artifacts(
        store,
        config,
        plan,
        submission,
        attempt=running.active_attempt,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="materialization"):
        store.recover_trial(config.run_id, TASK_ID, plan.trial_id)

    assert not terminal_path.exists()
    assert trial_artifact_snapshot(store, config, plan) == before


@pytest.mark.parametrize(
    "status",
    TERMINAL_SUBMISSION_STATUSES,
    ids=lambda status: status.value,
)
def test_recovery_without_prepare_rejects_ordinary_terminal_without_mutation(
    tmp_path, status
):
    store, config, _manifest, plan, _trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    submission = terminal_submission(
        plan.trial_id,
        status,
        target_materialization_id="materialization-unbound",
    )
    terminal_path = write_orphan_submission_artifacts(
        store,
        config,
        plan,
        submission,
        attempt=running.active_attempt,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="prepare"):
        store.recover_trial(config.run_id, TASK_ID, plan.trial_id)

    assert not terminal_path.exists()
    assert trial_artifact_snapshot(store, config, plan) == before


def test_recovery_without_prepare_requires_canonical_harness_binding(tmp_path):
    store, config, _manifest, plan, _trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    arbitrary = terminal_submission(
        plan.trial_id,
        SubmissionStatus.FAILED,
        target_materialization_id="materialization-arbitrary",
        failure_code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
    )
    terminal_path = write_orphan_submission_artifacts(
        store,
        config,
        plan,
        arbitrary,
        attempt=running.active_attempt,
    )
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(ArtifactIntegrityError, match="pre-materialization"):
        store.recover_trial(config.run_id, TASK_ID, plan.trial_id)

    assert not terminal_path.exists()
    assert trial_artifact_snapshot(store, config, plan) == before


def test_recovery_adopts_canonical_pre_materialization_harness_failure(tmp_path):
    store, config, _manifest, plan, _trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    expected_binding = artifact_module.derive_pre_materialization_failure_binding(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        attempt=running.active_attempt,
        eval_input_digest=plan.eval_input_digest,
        review_target_digest=make_input().review_target.digest(),
    )
    submission = terminal_submission(
        plan.trial_id,
        SubmissionStatus.FAILED,
        target_materialization_id=expected_binding,
        failure_code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
    )
    terminal_path = write_orphan_submission_artifacts(
        store,
        config,
        plan,
        submission,
        attempt=running.active_attempt,
    )

    recovery = store.recover_trial(config.run_id, TASK_ID, plan.trial_id)

    assert recovery.status is TrialStatus.FAILED
    assert recovery.terminal is True
    assert terminal_path.is_file()
    assert store.load_existing_submission(
        config.run_id, TASK_ID, plan.trial_id
    ) == submission


def test_abandonment_without_prepare_uses_canonical_harness_failure(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    recovery = store.recover_trial(config.run_id, TASK_ID, plan.trial_id)
    assert recovery.status is TrialStatus.INCOMPLETE
    expected_binding = artifact_module.derive_pre_materialization_failure_binding(
        run_id=config.run_id,
        task_id=TASK_ID,
        trial_id=plan.trial_id,
        attempt=running.active_attempt,
        eval_input_digest=plan.eval_input_digest,
        review_target_digest=make_input().review_target.digest(),
    )

    state = store.abandon_trial(config.run_id, TASK_ID, plan.trial_id)
    submission = store.load_existing_submission(config.run_id, TASK_ID, plan.trial_id)
    assert state.status is TrialStatus.FAILED
    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.HARNESS_MATERIALIZATION_ERROR
    assert submission.target_materialization_id == expected_binding
    assert state.terminal_receipt is not None
    assert state.terminal_receipt.failure_code is (
        FailureCode.HARNESS_MATERIALIZATION_ERROR
    )


def test_abandonment_after_prepare_uses_committed_materialization_binding(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    _running, materialization = prepare_active_trial(store, config, plan)

    state = store.abandon_trial(config.run_id, TASK_ID, plan.trial_id)
    submission = store.load_existing_submission(
        config.run_id, TASK_ID, plan.trial_id
    )

    assert state.status is TrialStatus.FAILED
    assert submission.target_materialization_id == materialization.materialization_id
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.PROCESS_KILLED


def test_abandonment_allows_harness_failure_after_prepare_with_bound_identity(
    tmp_path,
):
    store, config, _, plan, _ = make_store(tmp_path)
    _running, materialization = prepare_active_trial(store, config, plan)

    state = store.abandon_trial(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        failure_code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
    )
    submission = store.load_existing_submission(
        config.run_id,
        TASK_ID,
        plan.trial_id,
    )

    assert state.status is TrialStatus.FAILED
    assert submission.target_materialization_id == materialization.materialization_id
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.HARNESS_MATERIALIZATION_ERROR


def test_harness_materialization_failure_rejects_agent_owned_metadata(
    tmp_path,
):
    store, config, _, plan, _ = make_store(tmp_path)
    running, materialization = prepare_active_trial(store, config, plan)
    payload = terminal_submission(
        plan.trial_id,
        SubmissionStatus.FAILED,
        target_materialization_id=materialization.materialization_id,
        failure_code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
        include_trace=True,
    ).to_dict()
    payload["failure"]["retryable"] = True
    payload["usage"].update(
        {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
            "tool_calls": 1,
            "cost_amount": 1,
            "cost_currency": "USD",
        }
    )
    submission = EvalSubmission.from_dict(payload)
    before = trial_artifact_snapshot(store, config, plan)

    with pytest.raises(SchemaError, match="Agent-owned metadata"):
        store.finalize_submission(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            submission,
            attempt=running.active_attempt,
            runner_artifacts=required_runner_artifacts(submission),
        )

    assert trial_artifact_snapshot(store, config, plan) == before


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


def test_evaluation_typed_context_artifacts_round_trip_env_like_code(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    complete_trial(store, config, plan)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(), config.resource_budgets
    )
    content = "count = len(items)\naffinity = score(item)"
    review_payload = _review_context_payload(content)
    input_payload = _judge_input_context_payload(content)
    output_payload = _model_turn_context_block_payload(content)

    receipt = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=execution,
        revision="typed-context-round-trip",
        intent_matches={},
        review_matches=review_payload,
        judge_input=input_payload,
        judge_output=output_payload,
        score={"total": 1},
    )
    assert receipt.evaluation_id is not None

    bundle = store.load_evaluation_bundle(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        receipt.evaluation_id,
    )
    assert bundle.intent_matches == {}
    assert bundle.review_matches == review_payload
    assert bundle.judge_input == input_payload
    assert bundle.judge_output == output_payload


def _safe_evaluation_context_payloads() -> dict:
    return {
        "review_matches": _review_context_payload("ordinary context"),
        "judge_input": _judge_input_context_payload("ordinary context"),
        "judge_output": _model_turn_context_block_payload("ordinary context"),
    }


@pytest.mark.parametrize(
    "content",
    [
        "count = len(items)\napi_key=ordinary-secret",
        "<think>private intermediate steps</think>",
    ],
)
def test_evaluation_judge_output_context_still_rejects_sensitive_content(
    tmp_path,
    content,
):
    store, config, _, plan, _ = make_store(tmp_path)
    complete_trial(store, config, plan)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(), config.resource_budgets
    )
    payload = _model_turn_context_block_payload(content)
    safe_payloads = _safe_evaluation_context_payloads()

    with pytest.raises(SchemaError):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=execution,
            revision="typed-context-sensitive",
            intent_matches={},
            review_matches=safe_payloads["review_matches"],
            judge_input=safe_payloads["judge_input"],
            judge_output=payload,
            score={},
        )


def test_evaluation_score_does_not_enable_typed_context_exception(tmp_path):
    store, config, _, plan, _ = make_store(tmp_path)
    complete_trial(store, config, plan)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(), config.resource_budgets
    )
    score = _model_turn_context_block_payload(
        "count = len(items)\naffinity = score(item)"
    )
    safe_payloads = _safe_evaluation_context_payloads()

    with pytest.raises(SchemaError, match="full environment dump"):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=execution,
            revision="typed-context-rejected-score",
            intent_matches={},
            review_matches=safe_payloads["review_matches"],
            judge_input=safe_payloads["judge_input"],
            judge_output=safe_payloads["judge_output"],
            score=score,
        )


@pytest.mark.parametrize("artifact_name", ["intent_matches", "judge_output"])
def test_evaluation_context_policy_rejects_untyped_or_wrong_schema_payloads(
    tmp_path,
    artifact_name,
):
    store, config, _, plan, _ = make_store(tmp_path)
    complete_trial(store, config, plan)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(), config.resource_budgets
    )
    payload = _model_turn_context_block_payload(
        "count = len(items)\naffinity = score(item)"
    )
    if artifact_name == "intent_matches":
        payload = {"not_a_judge_schema": payload}
    else:
        payload["schema_version"] = "unknown"
    safe_payloads = _safe_evaluation_context_payloads()
    values = {
        "intent_matches": {},
        "review_matches": safe_payloads["review_matches"],
        "judge_input": safe_payloads["judge_input"],
        "judge_output": safe_payloads["judge_output"],
        "score": {},
    }
    values[artifact_name] = payload

    with pytest.raises(SchemaError, match="full environment dump"):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=execution,
            revision="typed-context-policy-rejected-" + artifact_name,
            **values,
        )


def test_evaluation_context_policy_rejects_typed_schema_with_wrong_root_fields(
    tmp_path,
):
    store, config, _, plan, _ = make_store(tmp_path)
    complete_trial(store, config, plan)
    execution = EvaluatorExecutionConfig.from_resource_budgets(
        evaluator_config(), config.resource_budgets
    )
    safe_payloads = _safe_evaluation_context_payloads()

    with pytest.raises(SchemaError):
        store.write_evaluation(
            config.run_id,
            TASK_ID,
            plan.trial_id,
            evaluator_execution=execution,
            revision="typed-context-invalid-root",
            intent_matches={},
            review_matches=safe_payloads["review_matches"],
            judge_input={
                "schema_version": "eval_judge_input_artifact_v1",
                "claims": [],
            },
            judge_output=safe_payloads["judge_output"],
            score={},
        )


def test_evaluation_total_budget_is_preflighted_before_artifacts_publish(tmp_path):
    constrained = ResourceBudgets(
        agent_timeout_seconds=900,
        evaluator_timeout_seconds=300,
        max_agent_output_bytes=256,
        max_trace_bytes=256,
        # The v2 evaluator execution config is larger than the old
        # scalar-Judge fixture. Leave enough per-file room for it while
        # keeping the cumulative budget intentionally too small.
        max_execution_artifact_file_bytes=5_200,
        max_execution_artifact_total_bytes=5_200,
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

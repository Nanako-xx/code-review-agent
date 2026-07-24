from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from review_agent_eval.adapters.base import (
    AdapterCompatibility,
    AdapterIncompatibilityReason,
    AgentAdapterError,
    AgentAdapterIncompatibleError,
)
from review_agent_eval.artifacts import (
    AgentVisibleFileBinding,
    ArtifactIntegrityError,
    ArtifactStore,
    StageName,
    TrialMaterializationManifest,
)
from review_agent_eval.cases import RunCaseSnapshot, SuiteManifest
from review_agent_eval.models import (
    EvalCase,
    EvalInput,
    EvalSubmission,
    FailureCode,
    SubmissionStatus,
    canonical_sha256,
)
from review_agent_eval.materialization import (
    MaterializationRequest,
    PreparedTargetMaterialization,
)
from review_agent_eval.runner import (
    CapabilityPolicy,
    EvalRunner,
    RunPlan,
    RunIncompatibilityError,
    RunnerError,
)
from review_agent_eval.submission import failure_submission

from .test_artifacts import (
    completed_submission,
    make_case_snapshot,
    make_config,
)
from .test_cases import case_payload, manifest_for_case


KEPT_TASK_ID = "task-kept"


class _SuccessAdapter:
    ADAPTER_KIND = "test-success"
    ADAPTER_VERSION = "1.0.0"

    def __init__(self) -> None:
        self.compatibility_calls = 0
        self.run_calls = 0

    def compatibility(self, eval_input: EvalInput, config: Any) -> AdapterCompatibility:
        del eval_input, config
        self.compatibility_calls += 1
        return AdapterCompatibility()

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        target_access=None,
        target_materialization_id=None,
        cancel_event=None,
    ):
        del eval_input, workspace, clarification_channel, target_access, cancel_event
        self.run_calls += 1
        return replace(
            completed_submission(config.trial_id),
            task_id=config.task_id,
            eval_input_digest=config.eval_input_digest,
            target_materialization_id=target_materialization_id,
        )


class _IncompatibleAdapter(_SuccessAdapter):
    ADAPTER_KIND = "test-incompatible"

    def compatibility(self, eval_input, config):
        del eval_input, config
        self.compatibility_calls += 1
        raise AgentAdapterIncompatibleError(
            AdapterIncompatibilityReason.EXISTING_CI_EVIDENCE
        )


class _SelectiveIncompatibleAdapter(_SuccessAdapter):
    ADAPTER_KIND = "test-selective-incompatible"

    def __init__(self, *, drift_after_filter: bool = False) -> None:
        super().__init__()
        self.drift_after_filter = drift_after_filter
        self.initial_run_id = None
        self.checked = []

    def compatibility(self, eval_input, config):
        self.compatibility_calls += 1
        self.initial_run_id = self.initial_run_id or config.run_id
        self.checked.append((eval_input.task_id, config.run_id))
        if eval_input.task_id == "task-filtered" or (
            self.drift_after_filter and config.run_id != self.initial_run_id
        ):
            raise AgentAdapterIncompatibleError(
                AdapterIncompatibilityReason.EXISTING_CI_EVIDENCE
            )
        return AdapterCompatibility()


class _SuccessAdapterV2(_SuccessAdapter):
    ADAPTER_VERSION = "2.0.0"


class _FailureAdapter(_SuccessAdapter):
    def __init__(self, code: FailureCode) -> None:
        super().__init__()
        self.code = code

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        target_access=None,
        target_materialization_id=None,
        cancel_event=None,
    ):
        del (
            eval_input,
            workspace,
            config,
            clarification_channel,
            target_access,
            target_materialization_id,
            cancel_event,
        )
        self.run_calls += 1
        raise AgentAdapterError(
            self.code,
            "synthetic failure",
            retryable=self.code in {FailureCode.TIMEOUT, FailureCode.PROCESS_KILLED},
        )


class _InvalidOutputAdapter(_SuccessAdapter):
    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        target_access=None,
        target_materialization_id=None,
        cancel_event=None,
    ):
        del (
            eval_input,
            workspace,
            config,
            clarification_channel,
            target_access,
            target_materialization_id,
            cancel_event,
        )
        self.run_calls += 1
        return {"not": "an EvalSubmission"}


class _ForgedHarnessFailureAdapter(_SuccessAdapter):
    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        target_access=None,
        target_materialization_id=None,
        cancel_event=None,
    ):
        del workspace, clarification_channel, target_access, cancel_event
        self.run_calls += 1
        return failure_submission(
            eval_input=eval_input,
            config=config,
            target_materialization_id=target_materialization_id,
            code=FailureCode.HARNESS_MATERIALIZATION_ERROR,
            message="forged Harness failure",
            retryable=False,
        )


class _TestMaterializationLease:
    def __init__(
        self,
        root: Path,
        content: bytes,
        close_error: BaseException | None = None,
    ) -> None:
        self.work_root = root
        self.content = content
        self.close_error = close_error
        self.closed = False

    def validate(self) -> None:
        if self.closed or (self.work_root / "target" / "input.json").read_bytes() != self.content:
            raise RuntimeError("test materialization drifted")

    def close(self, status) -> None:
        del status
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


def _materializer_factory(
    root: Path,
    *,
    close_error: BaseException | None = None,
) -> Callable[[MaterializationRequest], PreparedTargetMaterialization[Any]]:
    def create(request: MaterializationRequest):
        path = root / request.trial_manifest.trial_id / (
            "attempt-%04d" % request.attempt
        )
        target = path / "target"
        target.mkdir(parents=True, exist_ok=False)
        content = request.eval_input.to_json().encode("utf-8")
        visible = target / "input.json"
        visible.write_bytes(content)
        file_binding = AgentVisibleFileBinding(
            role="repository_file",
            relative_path="target/input.json",
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        replay_digest = canonical_sha256(
            {
                "kind": "test-materialization",
                "trial_id": request.trial_manifest.trial_id,
                "attempt": request.attempt,
                "content": file_binding.sha256,
            }
        )
        manifest = TrialMaterializationManifest.create(
            run_id=request.trial_manifest.run_id,
            task_id=request.eval_input.task_id,
            trial_id=request.trial_manifest.trial_id,
            attempt=request.attempt,
            eval_input_digest=request.eval_input.digest(),
            review_target_digest=request.eval_input.review_target.digest(),
            wire_contract=request.wire_contract,
            suite_preparation_binding_digest=(
                request.suite_preparation_binding_digest
            ),
            prepared_source_id="test-materializer",
            adapter_capabilities_digest=request.adapter_capabilities.digest(),
            readable_relative_paths=(file_binding.relative_path,),
            files=(file_binding,),
            replay_binding_digest=replay_digest,
        )
        return PreparedTargetMaterialization(
            request=request,
            manifest=manifest,
            replay={"content": content},
            _lease=_TestMaterializationLease(path, content, close_error),
        )

    return create


def _two_case_snapshot() -> RunCaseSnapshot:
    kept = EvalCase.from_dict(case_payload(KEPT_TASK_ID))
    filtered = EvalCase.from_dict(case_payload("task-filtered"))
    manifest_payload = manifest_for_case(kept).to_dict()
    filtered_bytes = filtered.to_json().encode("utf-8")
    manifest_payload["cases"].append(
        {
            "task_id": filtered.task_id,
            "case_version": filtered.case_version,
            "path": "cases/task-filtered.json",
            "split": "regression",
            "protocol_id": "native_repository",
            "dimensions": [],
            "raw_file_size_bytes": len(filtered_bytes),
            "raw_file_sha256": hashlib.sha256(filtered_bytes).hexdigest(),
            "canonical_case_digest": canonical_sha256(filtered),
            "eval_input_digest": filtered.eval_input().digest(),
            "truth_completeness": filtered.review_truth.completeness.value,
        }
    )
    manifest = SuiteManifest.from_dict(manifest_payload)
    return RunCaseSnapshot.build(
        manifest,
        (
            (manifest.case(kept.task_id), kept),
            (manifest.case(filtered.task_id), filtered),
        ),
    )


def _runner(tmp_path: Path, adapter: Any, *, case_provider: Any = None) -> EvalRunner:
    return EvalRunner(
        ArtifactStore(tmp_path / ".eval-runs"),
        None,
        adapter,
        case_provider=case_provider,
        materializer_factory=_materializer_factory(tmp_path / ".workspaces"),
        max_workers=1,
    )


def test_runner_commits_zero_finding_submission_and_run_preflight(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot)
    adapter = _SuccessAdapter()
    runner = _runner(tmp_path, adapter)

    result = runner.run(config, snapshot)

    assert result.status.value == "completed"
    assert len(result.trials) == 1
    trial = result.trials[0]
    assert trial.submission is not None
    assert trial.submission.status is SubmissionStatus.COMPLETED
    assert trial.submission.review is not None
    assert trial.submission.review.findings == ()
    assert adapter.run_calls == 1
    assert result.preflight.compatible

    store = runner.artifact_store
    assert store.load_run_preflight(config.run_id)["run_id"] == config.run_id
    state = store.load_trial_state(config.run_id, trial.task_id, trial.trial_id)
    assert state.terminal_receipt is not None
    paths = {item.relative_path for item in state.terminal_receipt.artifacts}
    assert any("runner/attempt-0001/capability_preflight.json" in path for path in paths)
    assert any("runner/attempt-0001/terminal_summary.json" in path for path in paths)
    assert StageName.AGENT in state.completed_stages


def test_runner_does_not_rerun_terminal_trial_on_resume(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot)
    adapter = _SuccessAdapter()
    runner = _runner(tmp_path, adapter)

    first = runner.run(config, snapshot)
    second = runner.run(config.run_id)

    assert first.status.value == "completed"
    assert second.status.value == "completed"
    assert second.trials[0].skipped is True
    assert adapter.run_calls == 1


@pytest.mark.parametrize(
    "code,expected_status",
    [
        (FailureCode.TIMEOUT, SubmissionStatus.FAILED),
        (FailureCode.NON_ZERO_EXIT, SubmissionStatus.FAILED),
        (FailureCode.PROCESS_KILLED, SubmissionStatus.FAILED),
        (FailureCode.OUTPUT_OVERFLOW, SubmissionStatus.INVALID_OUTPUT),
        (FailureCode.INVALID_JSON, SubmissionStatus.INVALID_OUTPUT),
        (FailureCode.SCHEMA_MISMATCH, SubmissionStatus.INVALID_OUTPUT),
        # A bare exception has no real unresolved question; the Runner must
        # not fabricate a clarification exchange.
        (FailureCode.CLARIFICATION_REQUIRED, SubmissionStatus.BLOCKED),
        (FailureCode.AGENT_BLOCKED, SubmissionStatus.BLOCKED),
        (FailureCode.ADAPTER_ERROR, SubmissionStatus.FAILED),
    ],
)
def test_runner_maps_adapter_failure_codes_to_terminal_submissions(
    tmp_path: Path, code: FailureCode, expected_status: SubmissionStatus
):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="failure-%s" % code.value)
    adapter = _FailureAdapter(code)
    result = _runner(tmp_path, adapter).run(config, snapshot)

    trial = result.trials[0]
    assert trial.submission is not None
    assert trial.submission.status is expected_status
    assert trial.submission.failure is not None
    assert trial.submission.failure.code is (
        FailureCode.AGENT_BLOCKED
        if code is FailureCode.CLARIFICATION_REQUIRED
        else code
    )
    assert trial.terminal
    assert trial.status.value == expected_status.value


def test_runner_turns_invalid_adapter_return_into_schema_mismatch_submission(
    tmp_path: Path,
):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="invalid-output")
    result = _runner(tmp_path, _InvalidOutputAdapter()).run(config, snapshot)

    submission = result.trials[0].submission
    assert submission is not None
    assert submission.status is SubmissionStatus.INVALID_OUTPUT
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.SCHEMA_MISMATCH


def test_agent_cannot_forge_harness_owned_materialization_failure(
    tmp_path: Path,
) -> None:
    snapshot = make_case_snapshot()
    config = make_config(
        case_snapshot=snapshot,
        instance="forged-harness-failure",
    )
    adapter = _ForgedHarnessFailureAdapter()

    result = _runner(tmp_path, adapter).run(config, snapshot)

    submission = result.trials[0].submission
    assert adapter.run_calls == 1
    assert submission is not None
    assert submission.status is SubmissionStatus.INVALID_OUTPUT
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.SCHEMA_MISMATCH


def test_runner_enforces_agent_output_budget_before_terminal_commit(tmp_path: Path):
    snapshot = make_case_snapshot()
    defaults = make_config(case_snapshot=snapshot, instance="output-budget")
    tiny_budgets = replace(
        defaults.resource_budgets,
        max_agent_output_bytes=1,
    )
    config = make_config(
        case_snapshot=snapshot,
        instance="output-budget",
        resource_budgets=tiny_budgets,
    )

    result = _runner(tmp_path, _SuccessAdapter()).run(config, snapshot)

    submission = result.trials[0].submission
    assert submission is not None
    assert submission.status is SubmissionStatus.INVALID_OUTPUT
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.OUTPUT_OVERFLOW


def test_runner_commits_materialization_failure_without_calling_agent(
    tmp_path: Path,
):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="workspace-failure")

    def broken_materializer(request):
        del request
        raise RuntimeError("private preparation detail must not leak")

    first_adapter = _SuccessAdapter()
    runner = EvalRunner(
        ArtifactStore(tmp_path / ".eval-runs"),
        None,
        first_adapter,
        materializer_factory=broken_materializer,
    )
    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert result.status.value == "completed"
    assert trial.status.value == "failed"
    assert trial.submission is not None
    assert trial.submission.failure is not None
    assert trial.submission.failure.code is (
        FailureCode.HARNESS_MATERIALIZATION_ERROR
    )
    assert trial.submission.intent is None
    assert trial.submission.review is None
    assert trial.attempt == 1
    assert trial.diagnostic == "materialization failure: RuntimeError"
    assert first_adapter.run_calls == 0
    state = runner.artifact_store.load_trial_state(
        config.run_id, trial.task_id, trial.trial_id
    )
    assert state.terminal_receipt is not None
    assert StageName.PREPARE not in state.completed_stages


def test_runner_returns_sanitized_cleanup_failure_in_trial_diagnostic(
    tmp_path: Path,
) -> None:
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="cleanup-diagnostic")
    adapter = _SuccessAdapter()
    runner = EvalRunner(
        ArtifactStore(tmp_path / ".eval-runs"),
        None,
        adapter,
        materializer_factory=_materializer_factory(
            tmp_path / ".workspaces",
            close_error=PermissionError("cleanup-secret-must-not-leak"),
        ),
        max_workers=1,
    )

    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert trial.submission is not None
    assert trial.submission.status is SubmissionStatus.COMPLETED
    assert trial.diagnostic == "completed Submission; cleanup failure: PermissionError"
    assert "cleanup-secret-must-not-leak" not in trial.diagnostic


def test_strict_capability_preflight_rejects_before_run_plan_creation(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="strict-incompatible")
    adapter = _IncompatibleAdapter()
    runner = _runner(tmp_path, adapter)

    with pytest.raises(RunIncompatibilityError) as raised:
        runner.create_run(config, snapshot, policy=CapabilityPolicy.STRICT)

    assert raised.value.preflight.incompatible_task_ids == (snapshot.cases[0].task_id,)
    assert not (runner.artifact_store.root / config.run_id).exists()
    candidate = runner.artifact_store.load_preflight_candidate(config.run_id)
    assert candidate["preflight"] == raised.value.preflight.to_dict()
    assert adapter.run_calls == 0


def test_strict_preflight_rejects_conflicting_candidate_audit(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="strict-audit-conflict")
    runner = _runner(tmp_path, _IncompatibleAdapter())
    runner.artifact_store.write_preflight_candidate(
        config.run_id,
        run_config_digest=canonical_sha256(config),
        case_snapshot_digest=snapshot.digest(),
        preflight={"stale": True},
    )

    with pytest.raises(ArtifactIntegrityError, match="conflicts"):
        runner.create_run(config, snapshot, policy=CapabilityPolicy.STRICT)

    assert not (runner.artifact_store.root / config.run_id).exists()


def test_filter_mode_never_creates_an_empty_run(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="filter-incompatible")
    runner = _runner(tmp_path, _IncompatibleAdapter())

    with pytest.raises(RunIncompatibilityError, match="empty Run"):
        runner.create_run(config, snapshot, policy=CapabilityPolicy.FILTER)
    assert not (runner.artifact_store.root / config.run_id).exists()


def test_unpersistable_preflight_fails_before_run_directory_creation(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(
        case_snapshot=snapshot,
        instance="preflight-size",
        trial_count=1_000,
    )
    minimum_file = max(
        len(config.to_json().encode("utf-8")),
        len(snapshot.to_json().encode("utf-8")),
    )
    store = ArtifactStore(
        tmp_path / ".eval-runs",
        max_file_bytes=minimum_file + 512,
        max_total_read_bytes=256 * 1024 * 1024,
    )
    runner = EvalRunner(
        store,
        None,
        _SuccessAdapter(),
        materializer_factory=_materializer_factory(tmp_path / ".workspaces"),
    )

    with pytest.raises(ArtifactIntegrityError, match="preflight"):
        runner.create_run(config, snapshot)

    assert not (store.root / config.run_id).exists()


def test_filter_mode_rechecks_final_identity_before_creating_the_run(tmp_path: Path):
    snapshot = _two_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="partial-filter")
    adapter = _SelectiveIncompatibleAdapter()
    runner = _runner(tmp_path, adapter)

    setup = runner.create_run(config, snapshot, policy=CapabilityPolicy.FILTER)

    assert setup.config.run_id != config.run_id
    assert setup.case_snapshot.snapshot_id != snapshot.snapshot_id
    assert tuple(item.task_id for item in setup.case_snapshot.cases) == (
        KEPT_TASK_ID,
    )
    assert setup.preflight.run_id == setup.config.run_id
    assert setup.preflight.filtered_from_run_id == config.run_id
    assert setup.preflight.checked_trials == ((KEPT_TASK_ID, 1),)
    assert setup.preflight.compatible_task_ids == (KEPT_TASK_ID,)
    assert setup.preflight.incompatible_task_ids == ()
    assert setup.preflight.coverage == {
        "checked_trials": 1,
        "compatible_cases": 1,
        "incompatible_cases": 0,
        "issues": 1,
    }
    assert adapter.compatibility_calls == 3
    assert {run_id for _task_id, run_id in adapter.checked} == {
        config.run_id,
        setup.config.run_id,
    }
    assert not (runner.artifact_store.root / config.run_id).exists()
    persisted = runner.artifact_store.load_run_preflight(setup.config.run_id)
    assert persisted == setup.preflight.to_dict()


def test_filter_plan_commits_without_repeating_preflight(tmp_path: Path):
    snapshot = _two_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="planned-filter")
    adapter = _SelectiveIncompatibleAdapter()
    runner = _runner(tmp_path, adapter)

    plan = runner.plan_run(config, snapshot, policy=CapabilityPolicy.FILTER)

    assert plan.config.run_id != config.run_id
    assert tuple(item.task_id for item in plan.case_snapshot.cases) == (
        KEPT_TASK_ID,
    )
    assert adapter.compatibility_calls == 3

    setup = runner.commit_run(plan)

    assert setup.config == plan.config
    assert setup.case_snapshot == plan.case_snapshot
    assert setup.preflight == plan.preflight
    assert adapter.compatibility_calls == 3
    assert runner.artifact_store.load_run_preflight(
        plan.config.run_id
    ) == plan.preflight.to_dict()


def test_run_plan_rejects_direct_construction(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="forged-plan")
    runner = _runner(tmp_path, _SuccessAdapter())
    trusted = runner.plan_run(config, snapshot)

    with pytest.raises(TypeError, match="plan_run"):
        RunPlan(
            config=trusted.config,
            case_snapshot=trusted.case_snapshot,
            preflight=trusted.preflight,
        )


def test_run_plan_cannot_be_committed_by_a_different_runner(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="cross-runner-plan")
    runner_a_root = tmp_path / "runner-a"
    runner_b_root = tmp_path / "runner-b"
    runner_a_root.mkdir()
    runner_b_root.mkdir()
    runner_a = _runner(runner_a_root, _SuccessAdapter())
    runner_b = _runner(runner_b_root, _SuccessAdapter())
    plan = runner_a.plan_run(config, snapshot)

    with pytest.raises(RunnerError, match="different EvalRunner"):
        runner_b.commit_run(plan)

    assert not (runner_b.artifact_store.root / plan.config.run_id).exists()


def test_filter_mode_fails_closed_if_final_identity_drifts(tmp_path: Path):
    snapshot = _two_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="filter-drift")
    adapter = _SelectiveIncompatibleAdapter(drift_after_filter=True)
    runner = _runner(tmp_path, adapter)

    with pytest.raises(RunIncompatibilityError, match="final") as raised:
        runner.create_run(config, snapshot, policy=CapabilityPolicy.FILTER)

    assert raised.value.config.run_id != config.run_id
    assert raised.value.preflight.incompatible_task_ids == (KEPT_TASK_ID,)
    assert not (runner.artifact_store.root / config.run_id).exists()
    assert not (runner.artifact_store.root / raised.value.config.run_id).exists()
    assert adapter.compatibility_calls == 3


def test_resume_rejects_adapter_identity_drift_before_any_trial_work(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="resume-adapter-drift")
    first_runner = _runner(tmp_path, _SuccessAdapter())
    first = first_runner.run(config, snapshot)
    changed_adapter = _SuccessAdapterV2()
    resumed_runner = EvalRunner(
        first_runner.artifact_store,
        None,
        changed_adapter,
        materializer_factory=_materializer_factory(
            tmp_path / ".resume-workspaces"
        ),
    )

    with pytest.raises(RunIncompatibilityError, match="identity") as raised:
        resumed_runner.run(config.run_id)

    assert first.status.value == "completed"
    assert raised.value.preflight.adapter_version == "1.0.0"
    assert changed_adapter.compatibility_calls == 0
    assert changed_adapter.run_calls == 0
    assert not (tmp_path / ".resume-workspaces").exists()

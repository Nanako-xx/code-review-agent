from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from review_agent_eval.adapters.base import AdapterCompatibility
from review_agent_eval.adapters.current_agent import (
    CURRENT_AGENT_ADAPTER_KIND,
    CurrentAgentAdapter,
    current_agent_capabilities,
)
from review_agent_eval.adapters.subprocess_agent import (
    SUBPROCESS_JSON_ADAPTER_KIND,
    SubprocessAgentAdapter,
    subprocess_adapter_capabilities,
)
from review_agent_eval.artifacts import (
    ArtifactStore,
    StageName,
    StageReceipt,
    TrialMaterializationManifest,
)
from review_agent_eval.cases import RunCaseSnapshot, SuiteManifest
from review_agent_eval.config import AgentConfigSnapshot, EvalRunConfig, SuiteRunConfig
from review_agent_eval.frozen_context import (
    FROZEN_CONTEXT_TARGET_PATH,
    FrozenContextTargetMaterializer,
)
from review_agent_eval.materialization import (
    MaterializationError,
    PreparedTargetMaterialization,
)
from review_agent_eval.models import (
    EvalCase,
    EvalInput,
    FailureCode,
    ReviewTargetKind,
    SubmissionStatus,
    canonical_sha256,
)
from review_agent_eval.runner import EvalRunner, RunIncompatibilityError

from .test_artifacts import completed_submission
from .test_config import budgets, evaluator_config, matcher_config
from .test_frozen_context import (
    _eval_input,
    _preparation_binding,
    _prepared_bundle,
)


def _digest(label: str) -> str:
    return canonical_sha256({"fixture": label})


def _frozen_snapshot(
    prepared: Any,
    *,
    preparation_overrides: dict[str, Any] | None = None,
) -> RunCaseSnapshot:
    eval_input = _eval_input(prepared)
    target = eval_input.review_target
    preparation = _preparation_binding(prepared).to_dict()
    if preparation_overrides:
        preparation.update(preparation_overrides)
    case = EvalCase.from_dict(
        {
            "schema_version": "eval_case_v2",
            "task_id": eval_input.task_id,
            "case_version": 1,
            "source": {
                "suite": "swe-frozen-runtime-fixture",
                "origin": "swe_prbench",
                "source_id": eval_input.task_id,
                "source_version": "fixture-v1",
                "source_uri": "https://example.test/swe-prbench",
                "license": "CC-BY-4.0",
                "content_hash": target.source_binding_digest,
            },
            "input": {"review_target": target.to_dict()},
            "clarification_script": {"max_rounds": 1, "answers": []},
            "intent_truth": {
                "scorable": False,
                "authority": None,
                "expected_claims": [],
                "forbidden_claims": [],
                "clarification_policy": None,
            },
            "review_truth": {
                "completeness": "human_observed",
                "novel_finding_policy": "verify",
                "expected_findings": [],
                "known_invalid_findings": [],
            },
            "review_evaluator_context": {"truth_contexts": []},
        }
    )
    case_bytes = case.to_json().encode("utf-8")
    manifest = SuiteManifest.from_dict(
        {
            "schema_version": "suite_manifest_v2",
            "suite_id": "swe-frozen-runtime-fixture",
            "suite_version": "fixture-v1",
            "wire_contract": {
                "case_schema_version": "eval_case_v2",
                "input_schema_version": "eval_input_v2",
                "submission_schema_version": "eval_submission_v2",
                "review_target_kind": "frozen_context",
                "materializer_protocol": "frozen-context-materializer-v2",
            },
            "source": {
                "kind": "public",
                "source_id": "swe-prbench-fixture",
                "source_version": "fixture-v1",
                "source_uri": "https://example.test/swe-prbench",
                "license": "CC-BY-4.0",
                "content_hash": prepared.manifest.source_manifest_digest,
                "preparation_binding": preparation,
            },
            "cases": [
                {
                    "task_id": case.task_id,
                    "case_version": case.case_version,
                    "path": "cases/frozen-runtime.json",
                    "split": "capability",
                    "protocol_id": "official_frozen_context",
                    "dimensions": [],
                    "raw_file_size_bytes": len(case_bytes),
                    "raw_file_sha256": hashlib.sha256(case_bytes).hexdigest(),
                    "canonical_case_digest": case.digest(),
                    "eval_input_digest": eval_input.digest(),
                    "truth_completeness": case.review_truth.completeness.value,
                }
            ],
        }
    )
    return RunCaseSnapshot.build(manifest, ((manifest.cases[0], case),))


def _agent_snapshot(
    *,
    current: bool,
    capabilities: Any = None,
) -> AgentConfigSnapshot:
    executable = str(Path(sys.executable).resolve())
    if current:
        adapter = {
            "kind": CURRENT_AGENT_ADAPTER_KIND,
            "command": [executable, "-m", "review_agent"],
            "review_arguments": ["--reviewer-provider=fake"],
            "environment_allowlist": [],
            "memory_mode": "off",
        }
        agent_id = "agent-current-frozen-preflight"
        provider = "fake"
        model = "fake-reviewer"
    else:
        capabilities = capabilities or subprocess_adapter_capabilities()
        adapter = {
            "kind": SUBPROCESS_JSON_ADAPTER_KIND,
            "command": [executable, "-c", "raise SystemExit(99)"],
            "environment_allowlist": [],
            "capabilities": capabilities.to_dict(),
        }
        agent_id = "agent-frozen-runtime"
        provider = "subprocess"
        model = "none"
    return AgentConfigSnapshot(
        agent_id=agent_id,
        agent_name="Frozen runtime fixture Agent",
        agent_version="2",
        commit="a" * 40,
        model=model,
        provider=provider,
        parameters={"adapter": adapter},
        prompt_config_digest=_digest("agent-prompt-" + agent_id),
    )


def _run_config(
    snapshot: RunCaseSnapshot,
    *,
    current: bool,
    instance: str,
    capabilities: Any = None,
) -> EvalRunConfig:
    capabilities = capabilities or (
        current_agent_capabilities()
        if current
        else subprocess_adapter_capabilities()
    )
    return EvalRunConfig.create(
        run_instance_key=instance,
        agent=_agent_snapshot(
            current=current,
            capabilities=capabilities,
        ),
        clarification_matcher=matcher_config(),
        evaluator=evaluator_config(),
        suite=SuiteRunConfig.from_case_snapshot(snapshot),
        adapter_capabilities=capabilities,
        trial_count=1,
        resource_budgets=budgets(parallel=1),
    )


class _FrozenSuccessAdapter:
    ADAPTER_KIND = SUBPROCESS_JSON_ADAPTER_KIND
    ADAPTER_VERSION = "2"

    def __init__(self) -> None:
        self.run_calls = 0
        self.received_bytes: bytes | None = None
        self.received_access = None

    def compatibility(self, eval_input: EvalInput, config: Any) -> AdapterCompatibility:
        assert eval_input.review_target.kind is ReviewTargetKind.FROZEN_CONTEXT
        assert eval_input.review_target.kind in config.adapter_capabilities.target_kinds
        return AdapterCompatibility()

    def run(
        self,
        eval_input: EvalInput,
        workspace: Path,
        config: Any,
        clarification_channel: Any,
        *,
        target_access: Any,
        target_materialization_id: str,
        cancel_event: Any = None,
    ):
        del clarification_channel, cancel_event
        self.run_calls += 1
        self.received_bytes = (workspace / FROZEN_CONTEXT_TARGET_PATH).read_bytes()
        self.received_access = target_access
        assert target_access.target_materialization_id == target_materialization_id
        return replace(
            completed_submission(config.trial_id),
            task_id=eval_input.task_id,
            agent_id=config.agent_id,
            eval_input_digest=eval_input.digest(),
            target_materialization_id=target_materialization_id,
        )


class _TamperOnSecondValidationLease:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.validation_calls = 0

    @property
    def closed(self) -> bool:
        return self.inner.closed

    @property
    def work_root(self) -> Path:
        return self.inner.work_root

    def validate(self) -> None:
        self.validation_calls += 1
        if self.validation_calls == 2:
            target = self.work_root / FROZEN_CONTEXT_TARGET_PATH
            target.chmod(0o600)
            target.write_bytes(b"replaced after Prepare")
        self.inner.validate()

    def close(self, status: Any) -> None:
        self.inner.close(status)


class _TamperingFrozenMaterializer(FrozenContextTargetMaterializer):
    def materialize(self, request: Any):
        materialized = super().materialize(request)
        return PreparedTargetMaterialization(
            request=materialized.request,
            manifest=materialized.manifest,
            replay=materialized.replay,
            _lease=_TamperOnSecondValidationLease(materialized._lease),
        )


class _WindowDriftLease:
    def __init__(self, inner: Any, mode: str) -> None:
        self.inner = inner
        self.mode = mode
        self.drifted = False

    @property
    def closed(self) -> bool:
        return self.inner.closed

    @property
    def work_root(self) -> Path:
        path = self.inner.work_root
        if not self.drifted:
            self.drifted = True
            held = path.with_name("held-before-prepare")
            path.rename(held)
            if self.mode == "replacement":
                path.mkdir()
                (path / "sentinel.txt").write_text(
                    "replacement must survive",
                    encoding="utf-8",
                )
        return path

    def validate(self) -> None:
        self.inner.validate()

    def close(self, status: Any) -> None:
        self.inner.close(status)


class _WindowDriftMaterializer(FrozenContextTargetMaterializer):
    def __init__(self, *, mode: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        self.materialization_id: str | None = None

    def materialize(self, request: Any):
        materialized = super().materialize(request)
        self.materialization_id = materialized.materialization_id
        return PreparedTargetMaterialization(
            request=materialized.request,
            manifest=materialized.manifest,
            replay=materialized.replay,
            _lease=_WindowDriftLease(materialized._lease, self.mode),
        )


class _TargetDriftingAdapter(_FrozenSuccessAdapter):
    def run(self, *args: Any, **kwargs: Any):
        workspace = args[1]
        submission = super().run(*args, **kwargs)
        target = workspace / FROZEN_CONTEXT_TARGET_PATH
        target.chmod(0o600)
        target.write_bytes(b"Agent replaced immutable Target")
        return submission


def _runner(
    tmp_path: Path,
    prepared: Any,
    adapter: Any,
    *,
    materializer: Any = None,
) -> EvalRunner:
    selected_materializer = materializer or FrozenContextTargetMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".frozen-workspaces",
    )
    return EvalRunner(
        ArtifactStore(tmp_path / ".eval-runs"),
        None,
        adapter,
        target_materializers={
            ReviewTargetKind.FROZEN_CONTEXT: selected_materializer
        },
        max_workers=1,
    )


def test_frozen_target_runs_through_one_runner_and_binds_all_identities(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    snapshot = _frozen_snapshot(prepared)
    config = _run_config(snapshot, current=False, instance="frozen-runtime-success")
    adapter = _FrozenSuccessAdapter()
    runner = _runner(tmp_path, prepared, adapter)

    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert trial.status.value == "completed"
    assert trial.submission is not None
    assert trial.submission.status is SubmissionStatus.COMPLETED
    assert adapter.run_calls == 1
    assert adapter.received_bytes is not None
    assert hashlib.sha256(adapter.received_bytes).hexdigest() == (
        snapshot.eval_input(trial.task_id).review_target.rendered_sha256
    )
    state = runner.artifact_store.load_trial_state(
        config.run_id,
        trial.task_id,
        trial.trial_id,
    )
    assert StageName.PREPARE in state.completed_stages
    assert StageName.AGENT in state.completed_stages

    plan = runner.artifact_store.load_trial_manifest(
        config.run_id,
        trial.task_id,
        trial.trial_id,
    )
    run_root = runner.artifact_store.root / config.run_id
    prepare_path = run_root.joinpath(
        *runner.artifact_store._receipt_path(  # type: ignore[attr-defined]
            plan,
            StageName.PREPARE,
            1,
        ).split("/")
    )
    prepare = StageReceipt.from_json(prepare_path.read_bytes())
    assert prepare.materialization_manifest is not None
    materialization_path = run_root.joinpath(
        *prepare.materialization_manifest.relative_path.split("/")
    )
    materialization = TrialMaterializationManifest.from_json(
        materialization_path.read_bytes()
    )
    assert prepare.materialization_id == materialization.materialization_id
    assert prepare.materialization_id == trial.workspace_binding_id
    assert prepare.materialization_id == trial.submission.target_materialization_id
    assert prepare.target_access == materialization.target_access
    assert adapter.received_access == materialization.target_access
    assert materialization.attempt == 1
    assert materialization.trial_id == trial.trial_id
    assert materialization.eval_input_digest == trial.submission.eval_input_digest


def test_current_agent_is_preflight_incompatible_with_frozen_target(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    snapshot = _frozen_snapshot(prepared)
    config = _run_config(snapshot, current=True, instance="current-frozen-preflight")
    runner = _runner(tmp_path, prepared, CurrentAgentAdapter())

    with pytest.raises(RunIncompatibilityError) as raised:
        runner.create_run(config, snapshot)

    assert raised.value.preflight.incompatible_task_ids == (
        snapshot.cases[0].task_id,
    )
    assert raised.value.preflight.issues[0].reason == (
        "adapter_incompatible.target_kind"
    )
    assert not (runner.artifact_store.root / config.run_id).exists()


def test_repository_only_subprocess_is_preflight_incompatible_with_frozen_target(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    snapshot = _frozen_snapshot(prepared)
    capabilities = subprocess_adapter_capabilities(
        target_kinds=(ReviewTargetKind.REPOSITORY,),
    )
    config = _run_config(
        snapshot,
        current=False,
        instance="repository-only-subprocess-frozen-preflight",
        capabilities=capabilities,
    )
    runner = _runner(tmp_path, prepared, SubprocessAgentAdapter())

    with pytest.raises(RunIncompatibilityError) as raised:
        runner.create_run(config, snapshot)

    assert raised.value.preflight.issues[0].reason == (
        "adapter_incompatible.target_kind"
    )
    assert not (runner.artifact_store.root / config.run_id).exists()


def test_target_replaced_after_prepare_is_terminal_harness_failure_before_agent(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    snapshot = _frozen_snapshot(prepared)
    config = _run_config(
        snapshot,
        current=False,
        instance="frozen-post-prepare-replacement",
    )
    adapter = _FrozenSuccessAdapter()
    materializer = _TamperingFrozenMaterializer(
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".frozen-workspaces",
    )
    runner = _runner(
        tmp_path,
        prepared,
        adapter,
        materializer=materializer,
    )

    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert adapter.run_calls == 0
    assert trial.status.value == "failed"
    assert trial.submission is not None
    assert trial.submission.failure is not None
    assert trial.submission.failure.code is FailureCode.HARNESS_MATERIALIZATION_ERROR
    assert trial.submission.target_materialization_id == trial.workspace_binding_id
    state = runner.artifact_store.load_trial_state(
        config.run_id,
        trial.task_id,
        trial.trial_id,
    )
    assert StageName.PREPARE in state.completed_stages
    assert StageName.AGENT in state.completed_stages
    assert state.terminal_receipt is not None


def test_target_drift_after_adapter_invocation_is_nonretryable_adapter_error(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    snapshot = _frozen_snapshot(prepared)
    config = _run_config(
        snapshot,
        current=False,
        instance="frozen-agent-owned-drift",
    )
    adapter = _TargetDriftingAdapter()
    runner = _runner(tmp_path, prepared, adapter)

    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert adapter.run_calls == 1
    assert trial.submission is not None
    assert trial.submission.failure is not None
    assert trial.submission.failure.code is FailureCode.ADAPTER_ERROR
    assert trial.submission.failure.retryable is False


@pytest.mark.parametrize("mode", ["missing", "replacement"])
def test_materialize_prepare_window_failure_uses_real_binding_without_agent(
    tmp_path: Path,
    mode: str,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    snapshot = _frozen_snapshot(prepared)
    config = _run_config(
        snapshot,
        current=False,
        instance="frozen-prepare-window-" + mode,
    )
    adapter = _FrozenSuccessAdapter()
    materializer = _WindowDriftMaterializer(
        mode=mode,
        bundle_root=prepared.root,
        workspace_root=tmp_path / ".frozen-workspaces",
    )
    runner = _runner(
        tmp_path,
        prepared,
        adapter,
        materializer=materializer,
    )

    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert adapter.run_calls == 0
    assert materializer.materialization_id is not None
    assert trial.submission is not None
    assert trial.submission.failure is not None
    assert trial.submission.failure.code is FailureCode.HARNESS_MATERIALIZATION_ERROR
    assert trial.submission.target_materialization_id == materializer.materialization_id
    assert trial.workspace_binding_id == materializer.materialization_id
    state = runner.artifact_store.load_trial_state(
        config.run_id,
        trial.task_id,
        trial.trial_id,
    )
    assert StageName.PREPARE in state.completed_stages
    assert state.terminal_receipt is not None


def test_frozen_materialization_failure_commits_harness_error_without_agent_call(
    tmp_path: Path,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    snapshot = _frozen_snapshot(prepared)
    config = _run_config(snapshot, current=False, instance="frozen-runtime-tamper")
    binding = prepared.manifest.records[0]
    (prepared.root / binding.path).write_bytes(b"tampered frozen record")
    adapter = _FrozenSuccessAdapter()
    runner = _runner(tmp_path, prepared, adapter)

    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert adapter.run_calls == 0
    assert trial.status.value == "failed"
    assert trial.submission is not None
    assert trial.submission.failure is not None
    assert trial.submission.failure.code is FailureCode.HARNESS_MATERIALIZATION_ERROR
    state = runner.artifact_store.load_trial_state(
        config.run_id,
        trial.task_id,
        trial.trial_id,
    )
    assert StageName.PREPARE not in state.completed_stages
    assert state.terminal_receipt is not None


@pytest.mark.parametrize(
    "field",
    [
        "frozen_bundle_trust_digest",
        "source_catalog_digest",
        "acquisition_receipt_digest",
        "source_manifest_digest",
        "filter_manifest_digest",
        "preparation_packet_digest",
    ],
)
def test_frozen_external_suite_trust_mismatch_never_reaches_agent(
    tmp_path: Path,
    field: str,
) -> None:
    prepared = _prepared_bundle(tmp_path)
    snapshot = _frozen_snapshot(
        prepared,
        preparation_overrides={field: "0" * 64},
    )
    config = _run_config(
        snapshot,
        current=False,
        instance="frozen-external-trust-" + field,
    )
    adapter = _FrozenSuccessAdapter()
    runner = _runner(tmp_path, prepared, adapter)

    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert adapter.run_calls == 0
    assert trial.submission is not None
    assert trial.submission.failure is not None
    assert trial.submission.failure.code is FailureCode.HARNESS_MATERIALIZATION_ERROR
    state = runner.artifact_store.load_trial_state(
        config.run_id,
        trial.task_id,
        trial.trial_id,
    )
    assert StageName.PREPARE not in state.completed_stages

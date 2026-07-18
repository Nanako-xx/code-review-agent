from __future__ import annotations

import base64
import hashlib
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from review_agent_eval.adapters.base import (
    AdapterCompatibility,
    AdapterIncompatibilityReason,
    AgentAdapterIncompatibleError,
)
from review_agent_eval.artifacts import ArtifactIntegrityError, ArtifactStore
from review_agent_eval.config import ResourceBudgets
from review_agent_eval.models import (
    ClarificationAction,
    ClarificationAnswer,
    ClarificationScript,
    EvalInput,
    FailureCode,
    IntentDimension,
    IntentResult,
    SubmissionIntent,
    SubmissionStatus,
    TraceRef,
)
from review_agent_eval.runner import (
    ADAPTER_IDENTITY_MISMATCH,
    AdapterDiagnostic,
    EvalRunner,
    RunIncompatibilityError,
)
from review_agent_eval.submission import failure_submission

from .test_artifacts import TASK_ID, completed_submission, make_case_snapshot, make_config
from .test_config import run_config
from .test_runner import _workspace_factory


class _DynamicIncompatibilityAdapter:
    ADAPTER_KIND = "test-dynamic"
    ADAPTER_VERSION = "1.0.0"

    def __init__(self) -> None:
        self.dynamic = True
        self.run_calls = 0
        self.workspaces = []

    def compatibility(self, eval_input, config):
        del eval_input, config
        return AdapterCompatibility()

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        cancel_event=None,
    ):
        del eval_input, clarification_channel, cancel_event
        self.run_calls += 1
        self.workspaces.append(str(workspace))
        if self.dynamic:
            raise AgentAdapterIncompatibleError(
                AdapterIncompatibilityReason.EXISTING_CI_EVIDENCE
            )
        return completed_submission(config.trial_id)


class _StaticIncompatibilityAdapter(_DynamicIncompatibilityAdapter):
    ADAPTER_KIND = "test-static-incompatible"

    def compatibility(self, eval_input, config):
        del eval_input, config
        raise AgentAdapterIncompatibleError(
            AdapterIncompatibilityReason.EXISTING_CI_EVIDENCE
        )


class _DiagnosticSuccessAdapter:
    ADAPTER_KIND = "test-diagnostic"
    ADAPTER_VERSION = "2.0.0"

    def __init__(self) -> None:
        self.run_calls = 0
        self.last_diagnostics = AdapterDiagnostic(
            stdout=b"ordinary stdout",
            stderr=b"ordinary stderr",
            stdout_bytes=15,
            stderr_bytes=15,
        )

    def compatibility(self, eval_input, config):
        del eval_input, config
        return AdapterCompatibility()

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        cancel_event=None,
    ):
        del eval_input, workspace, clarification_channel, cancel_event
        self.run_calls += 1
        return completed_submission(config.trial_id)


class _ParallelAdapter(_DiagnosticSuccessAdapter):
    def __init__(self, workspaces=None, lock=None) -> None:
        super().__init__()
        self._lock = lock or threading.Lock()
        self.workspaces = workspaces if workspaces is not None else []

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        cancel_event=None,
    ):
        with self._lock:
            self.workspaces.append((config.trial_id, str(workspace)))
        return super().run(
            eval_input,
            workspace,
            config,
            clarification_channel,
            cancel_event=cancel_event,
        )


class _LooseMatcher:
    def __init__(self, digest: str) -> None:
        self.binding_digest = digest

    def equivalent(self, dimension, actual_claim, scripted_claim):
        del dimension
        return " ".join(actual_claim.split()).casefold() == " ".join(
            scripted_claim.split()
        ).casefold()


class _LooseMatcherFactory:
    def build(self, snapshot):
        return _LooseMatcher(snapshot.digest())


class _WrongDigestMatcherFactory:
    def build(self, snapshot):
        del snapshot
        return _LooseMatcher("0" * 64)


class _BrokenMatcher(_LooseMatcher):
    def equivalent(self, dimension, actual_claim, scripted_claim):
        del dimension, actual_claim, scripted_claim
        raise RuntimeError("matcher secret")


class _BrokenMatcherFactory:
    def build(self, snapshot):
        return _BrokenMatcher(snapshot.digest())


class _ClarificationAdapter:
    ADAPTER_KIND = "test-clarification"
    ADAPTER_VERSION = "1.0.0"

    def compatibility(self, eval_input, config):
        del eval_input, config
        return AdapterCompatibility()

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        cancel_event=None,
    ):
        del workspace, cancel_event
        exchange = clarification_channel.ask(
            question_id="question-runner",
            dimension=IntentDimension.GOAL,
            question="Should the goal be confirmed?",
            material_claim="goal claim",
            proposed_values=(),
        )
        intent = SubmissionIntent(
            status=IntentResult.INSUFFICIENT,
            goal=None,
            acceptance_criteria=(),
            scope=(),
            constraints=(),
            claims=(),
            clarification_questions=(exchange,),
            uncertainties=("clarification unresolved",),
        )
        return failure_submission(
            eval_input=eval_input,
            config=config,
            code=FailureCode.CLARIFICATION_REQUIRED,
            message="question remains unresolved",
            retryable=False,
            intent=intent,
        )


class _TraceAdapter(_DiagnosticSuccessAdapter):
    ADAPTER_KIND = "test-trace"

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        cancel_event=None,
    ):
        del eval_input, clarification_channel, cancel_event
        self.run_calls += 1
        trace_dir = workspace / "agent-trace"
        trace_dir.mkdir()
        (trace_dir / "events.jsonl").write_bytes(b'{"event":"private"}\n')
        return replace(
            completed_submission(config.trial_id),
            trace_ref=TraceRef.from_dict(
                {"type": "local_path", "value": "agent-trace"}
            ),
        )


class _FactoryDriftAdapter(_DiagnosticSuccessAdapter):
    ADAPTER_KIND = "test-diagnostic"
    ADAPTER_VERSION = "99.0.0"

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        cancel_event=None,
    ):
        del eval_input, workspace, config, clarification_channel, cancel_event
        raise AssertionError("identity-drifted Adapter must never run")


class _CancellableAdapter(_DiagnosticSuccessAdapter):
    ADAPTER_KIND = "test-cancellable"

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        cancel_event=None,
    ):
        del workspace, clarification_channel
        self.started.set()
        while cancel_event is None or not cancel_event.is_set():
            time.sleep(0.005)
        self.run_calls += 1
        return failure_submission(
            eval_input=eval_input,
            config=config,
            code=FailureCode.PROCESS_KILLED,
            message="cancelled",
            retryable=False,
        )


def _runner(tmp_path: Path, adapter: Any, **kwargs: Any) -> EvalRunner:
    return EvalRunner(
        ArtifactStore(tmp_path / ".eval-runs"),
        None,
        adapter,
        workspace_factory=_workspace_factory(tmp_path / ".workspaces"),
        max_workers=kwargs.pop("max_workers", 1),
        **kwargs,
    )


def test_dynamic_incompatibility_is_not_fabricated_as_adapter_error_and_retry_uses_new_attempt(
    tmp_path: Path,
):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="dynamic-incompatibility")
    adapter = _DynamicIncompatibilityAdapter()
    runner = _runner(tmp_path, adapter)

    first = runner.run(config, snapshot)
    assert first.trials[0].submission is None
    assert first.trials[0].incompatibility == (
        AdapterIncompatibilityReason.EXISTING_CI_EVIDENCE.value
    )
    assert first.status.value == "incomplete"

    adapter.dynamic = False
    second = runner.run(config.run_id)
    assert second.trials[0].submission is not None
    assert second.trials[0].submission.status is SubmissionStatus.COMPLETED
    assert second.trials[0].attempt == 2
    assert len(set(adapter.workspaces)) == 2
    assert "attempt-0001" in adapter.workspaces[0]
    assert "attempt-0002" in adapter.workspaces[1]


def test_execution_artifact_budget_never_blocks_control_plane_submission(tmp_path: Path):
    snapshot = make_case_snapshot()
    tiny = ResourceBudgets(
        agent_timeout_seconds=30,
        evaluator_timeout_seconds=30,
        max_agent_output_bytes=2 * 1024 * 1024,
        max_trace_bytes=64,
        max_execution_artifact_file_bytes=1,
        max_execution_artifact_total_bytes=2 * 1024 * 1024,
        max_parallel_trials=1,
    )
    config = make_config(
        case_snapshot=snapshot,
        instance="tiny-runner-budget",
        resource_budgets=tiny,
    )
    runner = _runner(tmp_path, _DiagnosticSuccessAdapter())

    result = runner.run(config, snapshot)

    assert result.trials[0].submission is not None
    assert result.trials[0].submission.status is SubmissionStatus.COMPLETED
    state = runner.artifact_store.load_trial_state(
        config.run_id, result.trials[0].task_id, result.trials[0].trial_id
    )
    assert state.terminal_receipt is not None
    assert any(
        ref.relative_path.endswith("clarification_match_receipts.json")
        for ref in state.terminal_receipt.artifacts
    )
    assert any(
        ref.relative_path.endswith("terminal_summary.json")
        for ref in state.terminal_receipt.artifacts
    )
    assert runner.artifact_store.load_existing_submission(
        config.run_id, result.trials[0].task_id, result.trials[0].trial_id
    ).status is SubmissionStatus.COMPLETED


def test_clarification_match_receipts_are_harness_private_and_hash_bound(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="clarification-receipt")
    script = ClarificationScript(max_rounds=1, answers=())
    runner = _runner(
        tmp_path,
        _ClarificationAdapter(),
        case_provider={TASK_ID: script},
        matcher_factory=_LooseMatcherFactory(),
    )

    result = runner.run(config, snapshot)

    submission = result.trials[0].submission
    assert submission is not None
    assert submission.status is SubmissionStatus.BLOCKED
    assert submission.intent is not None
    assert len(submission.intent.clarification_questions) == 1
    state = runner.artifact_store.load_trial_state(
        config.run_id, TASK_ID, result.trials[0].trial_id
    )
    assert state.terminal_receipt is not None
    receipt_ref = next(
        ref
        for ref in state.terminal_receipt.artifacts
        if ref.relative_path.endswith("clarification_match_receipts.json")
    )
    payload = runner.artifact_store.read_json_artifact(config.run_id, receipt_ref)
    assert payload["receipts"][0]["outcome"] == "unmatched"
    assert "material_claim" not in str(payload)
    assert "answers" not in str(payload)


def test_parallel_trials_have_distinct_workspace_and_terminal_namespaces(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = run_config(
        snapshot,
        run_instance_key="parallel-runner",
        trial_count=2,
    )
    workspaces = []
    lock = threading.Lock()
    adapter = _ParallelAdapter(workspaces, lock)
    runner = _runner(
        tmp_path,
        adapter,
        max_workers=2,
        adapter_factory=lambda: _ParallelAdapter(workspaces, lock),
    )

    result = runner.run(config, snapshot)

    assert result.status.value == "completed", [
        (
            item.trial_id,
            item.status.value,
            item.diagnostic,
            None if item.submission is None else item.submission.status.value,
        )
        for item in result.trials
    ]
    assert len(result.trials) == 2
    assert len({item.trial_id for item in result.trials}) == 2
    assert len({workspace for _trial, workspace in workspaces}) == 2


def test_parallel_trials_reject_shared_adapter_before_run_creation(
    tmp_path: Path,
):
    snapshot = make_case_snapshot()
    config = run_config(
        snapshot,
        run_instance_key="parallel-unsafe-adapter",
        trial_count=2,
    )
    runner = _runner(tmp_path, _DiagnosticSuccessAdapter(), max_workers=2)

    with pytest.raises(RuntimeError, match="parallel Trials"):
        runner.run(config, snapshot)

    assert not (runner.artifact_store.root / config.run_id).exists()


def test_pre_cancelled_run_commits_process_killed_without_creating_workspace(
    tmp_path: Path,
):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="pre-cancelled")
    adapter = _DiagnosticSuccessAdapter()
    runner = _runner(tmp_path, adapter)
    cancelled = threading.Event()
    cancelled.set()

    result = runner.run(config, snapshot, cancel_event=cancelled)

    submission = result.trials[0].submission
    assert result.status.value == "completed"
    assert submission is not None
    assert submission.status is SubmissionStatus.FAILED
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.PROCESS_KILLED
    assert adapter.run_calls == 0
    assert not (tmp_path / ".workspaces").exists()


def test_trace_capture_is_hash_bound_and_harness_private(tmp_path: Path):
    snapshot = make_case_snapshot()
    budgets = ResourceBudgets(
        agent_timeout_seconds=30,
        evaluator_timeout_seconds=30,
        max_agent_output_bytes=2 * 1024 * 1024,
        max_trace_bytes=64,
        max_execution_artifact_file_bytes=1_024,
        max_execution_artifact_total_bytes=2 * 1024 * 1024,
        max_parallel_trials=1,
    )
    config = make_config(
        case_snapshot=snapshot,
        instance="trace-capture",
        resource_budgets=budgets,
    )
    runner = _runner(tmp_path, _TraceAdapter())

    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert trial.submission is not None
    state = runner.artifact_store.load_trial_state(
        config.run_id, trial.task_id, trial.trial_id
    )
    assert state.terminal_receipt is not None
    trace_ref = next(
        item
        for item in state.terminal_receipt.artifacts
        if item.relative_path.endswith("trace_capture.json")
    )
    payload = runner.artifact_store.read_json_artifact(config.run_id, trace_ref)
    raw = b'{"event":"private"}\n'
    assert payload["captured"] is True
    assert payload["content_omitted"] is False
    assert payload["total_bytes"] == len(raw)
    assert payload["files"] == [
        {
            "path": "agent-trace/events.jsonl",
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "content_base64": base64.b64encode(raw).decode("ascii"),
            "content_truncated": False,
        }
    ]
    assert "private" not in str(payload)


def test_local_trace_that_cannot_fit_budget_keeps_trial_incomplete(tmp_path: Path):
    snapshot = make_case_snapshot()
    budgets = ResourceBudgets(
        agent_timeout_seconds=30,
        evaluator_timeout_seconds=30,
        max_agent_output_bytes=2 * 1024 * 1024,
        max_trace_bytes=64,
        max_execution_artifact_file_bytes=1,
        max_execution_artifact_total_bytes=2 * 1024 * 1024,
        max_parallel_trials=1,
    )
    config = make_config(
        case_snapshot=snapshot,
        instance="trace-budget-failure",
        resource_budgets=budgets,
    )
    runner = _runner(tmp_path, _TraceAdapter())

    result = runner.run(config, snapshot)

    assert result.status.value == "incomplete"
    assert result.trials[0].submission is None
    state = runner.artifact_store.load_trial_state(
        config.run_id,
        result.trials[0].task_id,
        result.trials[0].trial_id,
    )
    assert state.terminal_receipt is None


def test_post_commit_runner_crash_returns_the_canonical_terminal_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="post-commit-crash")
    adapter = _DiagnosticSuccessAdapter()
    runner = _runner(tmp_path, adapter)
    original = runner.artifact_store.finalize_submission
    crashed = False

    def commit_then_crash(*args, **kwargs):
        nonlocal crashed
        state = original(*args, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError("synthetic post-commit crash")
        return state

    monkeypatch.setattr(
        runner.artifact_store,
        "finalize_submission",
        commit_then_crash,
    )

    result = runner.run(config, snapshot)

    assert crashed is True
    assert result.status.value == "completed"
    assert result.trials[0].submission is not None
    assert result.trials[0].submission.status is SubmissionStatus.COMPLETED
    assert runner.artifact_store.load_existing_submission(
        config.run_id,
        result.trials[0].task_id,
        result.trials[0].trial_id,
    ) == result.trials[0].submission


def test_resume_adopts_orphan_submission_before_starting_a_new_attempt(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="orphan-resume")
    adapter = _DiagnosticSuccessAdapter()
    runner = _runner(tmp_path, adapter)
    setup = runner.create_run(config, snapshot)
    plan = setup.manifest.trials[0]
    trial_manifest = runner.artifact_store.load_trial_manifest(
        config.run_id, plan.task_id, plan.trial_id
    )
    running = runner.artifact_store.start_trial(
        config.run_id, plan.task_id, plan.trial_id
    )
    assert running.active_attempt == 1
    runner.artifact_store.write_prepare_stage(
        config.run_id,
        plan.task_id,
        plan.trial_id,
        snapshot.eval_input(plan.task_id),
        attempt=1,
    )
    base = "cases/%s/trials/%s" % (
        trial_manifest.case_path_id,
        plan.trial_id,
    )
    runner_base = "%s/runner/attempt-0001" % base
    runner.artifact_store._write_json(
        config.run_id,
        "%s/clarification_match_receipts.json" % runner_base,
        {"receipts": []},
    )
    runner.artifact_store._write_json(
        config.run_id,
        "%s/terminal_summary.json" % runner_base,
        {"status": "completed"},
    )
    runner.artifact_store._write_json(
        config.run_id,
        "%s/submission.json" % base,
        completed_submission(plan.trial_id),
    )

    result = runner.run(config.run_id)

    assert result.status.value == "completed"
    assert result.trials[0].skipped is True
    assert result.trials[0].attempt == 1
    assert result.trials[0].submission is not None
    assert result.trials[0].submission.status is SubmissionStatus.COMPLETED
    assert adapter.run_calls == 0


def test_per_trial_adapter_factory_identity_drift_is_not_scored_as_agent_failure(
    tmp_path: Path,
):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="factory-drift")
    base_adapter = _DiagnosticSuccessAdapter()
    runner = EvalRunner(
        ArtifactStore(tmp_path / ".eval-runs"),
        None,
        base_adapter,
        workspace_factory=_workspace_factory(tmp_path / ".workspaces"),
        adapter_factory=_FactoryDriftAdapter,
    )

    result = runner.run(config, snapshot)

    trial = result.trials[0]
    assert result.status.value == "incomplete"
    assert trial.status.value == "incomplete"
    assert trial.submission is None
    assert trial.incompatibility == ADAPTER_IDENTITY_MISMATCH
    assert base_adapter.run_calls == 0


def test_running_cancellation_is_forwarded_to_a_cancellable_adapter(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="running-cancel")
    adapter = _CancellableAdapter()
    runner = _runner(tmp_path, adapter)
    cancelled = threading.Event()
    result_holder = {}

    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", runner.run(config, snapshot, cancel_event=cancelled)
        ),
        daemon=True,
    )
    worker.start()
    assert adapter.started.wait(5)
    runner.cancel()
    worker.join(5)

    assert not worker.is_alive()
    assert cancelled.is_set() is False
    result = result_holder["result"]
    submission = result.trials[0].submission
    assert result.status.value == "completed"
    assert submission is not None
    assert submission.failure is not None
    assert submission.failure.code is FailureCode.PROCESS_KILLED


def test_missing_clarification_script_is_a_harness_failure(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="missing-script")
    runner = _runner(tmp_path, _ClarificationAdapter())

    result = runner.run(config, snapshot)

    assert result.status.value == "incomplete"
    assert result.trials[0].submission is None
    assert result.trials[0].diagnostic == "harness failure: RunnerError"


def test_adapter_factory_exception_is_a_harness_failure(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="factory-error")

    def raising_factory():
        raise RuntimeError("factory secret")

    base = _DiagnosticSuccessAdapter()
    runner = EvalRunner(
        ArtifactStore(tmp_path / ".eval-runs"),
        None,
        base,
        workspace_factory=_workspace_factory(tmp_path / ".workspaces"),
        adapter_factory=raising_factory,
    )

    result = runner.run(config, snapshot)

    assert result.status.value == "incomplete"
    assert result.trials[0].submission is None
    assert result.trials[0].diagnostic == "harness failure: _AdapterFactoryError"


def test_matcher_exception_is_a_harness_failure(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="matcher-error")
    script = ClarificationScript(
        max_rounds=1,
        answers=(
            ClarificationAnswer(
                answer_id="answer-runner",
                dimension=IntentDimension.GOAL,
                material_claim="goal claim",
                action=ClarificationAction.REJECT,
                response="No",
                corrected_values=(),
            ),
        ),
    )
    runner = _runner(
        tmp_path,
        _ClarificationAdapter(),
        case_provider={TASK_ID: script},
        matcher_factory=_BrokenMatcherFactory(),
    )

    result = runner.run(config, snapshot)

    assert result.status.value == "incomplete"
    assert result.trials[0].submission is None
    assert result.trials[0].diagnostic == "harness failure: RunnerError"


def test_matcher_binding_digest_mismatch_is_a_harness_failure(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = make_config(
        case_snapshot=snapshot,
        instance="matcher-binding-drift",
    )
    script = ClarificationScript(max_rounds=1, answers=())
    runner = _runner(
        tmp_path,
        _ClarificationAdapter(),
        case_provider={TASK_ID: script},
        matcher_factory=_WrongDigestMatcherFactory(),
    )

    result = runner.run(config, snapshot)

    assert result.status.value == "incomplete"
    assert result.trials[0].submission is None
    assert result.trials[0].diagnostic == "harness failure: RunnerError"


def test_runner_does_not_hide_artifact_integrity_failure_as_agent_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    snapshot = make_case_snapshot()
    config = make_config(case_snapshot=snapshot, instance="artifact-integrity")
    runner = _runner(tmp_path, _DiagnosticSuccessAdapter())

    def broken_finalize(*args, **kwargs):
        raise ArtifactIntegrityError("synthetic receipt conflict")

    monkeypatch.setattr(
        runner.artifact_store,
        "finalize_submission",
        broken_finalize,
    )
    result = runner.run(config, snapshot)

    assert result.status.value == "incomplete"
    assert result.trials[0].submission is None
    assert "harness failure" in result.trials[0].diagnostic


def test_preflight_supports_more_than_4096_incompatible_trials(tmp_path: Path):
    snapshot = make_case_snapshot()
    config = run_config(
        snapshot,
        run_instance_key="large-incompatible-preflight",
        trial_count=4_097,
    )
    runner = _runner(tmp_path, _StaticIncompatibilityAdapter())

    with pytest.raises(RunIncompatibilityError) as raised:
        runner.create_run(config, snapshot)

    error = raised.value
    assert len(error.preflight.checked_trials) == 4_097
    assert len(error.preflight.issues) == 4_097
    candidate = runner.artifact_store.load_preflight_candidate(config.run_id)
    assert candidate["preflight"]["coverage"]["checked_trials"] == 4_097

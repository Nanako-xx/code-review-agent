from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_agent_eval.adapters.base import AgentAdapterError, AgentRunConfig
from review_agent_eval.adapters.current_agent import CurrentAgentAdapter
from review_agent_eval.adapters.subprocess_agent import SubprocessAgentAdapter
from review_agent_eval.clarification import canonical_material_claim_matcher_snapshot
from review_agent_eval.config import (
    AgentConfigSnapshot,
    ResourceBudgets,
    derive_trial_id,
)
from review_agent_eval.models import (
    EvalCase,
    EvalInput,
    EvalSubmission,
    FailureCode,
    SchemaError,
    stable_id,
    validate_submission_for_case,
)
from review_agent_eval.runner import _invoke_adapter
from review_agent_eval.submission import failure_submission, parse_submission_output

from .test_models import case_payload, input_payload, submission_payload


TARGET_MATERIALIZATION_ID = stable_id(
    "workspace-binding", "submission-boundary-v2-tests"
)


class _ForbiddenChannel:
    def ask(self, **kwargs):
        del kwargs
        raise AssertionError("clarification is not expected")


def _config(eval_input: EvalInput, *, max_output_bytes: int = 256 * 1024) -> AgentRunConfig:
    agent = AgentConfigSnapshot(
        agent_id="agent-boundary",
        agent_name="Submission boundary fixture",
        agent_version="1.0.0",
        commit="c" * 40,
        model="none",
        provider="test",
        parameters={},
        prompt_config_digest="d" * 64,
    )
    run_id = stable_id("run", "submission-boundary-v2", agent.digest())
    matcher = canonical_material_claim_matcher_snapshot()
    total = max(max_output_bytes, 16 * 1024)
    return AgentRunConfig._from_verified_binding(
        run_id=run_id,
        task_id=eval_input.task_id,
        eval_input_digest=eval_input.digest(),
        clarification_matcher=matcher,
        clarification_matcher_config_digest=matcher.digest(),
        trial_index=1,
        trial_id=derive_trial_id(run_id, eval_input.task_id, 1),
        agent=agent,
        budgets=ResourceBudgets(
            agent_timeout_seconds=3,
            evaluator_timeout_seconds=3,
            max_agent_output_bytes=max_output_bytes,
            max_trace_bytes=4 * 1024,
            max_execution_artifact_file_bytes=total,
            max_execution_artifact_total_bytes=total,
            max_parallel_trials=1,
        ),
    )


def _completed_submission(
    eval_input: EvalInput,
    config: AgentRunConfig,
    target_materialization_id: str,
) -> EvalSubmission:
    payload = submission_payload()
    payload.update(
        task_id=eval_input.task_id,
        agent_id=config.agent_id,
        trial_id=config.trial_id,
        eval_input_digest=eval_input.digest(),
        target_materialization_id=target_materialization_id,
    )
    return EvalSubmission.from_dict(payload)


class _CapturingAdapter:
    def __init__(self) -> None:
        self.target_materialization_id = None

    def run(
        self,
        eval_input,
        workspace,
        config,
        clarification_channel,
        *,
        target_materialization_id,
        cancel_event,
    ):
        del workspace, clarification_channel, cancel_event
        self.target_materialization_id = target_materialization_id
        return _completed_submission(
            eval_input,
            config,
            target_materialization_id,
        )


def test_runner_adapter_invocation_propagates_materialization_binding(
    tmp_path: Path,
) -> None:
    eval_input = EvalInput.from_dict(input_payload())
    config = _config(eval_input)
    adapter = _CapturingAdapter()

    submission = _invoke_adapter(
        adapter,
        eval_input,
        tmp_path,
        config,
        _ForbiddenChannel(),
        object(),
        target_materialization_id=TARGET_MATERIALIZATION_ID,
    )

    assert adapter.target_materialization_id == TARGET_MATERIALIZATION_ID
    assert submission.target_materialization_id == TARGET_MATERIALIZATION_ID


@pytest.mark.parametrize(
    "adapter",
    [SubprocessAgentAdapter(), CurrentAgentAdapter()],
)
def test_active_adapter_failure_paths_preserve_materialization_binding(
    adapter, tmp_path: Path
) -> None:
    eval_input = EvalInput.from_dict(input_payload())
    config = _config(eval_input)

    submission = adapter.run(
        eval_input,
        tmp_path / "missing-workspace",
        config,
        _ForbiddenChannel(),
        target_materialization_id=TARGET_MATERIALIZATION_ID,
    )

    assert submission.target_materialization_id == TARGET_MATERIALIZATION_ID
    assert submission.eval_input_digest == eval_input.digest()
    assert submission.failure is not None


@pytest.mark.parametrize(
    "data",
    [
        b'{"broken":',
        b'{"value":' + b"9" * 5000 + b"}",
        b'{"duplicate":1,"duplicate":2}',
    ],
)
def test_parse_submission_output_maps_strict_json_failures_to_invalid_json(
    data: bytes,
) -> None:
    eval_input = EvalInput.from_dict(input_payload())
    config = _config(eval_input)

    with pytest.raises(AgentAdapterError) as exc_info:
        parse_submission_output(
            data,
            eval_input=eval_input,
            config=config,
            target_materialization_id=TARGET_MATERIALIZATION_ID,
        )

    assert exc_info.value.code is FailureCode.INVALID_JSON


def test_parse_submission_output_checks_output_limit_before_utf8_decode() -> None:
    eval_input = EvalInput.from_dict(input_payload())
    config = _config(eval_input, max_output_bytes=1024)
    data = b"\xff" * (config.max_output_bytes + 1)

    with pytest.raises(AgentAdapterError) as exc_info:
        parse_submission_output(
            data,
            eval_input=eval_input,
            config=config,
            target_materialization_id=TARGET_MATERIALIZATION_ID,
        )

    assert exc_info.value.code is FailureCode.OUTPUT_OVERFLOW


def test_parse_submission_output_distinguishes_schema_mismatch() -> None:
    eval_input = EvalInput.from_dict(input_payload())
    config = _config(eval_input)

    with pytest.raises(AgentAdapterError) as exc_info:
        parse_submission_output(
            json.dumps({"schema_version": "eval_submission_v2"}).encode("utf-8"),
            eval_input=eval_input,
            config=config,
            target_materialization_id=TARGET_MATERIALIZATION_ID,
        )

    assert exc_info.value.code is FailureCode.SCHEMA_MISMATCH


def test_parse_submission_output_accepts_one_strict_bound_submission() -> None:
    eval_input = EvalInput.from_dict(input_payload())
    config = _config(eval_input)
    expected = _completed_submission(
        eval_input,
        config,
        TARGET_MATERIALIZATION_ID,
    )

    actual = parse_submission_output(
        expected.to_json().encode("utf-8"),
        eval_input=eval_input,
        config=config,
        target_materialization_id=TARGET_MATERIALIZATION_ID,
    )

    assert actual == expected


@pytest.mark.parametrize("mismatch", ["task", "digest"])
def test_failure_submission_rejects_input_binding_mismatch(mismatch: str) -> None:
    eval_input = EvalInput.from_dict(input_payload())
    config = _config(eval_input)
    changed = input_payload()
    if mismatch == "task":
        changed["task_id"] = "different-task"
    else:
        changed["review_target"]["review_request"]["title"] = "Changed title"
    mismatched_input = EvalInput.from_dict(changed)

    with pytest.raises(SchemaError, match=mismatch):
        failure_submission(
            eval_input=mismatched_input,
            config=config,
            target_materialization_id=TARGET_MATERIALIZATION_ID,
            code=FailureCode.TIMEOUT,
            message="timed out",
            retryable=True,
        )


def test_case_validation_rejects_input_digest_mismatch_before_null_intent_return() -> None:
    case = EvalCase.from_dict(case_payload())
    payload = submission_payload()
    payload.update(
        task_id=case.task_id,
        eval_input_digest="0" * 64,
        status="failed",
        intent=None,
        review=None,
        failure={"code": "timeout", "message": "timed out", "retryable": True},
    )
    submission = EvalSubmission.from_dict(payload)

    with pytest.raises(SchemaError, match="digest"):
        validate_submission_for_case(submission, case)

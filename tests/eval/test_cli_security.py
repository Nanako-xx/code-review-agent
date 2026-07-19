from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import pytest

import review_agent_eval.cli as cli_module
from review_agent_eval.artifacts import ArtifactStore
from review_agent_eval.cli import EXIT_OK, EXIT_USAGE, main
from review_agent_eval.config import EvaluatorExecutionConfig
from review_agent_eval.models import TraceRef, TraceType

from .test_artifacts import (
    TASK_ID,
    completed_submission,
    make_input,
    make_store,
    required_runner_artifacts,
)


@pytest.mark.parametrize(
    ("command", "foreign_arguments", "handler_name"),
    (
        (
            "run-agent",
            ("--judge-provider", "fake"),
            "_handle_run_agent",
        ),
        (
            "run-agent",
            ("--evaluator-execution-config", "judge-execution.json"),
            "_handle_run_agent",
        ),
        (
            "evaluate",
            ("--agent-provider", "fake"),
            "_handle_evaluate",
        ),
        (
            "evaluate",
            ("--agent-command", "agent-command"),
            "_handle_evaluate",
        ),
    ),
)
def test_cli_rejects_foreign_stage_options_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    foreign_arguments: tuple[str, ...],
    handler_name: str,
) -> None:
    """Agent and Judge options cannot cross the stage boundary."""

    dispatched: list[Any] = []

    def forbidden_dispatch(args: Any) -> int:
        dispatched.append(args)
        raise AssertionError("foreign CLI arguments reached a stage handler")

    monkeypatch.setattr(cli_module, handler_name, forbidden_dispatch)

    code = main(
        [
            command,
            "run-security-boundary",
            "--suite-root",
            str(tmp_path / "suite"),
            *foreign_arguments,
        ]
    )
    captured = capsys.readouterr()

    assert code == EXIT_USAGE
    assert dispatched == []
    # argparse owns the diagnostic; the application must not emit a second
    # JSON error envelope or invoke a stage before rejecting the option.
    assert captured.out == ""
    assert captured.err


def _security_fixture(
    tmp_path: Path,
) -> tuple[
    ArtifactStore,
    Any,
    Any,
    Any,
    str,
    str,
    str,
    str,
]:
    """Create a receipt-bound evaluation containing controlled sensitive data."""

    store, config, _manifest, plan, _trial = make_store(tmp_path)
    running = store.start_trial(config.run_id, TASK_ID, plan.trial_id)
    assert running.active_attempt is not None
    store.write_prepare_stage(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        make_input(),
        attempt=running.active_attempt,
    )

    trace_value = "opaque-trace-ref-security-sentinel"
    submission = replace(
        completed_submission(plan.trial_id),
        trace_ref=TraceRef(TraceType.OPAQUE_ID, trace_value),
    )
    store.finalize_submission(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        submission,
        attempt=running.active_attempt,
        runner_artifacts=required_runner_artifacts(submission),
    )

    execution = EvaluatorExecutionConfig.from_resource_budgets(
        config.evaluator,
        config.resource_budgets,
    )
    raw_context = "raw-judge-context-security-sentinel"
    api_key_value = "opaque-api-key-security-sentinel"
    absolute_path = str(
        (tmp_path / "private" / "raw-judge-context.txt").resolve()
    )
    context_text = (
        f"{raw_context} at {absolute_path} with opaque-value {api_key_value}"
    )

    # These are intentionally shaped like the persisted Judge artifacts.  The
    # context is present in both the request and result so the test proves the
    # inspect projection, rather than merely relying on a missing field.
    judge_request = {
        "request_id": "judge-request-security-1",
        "task": "review_finding_equivalence",
        "context_blocks": [{"content": context_text}],
    }
    judge_input = {
        "schema_version": "eval_judge_input_artifact_v1",
        "evaluator_execution_digest": execution.digest(),
        "requests": [judge_request],
    }
    judge_output = {
        "schema_version": "eval_judge_output_artifact_v1",
        "results": [
            {
                "request": judge_request,
                "status": "graded",
                "source": "fake",
                "failure": None,
                "ungraded_reason": None,
            }
        ],
    }
    receipt = store.write_evaluation(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        evaluator_execution=execution,
        revision="cli-security-v1",
        intent_matches={
            "schema_version": "eval_intent_evaluation_v1",
            "status": "graded",
            "reason_codes": [],
        },
        review_matches={
            "schema_version": "eval_review_evaluation_v1",
            "status": "graded",
            "reason_codes": [],
        },
        judge_input=judge_input,
        judge_output=judge_output,
        score={
            "schema_version": "eval_trial_score_v1",
            "trial_id": plan.trial_id,
            "trial_index": 1,
            "submission_status": "completed",
            "metrics": [],
            "usage": {},
            "reason_codes": [],
        },
        report="# Redacted trial report\n",
    )

    loaded = store.load_evaluation_bundle(
        config.run_id,
        TASK_ID,
        plan.trial_id,
        receipt.evaluation_id,
    )
    # Assert the source fixture really contains the values that must not cross
    # the public inspection boundary.
    assert (
        loaded.judge_input["requests"][0]["context_blocks"][0]["content"]
        == context_text
    )
    assert trace_value == store.load_existing_submission(
        config.run_id,
        TASK_ID,
        plan.trial_id,
    ).trace_ref.value
    return (
        store,
        config,
        plan,
        receipt,
        trace_value,
        raw_context,
        api_key_value,
        absolute_path,
    )


def _root_arguments(tmp_path: Path, store: ArtifactStore) -> list[str]:
    return [
        "--suite-root",
        str(tmp_path / "suite"),
        "--runs-root",
        str(store.root),
        "--data-root",
        str(tmp_path / ".eval-data"),
        "--workspace-root",
        str(tmp_path / ".eval-workspaces"),
    ]


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str):
                yield key
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str):
                yield key
            yield from _keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _keys(child)


def test_inspect_json_is_a_redacted_projection_of_trace_and_judge_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (
        store,
        config,
        plan,
        receipt,
        trace_value,
        raw_context,
        api_key_value,
        absolute_path,
    ) = _security_fixture(tmp_path)
    # Keep an actual credential value in the process environment while
    # inspecting.  The inspect stage must not turn environment-backed model
    # configuration into output, even though only the variable name belongs
    # in persisted configuration.
    monkeypatch.setenv("REVIEW_AGENT_EVAL_API_KEY", api_key_value)
    (tmp_path / "suite").mkdir()

    code = main(
        [
            "inspect",
            config.run_id,
            *_root_arguments(tmp_path, store),
            "--task-id",
            TASK_ID,
            "--trial-id",
            plan.trial_id,
            "--evaluation-id",
            receipt.evaluation_id,
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()

    assert code == EXIT_OK
    assert captured.err == ""
    payload = json.loads(captured.out)
    inspection = payload["inspection"]

    trace = inspection["submission"]["trace"]
    assert trace == {
        "present": True,
        "type": TraceType.OPAQUE_ID.value,
        "value": "redacted",
    }

    judge = inspection["judge"]
    assert set(judge) == {
        "evaluator_execution_digest",
        "request_count",
        "status_counts",
        "source_counts",
        "task_counts",
        "failure_count",
        "ungraded_count",
    }
    assert judge["request_count"] == 1
    assert judge["status_counts"] == {"graded": 1}
    assert judge["source_counts"] == {"fake": 1}
    assert judge["task_counts"] == {"review_finding_equivalence": 1}
    assert judge["failure_count"] == 0
    assert judge["ungraded_count"] == 0

    # The public contract exposes Judge decision metadata only; raw request
    # context and trace-ref payload fields must not be projected under a new
    # nesting shape either.
    assert "judge_input" not in inspection
    assert "context_blocks" not in set(_keys(inspection))
    assert "content" not in set(_keys(inspection))
    assert "request" not in set(_keys(inspection["judge"]))

    output_strings = tuple(_strings(payload))
    for forbidden in (
        trace_value,
        raw_context,
        api_key_value,
        absolute_path,
        Path(absolute_path).as_posix(),
    ):
        assert all(forbidden not in value for value in output_strings)

    # The human-readable projection is also receipt-bound.  Check it through
    # the same public command so a future format-specific path cannot bypass
    # the redaction contract.
    code = main(
        [
            "inspect",
            config.run_id,
            *_root_arguments(tmp_path, store),
            "--task-id",
            TASK_ID,
            "--trial-id",
            plan.trial_id,
            "--evaluation-id",
            receipt.evaluation_id,
            "--format",
            "markdown",
        ]
    )
    markdown = capsys.readouterr()
    assert code == EXIT_OK
    assert markdown.err == ""
    assert markdown.out.startswith("# Evaluation inspection\n")
    assert "# Redacted trial report" not in markdown.out
    for forbidden in (
        trace_value,
        raw_context,
        api_key_value,
        absolute_path,
        Path(absolute_path).as_posix(),
    ):
        assert forbidden not in markdown.out

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from review_agent_eval.artifacts import ArtifactStore
from review_agent_eval.cli import EXIT_OK, main

from .test_cli import _root_arguments, _without_message, _write_cli_suite


def _json_command(
    args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> dict:
    assert main(args) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def test_rejudge_changes_report_identity_and_resume_reuses_exact_namespace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots, program = _write_cli_suite(tmp_path)
    common = _root_arguments(roots)
    agent_command = [
        str(Path(sys.executable).resolve()),
        str(program),
        "success",
        "{agent_id}",
        "{task_id}",
        "{trial_id}",
    ]

    prepare_args = [
        "prepare",
        *common,
        "--agent-adapter",
        "subprocess",
        "--json",
    ]
    for item in agent_command:
        prepare_args.extend(("--agent-command", item))
    prepared = _json_command(prepare_args, capsys)
    run_id = prepared["run_id"]

    agent_run = _json_command(
        ["run-agent", run_id, *common, "--json"],
        capsys,
    )
    trial_id = agent_run["trials"][0]["trial_id"]
    store = ArtifactStore(roots["runs"], create_root=False)
    submission_before = store.load_existing_submission(
        run_id,
        "task-001",
        trial_id,
    )

    evaluation_args = [
        "evaluate",
        run_id,
        *common,
        "--revision",
        "rejudge-regression-v1",
        "--judge-provider",
        "fake",
        "--judge-model",
        "judge-model-v1",
        "--json",
    ]
    first = _json_command(evaluation_args, capsys)
    first_namespace = store.load_run_evaluation(
        run_id,
        first["evaluation_id"],
    )
    first_trial_namespace = store.load_evaluation_bundle(
        run_id,
        "task-001",
        trial_id,
        first["evaluation_id"],
    )

    resumed = _json_command(
        [*evaluation_args, "--resume"],
        capsys,
    )
    resumed_namespace = store.load_run_evaluation(
        run_id,
        resumed["evaluation_id"],
    )
    resumed_trial_namespace = store.load_evaluation_bundle(
        run_id,
        "task-001",
        trial_id,
        resumed["evaluation_id"],
    )

    changed_args = list(evaluation_args)
    changed_args[changed_args.index("--judge-model") + 1] = "judge-model-v2"
    changed = _json_command(changed_args, capsys)
    changed_namespace = store.load_run_evaluation(
        run_id,
        changed["evaluation_id"],
    )
    changed_trial_namespace = store.load_evaluation_bundle(
        run_id,
        "task-001",
        trial_id,
        changed["evaluation_id"],
    )
    submission_after = store.load_existing_submission(
        run_id,
        "task-001",
        trial_id,
    )

    # Same evaluator execution + revision resumes the exact immutable Run
    # report namespace, including both ReportBuilder's summary and Markdown.
    assert _without_message(resumed) == _without_message(first)
    assert resumed_namespace.to_dict() == first_namespace.to_dict()
    assert resumed_trial_namespace.to_dict() == first_trial_namespace.to_dict()

    # Changing only the Judge model changes the evaluator execution digest;
    # the existing Submission remains the same source for both evaluations.
    assert changed["evaluation_id"] != first["evaluation_id"]
    assert changed["summary_id"] != first["summary_id"]
    assert changed["report_digest"] != first["report_digest"]
    assert changed_namespace.namespace.evaluator_execution_digest != (
        first_namespace.namespace.evaluator_execution_digest
    )
    assert changed_namespace.report != first_namespace.report
    assert changed_trial_namespace.evaluation_id != first_trial_namespace.evaluation_id
    assert changed_trial_namespace.report != first_trial_namespace.report
    assert changed_trial_namespace.submission_digest == (
        first_trial_namespace.submission_digest
    )
    assert submission_after.digest() == submission_before.digest()
    assert len(store.list_run_evaluations(run_id)) == 2

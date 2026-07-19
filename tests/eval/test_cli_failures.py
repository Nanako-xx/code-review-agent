from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_agent_eval.cli import (
    EXIT_CONFLICT,
    EXIT_PRECONDITION,
    EXIT_USAGE,
    main,
)

from .test_datasets import write_suite


def test_prepare_rejects_overwrite_before_creating_control_roots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = tmp_path / "suite"
    write_suite(suite)
    runs = tmp_path / ".eval-runs"

    code = main(
        [
            "prepare",
            "--suite-root",
            str(suite),
            "--runs-root",
            str(runs),
            "--data-root",
            str(tmp_path / ".eval-data"),
            "--workspace-root",
            str(tmp_path / ".eval-workspaces"),
            "--overwrite",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_CONFLICT
    assert payload["error_code"] == "conflict"
    assert not runs.exists()


def test_missing_suite_fails_without_echoing_absolute_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "private-user-directory" / "missing-suite"

    code = main(
        [
            "prepare",
            "--suite-root",
            str(missing),
            "--dry-run",
            "--json",
        ]
    )
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert code == EXIT_PRECONDITION
    assert str(tmp_path) not in output
    assert payload["status"] == "error"


def test_cli_usage_code_is_stable_for_missing_required_arguments() -> None:
    assert main(["prepare"]) == EXIT_USAGE


def test_missing_run_is_a_precondition_failure_not_an_operational_crash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    code = main(
        [
            "run-agent",
            "run-does-not-exist",
            "--suite-root",
            str(suite),
            "--runs-root",
            str(tmp_path / ".eval-runs"),
            "--data-root",
            str(tmp_path / ".eval-data"),
            "--workspace-root",
            str(tmp_path / ".eval-workspaces"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == EXIT_PRECONDITION
    assert payload["error_code"] == "precondition"
    assert payload["message"] in {
        "ArtifactStateError",
        "FileNotFoundError",
        "CliPreconditionError",
    }

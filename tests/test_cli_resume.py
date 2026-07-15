from pathlib import Path
import json

import pytest

from conftest import run_git
from review_agent.cli import main
from review_agent.session import (
    SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION,
    session_phases_for_schema,
)


@pytest.fixture(autouse=True)
def cli_memory_root(tmp_path: Path, monkeypatch) -> Path:
    root = (tmp_path / "memory-root").resolve()
    monkeypatch.setenv("REVIEW_AGENT_MEMORY_ROOT", str(root))
    return root


def test_cli_resume_prints_completed_run_summary(
    git_repo: Path,
    capsys,
    cli_memory_root: Path,
) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return token == 'ok'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")
    assert (
        main(
            [
                "review",
                "--repo",
                str(git_repo),
                "--base",
                "HEAD~1",
                "--head",
                "HEAD",
                "--intent",
                "Add auth token check",
                "--non-interactive",
            ]
        )
        == 0
    )
    capsys.readouterr()
    run_id = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1].name

    assert main(["resume", run_id, "--repo", str(git_repo)]) == 0

    output = capsys.readouterr().out
    assert "Resume" in output
    assert f"Review ID: {run_id}" in output
    assert "Status: completed" in output
    assert "Phase: completed" in output
    assert "Requested Base: HEAD~1" in output
    assert "Requested Head: HEAD" in output
    assert f"Resolved Base: {base}" in output
    assert f"Resolved Head: {head}" in output
    assert "final_risk.json (present)" in output
    assert "report.md (present)" in output
    assert "review_brief.json (present)" in output
    assert "Memory mode: read-write" in output
    assert "Memory root fingerprint:" in output
    assert "Audit: valid" in output


def test_cli_resume_legacy_run_without_session_uses_state_revisions(git_repo: Path, capsys) -> None:
    review_id = "review-legacy"
    run_dir = git_repo / ".review-agent" / "runs" / review_id
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        json.dumps(
            {
                "review_id": review_id,
                "status": "completed",
                "phase": "completed",
                "repository_path": str(git_repo),
                "base_revision": "legacy-base",
                "head_revision": "legacy-head",
                "message": "Legacy run completed",
                "artifacts": {"request": "request.json"},
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "request.json").write_text(
        json.dumps(
            {
                "repository_path": str(git_repo),
                "base_revision": "legacy-base",
                "head_revision": "legacy-head",
                "user_intent": None,
                "review_focus": None,
            }
        ),
        encoding="utf-8",
    )

    assert main(["resume", review_id, "--repo", str(git_repo)]) == 0

    output = capsys.readouterr().out
    assert "Base: legacy-base" in output
    assert "Head: legacy-head" in output
    assert "Resolved Base:" not in output
    assert "Resolved Head:" not in output


def test_cli_resume_session_does_not_require_legacy_state(git_repo: Path, capsys) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return bool(token)\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")
    assert main(["review", "--repo", str(git_repo), "--base", base, "--head", head, "--non-interactive"]) == 0
    capsys.readouterr()
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    (run_dir / "state.json").unlink()

    assert main(["resume", run_dir.name, "--repo", str(git_repo)]) == 0

    output = capsys.readouterr().out
    assert "Status: completed" in output
    assert "Phase: completed" in output
    assert f"Repository: {git_repo}" in output
    assert f"Resolved Base: {base}" in output
    assert f"Resolved Head: {head}" in output
    assert "report.md (present)" in output
    assert "Audit: valid" in output


def test_cli_resume_session_ignores_stale_legacy_state(git_repo: Path, capsys) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return bool(token)\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")
    assert main(["review", "--repo", str(git_repo), "--base", base, "--head", head, "--non-interactive"]) == 0
    capsys.readouterr()
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    state_path = run_dir / "state.json"
    stale_state = json.loads(state_path.read_text(encoding="utf-8"))
    stale_state.update(
        {
            "status": "failed",
            "phase": "failed",
            "repository_path": "stale-repository",
            "errors": ["stale-state-error"],
        }
    )
    state_path.write_text(json.dumps(stale_state), encoding="utf-8")

    assert main(["resume", run_dir.name, "--repo", str(git_repo)]) == 0

    output = capsys.readouterr().out
    assert "Status: completed" in output
    assert "Phase: completed" in output
    assert f"Repository: {git_repo}" in output
    assert "stale-repository" not in output
    assert "stale-state-error" not in output
    assert "Audit: valid" in output


def test_cli_resume_completed_session_rebuilds_tampered_reporting_phase(git_repo: Path, capsys) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return bool(token)\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")
    assert main(["review", "--repo", str(git_repo), "--base", base, "--head", head, "--non-interactive"]) == 0
    capsys.readouterr()
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    (run_dir / "report.md").write_text("tampered report\n", encoding="utf-8")

    exit_code = main(["resume", run_dir.name, "--repo", str(git_repo)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Action: continue_session" in output
    assert "Starting phase: reporting" in output
    assert "tampered report" not in (run_dir / "report.md").read_text(encoding="utf-8")


def test_cli_resume_completed_session_rebuilds_missing_reporting_artifact(git_repo: Path, capsys) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return bool(token)\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")
    assert main(["review", "--repo", str(git_repo), "--base", base, "--head", head, "--non-interactive"]) == 0
    capsys.readouterr()
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    (run_dir / "report.md").unlink()

    exit_code = main(["resume", run_dir.name, "--repo", str(git_repo)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Action: continue_session" in output
    assert "Starting phase: reporting" in output
    assert (run_dir / "report.md").exists()


def test_cli_resume_revision_drift_creates_and_prints_child_session(
    git_repo: Path,
    capsys,
    cli_memory_root: Path,
    monkeypatch,
    tmp_path: Path,
) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text(
        "def check(token):\n    return bool(token)\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    parent_head = run_git(git_repo, "rev-parse", "HEAD")
    assert (
        main(
            [
                "review",
                "--repo",
                str(git_repo),
                "--base",
                base,
                "--head",
                "HEAD",
                "--intent",
                "Add auth token check",
                "--memory-mode",
                "read",
                "--memory-curator-mode",
                "model",
                "--memory-curator-provider",
                "fake",
                "--memory-curator-model",
                "fixed-curator-model",
                "--memory-curator-api-key-env",
                "FIXED_CURATOR_API_KEY",
                "--non-interactive",
            ]
        )
        == 0
    )
    capsys.readouterr()
    runs_root = git_repo / ".review-agent" / "runs"
    parent_id = next(runs_root.iterdir()).name
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move symbolic head")
    child_head = run_git(git_repo, "rev-parse", "HEAD")
    changed_environment_root = (tmp_path / "changed-memory-root").resolve()
    monkeypatch.setenv("REVIEW_AGENT_MEMORY_ROOT", str(changed_environment_root))
    monkeypatch.setenv("FIXED_CURATOR_API_KEY", "changed-secret-value")

    assert main(["resume", parent_id, "--repo", str(git_repo)]) == 0

    output = capsys.readouterr().out
    child_line = next(
        line for line in output.splitlines() if line.strip().startswith("New review:")
    )
    child_id = child_line.split(":", 1)[1].strip()
    assert "Action: create_incremental_session" in output
    assert f"Parent review: {parent_id}" in output
    assert f"Review ID: {child_id}" in output
    assert "Change: head_moved" in output
    assert f"Full range: {base}..{child_head}" in output
    assert f"Incremental priority range: {parent_head}..{child_head}" in output
    assert (runs_root / child_id / "incremental_priority.json").exists()
    assert (runs_root / child_id / "report.md").exists()
    child_session_text = (runs_root / child_id / "session.json").read_text(
        encoding="utf-8"
    )
    child_session = json.loads(child_session_text)
    assert child_session["execution"]["memory"]["mode"] == "read"
    assert (
        child_session["execution"]["memory"]["root_path"]
        == cli_memory_root.as_posix()
    )
    assert child_session["execution"]["memory_curator"]["model"] == (
        "fixed-curator-model"
    )
    assert child_session["execution"]["memory_curator"]["api_key_env"] == (
        "FIXED_CURATOR_API_KEY"
    )
    assert "changed-secret-value" not in child_session_text
    assert not changed_environment_root.exists()


def test_cli_resume_legacy_drift_requires_and_accepts_explicit_v5_upgrade(
    git_repo: Path,
    capsys,
    cli_memory_root: Path,
) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text(
        "def check(token):\n    return bool(token)\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    assert (
        main(
            [
                "review",
                "--repo",
                str(git_repo),
                "--base",
                base,
                "--head",
                "HEAD",
                "--intent",
                "Add auth token check",
                "--memory-mode",
                "read",
                "--non-interactive",
            ]
        )
        == 0
    )
    capsys.readouterr()
    runs_root = git_repo / ".review-agent" / "runs"
    parent_id = next(path.name for path in runs_root.iterdir() if path.is_dir())
    parent_dir = runs_root / parent_id
    payload = json.loads((parent_dir / "session.json").read_text(encoding="utf-8"))
    payload["schema_version"] = SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION
    payload["execution"].pop("memory", None)
    payload["execution"].pop("memory_curator", None)
    legacy_phases = {
        phase.value
        for phase in session_phases_for_schema(
            SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION
        )
    }
    payload["phases"] = {
        name: checkpoint
        for name, checkpoint in payload["phases"].items()
        if name in legacy_phases
    }
    payload["artifacts"] = {
        name: descriptor
        for name, descriptor in payload["artifacts"].items()
        if descriptor["phase"] in legacy_phases
    }
    (parent_dir / "session.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move legacy symbolic head")

    assert main(["resume", parent_id, "--repo", str(git_repo)]) == 2
    assert "explicit compatible v5" in capsys.readouterr().err

    memory_root = (git_repo / "explicit-memory-root").resolve()
    assert (
        main(
            [
                "resume",
                parent_id,
                "--repo",
                str(git_repo),
                "--upgrade-to-v5",
                "--memory-mode",
                "read",
                "--memory-root",
                str(memory_root),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    child_line = next(
        line for line in output.splitlines() if line.strip().startswith("New review:")
    )
    child_id = child_line.split(":", 1)[1].strip()
    child = json.loads((runs_root / child_id / "session.json").read_text(encoding="utf-8"))
    assert child["schema_version"] == 5
    assert child["execution"]["memory"]["mode"] == "read"
    assert child["execution"]["memory"]["root_path"] == memory_root.as_posix()


def test_cli_resume_missing_run_returns_usage_error(tmp_path: Path, capsys) -> None:
    exit_code = main(["resume", "missing-review", "--repo", str(tmp_path)])

    assert exit_code == 2
    assert "Review run not found" in capsys.readouterr().err

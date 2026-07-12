from pathlib import Path
import json

from conftest import run_git
from review_agent.cli import main


def test_cli_resume_prints_completed_run_summary(git_repo: Path, capsys) -> None:
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
    assert main(["review", "--repo", str(git_repo), "--base", base, "--head", head]) == 0
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
    assert main(["review", "--repo", str(git_repo), "--base", base, "--head", head]) == 0
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
    assert main(["review", "--repo", str(git_repo), "--base", base, "--head", head]) == 0
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
    assert main(["review", "--repo", str(git_repo), "--base", base, "--head", head]) == 0
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


def test_cli_resume_missing_run_returns_usage_error(tmp_path: Path, capsys) -> None:
    exit_code = main(["resume", "missing-review", "--repo", str(tmp_path)])

    assert exit_code == 2
    assert "Review run not found" in capsys.readouterr().err

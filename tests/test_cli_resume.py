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


def test_cli_resume_missing_run_returns_usage_error(tmp_path: Path, capsys) -> None:
    exit_code = main(["resume", "missing-review", "--repo", str(tmp_path)])

    assert exit_code == 2
    assert "Review run not found" in capsys.readouterr().err

from pathlib import Path

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
                base,
                "--head",
                head,
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
    assert "report.md (present)" in output


def test_cli_resume_missing_run_returns_usage_error(tmp_path: Path, capsys) -> None:
    exit_code = main(["resume", "missing-review", "--repo", str(tmp_path)])

    assert exit_code == 2
    assert "Review run not found" in capsys.readouterr().err

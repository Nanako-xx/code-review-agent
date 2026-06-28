from pathlib import Path
import json

from conftest import run_git
from review_agent.cli import main


def test_cli_review_writes_current_schema_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return token == 'ok'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
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
            "--focus",
            "regression safety",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_root = git_repo / ".review-agent" / "runs"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "request.json").exists()
    assert (run_dirs[0] / "intent.json").exists()
    assert (run_dirs[0] / "risk.json").exists()
    assert (run_dirs[0] / "assignments.json").exists()
    assert (run_dirs[0] / "report.md").exists()

    intent = json.loads((run_dirs[0] / "intent.json").read_text(encoding="utf-8"))
    risk = json.loads((run_dirs[0] / "risk.json").read_text(encoding="utf-8"))
    assignments = json.loads((run_dirs[0] / "assignments.json").read_text(encoding="utf-8"))

    assert "uncertainties" in intent
    assert "unknowns" not in intent
    assert "signal_refs" in risk
    assert "evidence_refs" not in risk
    assert "initial_context" in assignments["assignments"][0]
    assert "provided_evidence_refs" not in assignments["assignments"][0]

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


def test_cli_review_with_fake_reviewer_writes_reviewer_artifacts(git_repo: Path):
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
            "--reviewer-provider",
            "fake",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_root = git_repo / ".review-agent" / "runs"
    run_dirs = sorted(run_root.iterdir())
    run_dir = run_dirs[-1]

    assert (run_dir / "reviewer_envelope.json").exists()
    assert (run_dir / "reviewer_raw_response.json").exists()
    assert (run_dir / "reviewer_result.json").exists()

    result = json.loads((run_dir / "reviewer_result.json").read_text(encoding="utf-8"))
    raw = json.loads((run_dir / "reviewer_raw_response.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert result["status"] == "partial"
    assert raw["provider_name"] == "fake"
    assert "## Single Reviewer Result" in report
    assert "Fake reviewer executed." in report


def test_cli_openai_compatible_provider_requires_api_key(git_repo: Path, monkeypatch, capsys):
    monkeypatch.delenv("REVIEW_AGENT_API_KEY", raising=False)
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
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
            "--reviewer-provider",
            "openai-compatible",
            "--reviewer-model",
            "review-model",
            "--reviewer-base-url",
            "https://example.test/v1",
            "--non-interactive",
        ]
    )

    assert exit_code == 2
    assert "Reviewer provider configuration error" in capsys.readouterr().out

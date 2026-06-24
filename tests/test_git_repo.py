from pathlib import Path

from conftest import run_git
from review_agent.git_repo import collect_change_summary


def test_collect_change_summary_lists_changed_files_and_excerpt(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "add subtract")
    head = run_git(git_repo, "rev-parse", "HEAD")

    summary = collect_change_summary(git_repo, base, head)

    assert summary.repository_path == str(git_repo)
    assert summary.base_revision == base
    assert summary.head_revision == head
    assert summary.changed_files == ["app.py"]
    assert "subtract" in "\n".join(summary.diff_excerpt)


def test_collect_change_summary_rejects_missing_repo(tmp_path: Path):
    missing = tmp_path / "missing"

    try:
        collect_change_summary(missing, "main", "HEAD")
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")

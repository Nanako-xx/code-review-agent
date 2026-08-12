from pathlib import Path

from conftest import run_git
from review_agent.git_repo import (
    change_summary_from_dict,
    change_summary_to_dict,
    collect_change_summary,
    collect_complete_diff_bytes,
    complete_diff_command,
)


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
    assert summary.file_changes[0].path == "app.py"
    assert summary.file_changes[0].status == "M"
    assert "subtract" in "\n".join(summary.file_diff_excerpts["app.py"])
    assert summary.diff_truncated is False
    assert change_summary_from_dict(change_summary_to_dict(summary)) == summary


def test_collect_change_summary_marks_global_excerpt_truncation(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "\n".join(f"value_{index} = {index}" for index in range(20)) + "\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "expand app")
    head = run_git(git_repo, "rev-parse", "HEAD")

    summary = collect_change_summary(
        git_repo,
        base,
        head,
        max_excerpt_lines=3,
        max_file_excerpt_lines=4,
    )

    assert summary.diff_truncated is True
    assert len(summary.diff_excerpt) == 3
    assert len(summary.file_diff_excerpts["app.py"]) == 4


def test_collect_change_summary_rejects_missing_repo(tmp_path: Path):
    missing = tmp_path / "missing"

    try:
        collect_change_summary(missing, "main", "HEAD")
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_complete_diff_command_fixes_security_and_reproducibility_options(
    git_repo: Path,
) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("changed\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change")
    head = run_git(git_repo, "rev-parse", "HEAD")

    command = complete_diff_command(base, head)
    patch = collect_complete_diff_bytes(git_repo, base, head)

    assert command[0] == "git"
    assert ("-c", "core.longpaths=true") in tuple(
        zip(command, command[1:])
    )
    assert "--no-ext-diff" in command
    assert "--no-textconv" in command
    assert "--binary" in command
    assert "--full-index" in command
    assert "--find-renames" in command
    assert "--find-copies-harder" in command
    assert "--src-prefix=a/" in command
    assert "--dst-prefix=b/" in command
    assert command[-3:] == [base, head, "--"]
    assert patch.startswith(b"diff --git ")

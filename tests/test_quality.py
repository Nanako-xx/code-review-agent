from pathlib import Path

from conftest import run_git
from review_agent.quality import detect_quality_gates, run_python_compile_gate


def test_detect_python_compile_for_python_repo(tmp_path: Path):
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")

    gates = detect_quality_gates(tmp_path)

    assert gates == ["python_compile"]


def test_python_compile_gate_passes_for_valid_python(tmp_path: Path):
    (tmp_path / "app.py").write_text("def ok():\n    return 1\n", encoding="utf-8")

    result = run_python_compile_gate(tmp_path)

    assert result.name == "python_compile"
    assert result.status == "passed"


def test_python_compile_gate_passes_for_utf8_bom_python(tmp_path: Path):
    (tmp_path / "app.py").write_text("\ufeffdef ok():\n    return 1\n", encoding="utf-8")

    result = run_python_compile_gate(tmp_path)

    assert result.status == "passed"


def test_python_compile_gate_fails_for_invalid_python(tmp_path: Path):
    (tmp_path / "bad.py").write_text("def broken(:\n    return 1\n", encoding="utf-8")

    result = run_python_compile_gate(tmp_path)

    assert result.status == "failed"
    assert "SyntaxError" in result.summary


def test_revision_python_compile_ignores_dirty_worktree(git_repo: Path):
    head = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def broken(:\n", encoding="utf-8")

    gates = detect_quality_gates(git_repo, revision=head)
    result = run_python_compile_gate(git_repo, revision=head)

    assert gates == ["python_compile"]
    assert result.status == "passed"
    assert "Compiled 1 Python files" in result.summary


def test_revision_python_compile_reads_non_checked_out_commit(git_repo: Path):
    (git_repo / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    run_git(git_repo, "add", "bad.py")
    run_git(git_repo, "commit", "-m", "add invalid python target")
    target_head = run_git(git_repo, "rev-parse", "HEAD")

    (git_repo / "bad.py").write_text("def fixed():\n    return 1\n", encoding="utf-8")
    run_git(git_repo, "add", "bad.py")
    run_git(git_repo, "commit", "-m", "fix python on current head")

    result = run_python_compile_gate(git_repo, revision=target_head)

    assert result.status == "failed"
    assert "bad.py" in result.summary

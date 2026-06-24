from pathlib import Path
import subprocess

import pytest


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "review-agent@example.test")
    run_git(repo, "config", "user.name", "Review Agent")
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    run_git(repo, "add", "app.py")
    run_git(repo, "commit", "-m", "initial")
    return repo

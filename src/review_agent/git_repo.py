from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class ChangeSummary:
    repository_path: str
    base_revision: str
    head_revision: str
    changed_files: list[str]
    diff_stat: str
    diff_excerpt: list[str]


def _run_git(repo: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def collect_change_summary(
    repo_path: Path,
    base_revision: str,
    head_revision: str,
    max_excerpt_lines: int = 120,
) -> ChangeSummary:
    repo = repo_path.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo}")
    if not (repo / ".git").exists():
        raise ValueError(f"Repository path is not a Git repository: {repo}")

    revision_range = f"{base_revision}..{head_revision}"
    changed_files = [
        line.strip()
        for line in _run_git(repo, ["diff", "--name-only", revision_range]).splitlines()
        if line.strip()
    ]
    diff_stat = _run_git(repo, ["diff", "--stat", revision_range]).strip()
    diff_lines = _run_git(repo, ["diff", "--unified=3", revision_range]).splitlines()
    diff_excerpt = diff_lines[:max_excerpt_lines]

    return ChangeSummary(
        repository_path=str(repo),
        base_revision=base_revision,
        head_revision=head_revision,
        changed_files=changed_files,
        diff_stat=diff_stat,
        diff_excerpt=diff_excerpt,
    )

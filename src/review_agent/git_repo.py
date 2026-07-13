from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Any, Mapping


@dataclass(frozen=True)
class ChangeSummary:
    repository_path: str
    base_revision: str
    head_revision: str
    changed_files: list[str]
    diff_stat: str
    diff_excerpt: list[str]
    file_changes: list["ChangedFile"] = field(default_factory=list)
    file_diff_excerpts: dict[str, list[str]] = field(default_factory=dict)
    diff_truncated: bool = False


@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: str
    previous_path: str | None = None


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
    max_file_excerpt_lines: int = 80,
) -> ChangeSummary:
    repo = repo_path.resolve()
    if not repo.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo}")
    if not (repo / ".git").exists():
        raise ValueError(f"Repository path is not a Git repository: {repo}")

    revision_range = f"{base_revision}..{head_revision}"
    file_changes = _parse_name_status(
        _run_git(repo, ["diff", "--name-status", "-M", revision_range])
    )
    changed_files = [item.path for item in file_changes]
    diff_stat = _run_git(repo, ["diff", "--stat", revision_range]).strip()
    diff_lines = _run_git(repo, ["diff", "--unified=3", revision_range]).splitlines()
    diff_excerpt = diff_lines[:max_excerpt_lines]
    file_diff_excerpts = {
        item.path: _run_git(
            repo,
            ["diff", "--unified=3", revision_range, "--", item.path],
        ).splitlines()[:max_file_excerpt_lines]
        for item in file_changes
    }

    return ChangeSummary(
        repository_path=str(repo),
        base_revision=base_revision,
        head_revision=head_revision,
        changed_files=changed_files,
        diff_stat=diff_stat,
        diff_excerpt=diff_excerpt,
        file_changes=file_changes,
        file_diff_excerpts=file_diff_excerpts,
        diff_truncated=len(diff_lines) > max_excerpt_lines,
    )


def _parse_name_status(raw: str) -> list[ChangedFile]:
    changes: list[ChangedFile] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status_token = parts[0]
        status = status_token[:1]
        if status in {"R", "C"}:
            if len(parts) != 3:
                raise ValueError(f"invalid Git rename/copy status row: {line}")
            changes.append(
                ChangedFile(
                    path=parts[2],
                    status=status,
                    previous_path=parts[1],
                )
            )
            continue
        if len(parts) != 2:
            raise ValueError(f"invalid Git name-status row: {line}")
        changes.append(ChangedFile(path=parts[1], status=status))
    return changes


def change_summary_to_dict(summary: ChangeSummary) -> dict[str, Any]:
    return {
        "repository_path": summary.repository_path,
        "base_revision": summary.base_revision,
        "head_revision": summary.head_revision,
        "changed_files": list(summary.changed_files),
        "diff_stat": summary.diff_stat,
        "diff_excerpt": list(summary.diff_excerpt),
        "file_changes": [
            {
                "path": item.path,
                "status": item.status,
                "previous_path": item.previous_path,
            }
            for item in summary.file_changes
        ],
        "file_diff_excerpts": {
            path: list(lines) for path, lines in summary.file_diff_excerpts.items()
        },
        "diff_truncated": summary.diff_truncated,
    }


def change_summary_from_dict(payload: Mapping[str, Any]) -> ChangeSummary:
    expected = {
        "repository_path",
        "base_revision",
        "head_revision",
        "changed_files",
        "diff_stat",
        "diff_excerpt",
        "file_changes",
        "file_diff_excerpts",
        "diff_truncated",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("change_summary must contain exactly the v1 fields")
    string_fields = (
        "repository_path",
        "base_revision",
        "head_revision",
        "diff_stat",
    )
    for name in string_fields:
        if not isinstance(payload[name], str):
            raise ValueError(f"change_summary.{name} must be a string")
    changed_files = _string_list(payload["changed_files"], "changed_files")
    diff_excerpt = _string_list(payload["diff_excerpt"], "diff_excerpt")
    rows = payload["file_changes"]
    if not isinstance(rows, list):
        raise ValueError("change_summary.file_changes must be a list")
    file_changes: list[ChangedFile] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "status",
            "previous_path",
        }:
            raise ValueError(
                f"change_summary.file_changes[{index}] must contain exactly path, status, previous_path"
            )
        path = row["path"]
        status = row["status"]
        previous_path = row["previous_path"]
        if not isinstance(path, str) or not path:
            raise ValueError(f"change_summary.file_changes[{index}].path must be non-empty")
        if not isinstance(status, str) or len(status) != 1:
            raise ValueError(f"change_summary.file_changes[{index}].status must be one character")
        if previous_path is not None and (
            not isinstance(previous_path, str) or not previous_path
        ):
            raise ValueError(
                f"change_summary.file_changes[{index}].previous_path must be non-empty or null"
            )
        file_changes.append(ChangedFile(path, status, previous_path))
    excerpts_payload = payload["file_diff_excerpts"]
    if not isinstance(excerpts_payload, Mapping):
        raise ValueError("change_summary.file_diff_excerpts must be an object")
    file_diff_excerpts: dict[str, list[str]] = {}
    for path, lines in excerpts_payload.items():
        if not isinstance(path, str) or not path:
            raise ValueError("change_summary.file_diff_excerpts keys must be non-empty strings")
        file_diff_excerpts[path] = _string_list(
            lines,
            f"file_diff_excerpts.{path}",
        )
    diff_truncated = payload["diff_truncated"]
    if type(diff_truncated) is not bool:
        raise ValueError("change_summary.diff_truncated must be a boolean")
    return ChangeSummary(
        repository_path=payload["repository_path"],
        base_revision=payload["base_revision"],
        head_revision=payload["head_revision"],
        changed_files=changed_files,
        diff_stat=payload["diff_stat"],
        diff_excerpt=diff_excerpt,
        file_changes=file_changes,
        file_diff_excerpts=file_diff_excerpts,
        diff_truncated=diff_truncated,
    )


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"change_summary.{field_name} must be a list of strings")
    return list(value)

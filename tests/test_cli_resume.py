from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import pytest

from review_agent.cli import main


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _commit(repo: Path) -> tuple[str, str]:
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "auth.py").write_text(
        "def check(token):\n    return bool(token)\n",
        encoding="utf-8",
    )
    _git(repo, "add", "auth.py")
    _git(repo, "commit", "-m", "add auth check")
    return base, _git(repo, "rev-parse", "HEAD")


def _review_args(repo: Path, root: Path, base: str, head: str) -> list[str]:
    return [
        "review",
        "--repo",
        str(repo),
        "--base",
        base,
        "--head",
        head,
        "--external-review-id",
        "resume-pr",
        "--workspace-root",
        str(root),
        "--intent",
        "Review the authentication behavior change.",
        "--reviewer-provider",
        "fake",
        "--format",
        "json",
    ]


def _locator(stderr: str) -> dict[str, str]:
    match = re.search(r"^Review locator: (\{.*\})$", stderr, re.MULTILINE)
    assert match is not None, stderr
    return json.loads(match.group(1))


def _resume_args(
    repo: Path,
    root: Path,
    locator: dict[str, str],
    *,
    provider: str = "none",
) -> list[str]:
    return [
        "resume",
        locator["session_id"],
        "--repo",
        str(repo),
        "--workspace-root",
        str(root),
        "--pr-id",
        locator["pr_id"],
        "--snapshot-id",
        locator["snapshot_id"],
        "--reviewer-provider",
        provider,
        "--format",
        "json",
    ]


def _snapshot(root: Path) -> Path:
    values = list(root.glob("pr/p-*/Snapshots/s-*"))
    assert len(values) == 1
    return values[0]


def _session(root: Path) -> Path:
    values = list(root.glob("pr/p-*/Sessions/u-*"))
    assert len(values) == 1
    return values[0]


def test_cli_resume_completed_v6_result_is_byte_stable(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _commit(git_repo)
    root = tmp_path / "workspace"
    assert main(_review_args(git_repo, root, base, head)) == 0
    first = capsys.readouterr()
    locator = _locator(first.err)

    assert main(_resume_args(git_repo, root, locator)) == 0

    resumed = capsys.readouterr()
    assert resumed.out == first.out
    assert _locator(resumed.err) == locator


def test_cli_resume_rebuilds_missing_pure_markdown(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _commit(git_repo)
    root = tmp_path / "workspace"
    assert main(_review_args(git_repo, root, base, head)) == 0
    locator = _locator(capsys.readouterr().err)
    markdown = _snapshot(root) / "Results" / "review.md"
    expected = markdown.read_bytes()
    markdown.unlink()

    assert main(_resume_args(git_repo, root, locator)) == 0

    capsys.readouterr()
    assert markdown.read_bytes() == expected


def test_cli_resume_rejects_tampered_authoritative_result(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _commit(git_repo)
    root = tmp_path / "workspace"
    assert main(_review_args(git_repo, root, base, head)) == 0
    locator = _locator(capsys.readouterr().err)
    result_path = _snapshot(root) / "Results" / "review-result.json"
    payload = json.loads(result_path.read_text("utf-8"))
    payload["uncertainties"].append("tampered")
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(_resume_args(git_repo, root, locator)) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "integrity" in captured.err.casefold() or "configuration" in captured.err.casefold()


def test_cli_resume_restarts_failed_preflight_without_new_session(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    base, head = _commit(git_repo)
    root = tmp_path / "workspace"
    from review_agent.product_runtime import DeterministicPreflight

    original = DeterministicPreflight.run

    def fail_once(*_args, **_kwargs):
        raise RuntimeError("transient preflight failure")

    monkeypatch.setattr(DeterministicPreflight, "run", fail_once)
    assert main(_review_args(git_repo, root, base, head)) == 1
    locator = _locator(capsys.readouterr().err)
    monkeypatch.setattr(DeterministicPreflight, "run", original)

    assert main(_resume_args(git_repo, root, locator, provider="fake")) == 0

    result = json.loads(capsys.readouterr().out)
    state = json.loads((_session(root) / "pipeline-state.json").read_text("utf-8"))
    assert result["status"] == "completed"
    assert state["status"] == "completed"
    assert state["phases"]["preflight"]["attempt"] == 2
    assert len(list(root.glob("pr/p-*/Sessions/u-*"))) == 1


def test_cli_resume_rejects_wrong_repository_identity(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _commit(git_repo)
    root = tmp_path / "workspace"
    assert main(_review_args(git_repo, root, base, head)) == 0
    locator = _locator(capsys.readouterr().err)
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init")
    _git(other, "config", "user.email", "other@example.test")
    _git(other, "config", "user.name", "Other")
    (other / "x.py").write_text("x = 1\n", encoding="utf-8")
    _git(other, "add", "x.py")
    _git(other, "commit", "-m", "initial")

    assert main(_resume_args(other, root, locator)) == 2

    assert "repository identity" in capsys.readouterr().err


def test_cli_legacy_review_is_inspect_only_and_never_resumed(
    git_repo: Path,
    capsys,
) -> None:
    review_id = "review-legacy"
    run_dir = git_repo / ".review-agent" / "runs" / review_id
    run_dir.mkdir(parents=True)
    state = {"review_id": review_id, "status": "running"}
    (run_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")

    assert main(["resume", review_id, "--repo", str(git_repo)]) == 2

    captured = capsys.readouterr()
    assert "inspect-only" in captured.err
    assert "no legacy Phase was run" in captured.err
    assert json.loads((run_dir / "state.json").read_text("utf-8")) == state


def test_cli_resume_missing_session_or_locator_returns_two(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    missing = {
        "pr_id": "PR-" + "a" * 64,
        "snapshot_id": "S-" + "b" * 64,
        "session_id": "SESSION-" + "c" * 64,
    }

    assert main(_resume_args(git_repo, tmp_path / "missing", missing)) == 2

    assert "required regular file does not exist" in capsys.readouterr().err


def test_cli_resume_rejects_legacy_upgrade_flag(
    git_repo: Path,
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "resume",
                "review-legacy",
                "--repo",
                str(git_repo),
                "--upgrade-to-v5",
            ]
        )

    assert raised.value.code == 2

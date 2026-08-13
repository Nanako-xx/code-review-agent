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


def _commit_change(repo: Path, content: str = "value = 2\n") -> tuple[str, str]:
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "app.py").write_text(content, encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "change app")
    return base, _git(repo, "rev-parse", "HEAD")


def _args(
    repo: Path,
    root: Path,
    base: str,
    head: str,
    *,
    external_id: str = "smoke-pr",
    provider: str = "fake",
    intent: str | None = "Review the app behavior change.",
) -> list[str]:
    values = [
        "review",
        "--repo",
        str(repo),
        "--base",
        base,
        "--head",
        head,
        "--external-review-id",
        external_id,
        "--workspace-root",
        str(root),
        "--reviewer-provider",
        provider,
        "--format",
        "json",
    ]
    if intent is not None:
        values.extend(("--intent", intent))
    return values


def _locator(stderr: str) -> dict[str, str]:
    match = re.search(r"^Review locator: (\{.*\})$", stderr, re.MULTILINE)
    assert match is not None, stderr
    return json.loads(match.group(1))


def _snapshot(root: Path) -> Path:
    values = list(root.glob("pr/p-*/Snapshots/s-*"))
    assert len(values) == 1
    return values[0]


def _session(root: Path) -> Path:
    values = list(root.glob("pr/p-*/Sessions/u-*"))
    assert len(values) == 1
    return values[0]


def test_cli_review_writes_only_v6_product_artifacts(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _commit_change(git_repo)
    root = tmp_path / "workspace"

    assert main(_args(git_repo, root, base, head)) == 0

    output = capsys.readouterr()
    result = json.loads(output.out)
    locator = _locator(output.err)
    snapshot = _snapshot(root)
    session = _session(root)
    assert result["pr_id"] == locator["pr_id"]
    assert result["snapshot_id"] == locator["snapshot_id"]
    assert (snapshot / "DiffArtifact" / "diff.patch").is_file()
    assert (snapshot / "DiffArtifact" / "index.json").is_file()
    assert (snapshot / "QualityGate" / "quality-gate.json").is_file()
    assert (snapshot / "ChangedSymbols" / "changed-symbols.json").is_file()
    assert list((snapshot / "Intent").glob("intent-*.json"))
    assert (snapshot / "Risk" / "risk.json").is_file()
    assert (snapshot / "ReviewPlan" / "plan.json").is_file()
    assert list((snapshot / "Results" / "reviewers").glob("r-*.json"))
    assert (snapshot / "Results" / "aggregation.json").is_file()
    assert (snapshot / "Results" / "review-result.json").is_file()
    assert (snapshot / "Results" / "review.md").is_file()
    state = json.loads((session / "pipeline-state.json").read_text("utf-8"))
    assert set(state["phases"]) == {
        "preflight",
        "intent",
        "planning",
        "reviewers",
        "aggregation",
    }
    assert state["status"] == "completed"
    assert not (snapshot / "completion.json").exists()
    assert not (snapshot / "review_brief.json").exists()
    assert not (git_repo / ".review-agent" / "runs").exists()


def test_cli_does_not_persist_api_secret(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    base, head = _commit_change(git_repo)
    root = tmp_path / "workspace"
    secret = "secret-that-must-never-be-written-2c40d6"
    monkeypatch.setenv("REVIEW_AGENT_API_KEY", secret)

    assert main(_args(git_repo, root, base, head)) == 0
    capsys.readouterr()

    for path in root.rglob("*"):
        if path.is_file():
            assert secret.encode("utf-8") not in path.read_bytes()


def test_cli_openai_configuration_error_returns_two_without_result(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    base, head = _commit_change(git_repo)
    root = tmp_path / "workspace"
    monkeypatch.delenv("MISSING_REVIEW_KEY", raising=False)
    args = _args(git_repo, root, base, head, provider="openai-compatible")
    args.extend(
        (
            "--reviewer-model",
            "review-model",
            "--reviewer-base-url",
            "https://reviewer.invalid/v1",
            "--reviewer-api-key-env",
            "MISSING_REVIEW_KEY",
        )
    )

    assert main(args) == 2

    captured = capsys.readouterr()
    assert "missing API key environment variable" in captured.err
    assert not list(root.glob("pr/p-*/Snapshots/s-*/Results/review-result.json"))


def test_cli_invalid_revision_does_not_create_workspace(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "workspace"

    exit_code = main(
        _args(git_repo, root, "missing-base", "HEAD", provider="none")
    )

    assert exit_code == 2
    assert "does not resolve" in capsys.readouterr().err.casefold()
    assert not root.exists()


def test_cli_preflight_failure_returns_one_and_keeps_resumable_locator(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    base, head = _commit_change(git_repo)
    root = tmp_path / "workspace"

    def fail_preflight(*_args, **_kwargs):
        raise RuntimeError("private preflight detail")

    monkeypatch.setattr(
        "review_agent.product_runtime.DeterministicPreflight.run",
        fail_preflight,
    )
    assert main(_args(git_repo, root, base, head)) == 1

    captured = capsys.readouterr()
    locator = _locator(captured.err)
    assert "private preflight detail" not in captured.err
    state = json.loads(
        (_session(root) / "pipeline-state.json").read_text("utf-8")
    )
    assert state["status"] == "failed"
    assert state["current_phase"] == "preflight"
    assert state["phases"]["preflight"]["error_code"] == "preflight_failed"
    assert locator["session_id"].startswith("SESSION-")
    assert not (_snapshot(root) / "Results" / "review-result.json").exists()


def test_cli_diff_uses_committed_head_not_dirty_worktree(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _commit_change(git_repo, "value = 'committed'\n")
    (git_repo / "app.py").write_text("value = 'dirty'\n", encoding="utf-8")
    root = tmp_path / "workspace"

    assert main(_args(git_repo, root, base, head)) == 0
    capsys.readouterr()

    patch = (_snapshot(root) / "DiffArtifact" / "diff.patch").read_text("utf-8")
    assert "committed" in patch
    assert "dirty" not in patch


def test_cli_fake_reviewer_handles_empty_diff(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    revision = _git(git_repo, "rev-parse", "HEAD")
    root = tmp_path / "workspace"

    assert main(_args(git_repo, root, revision, revision)) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert result["findings"] == []
    assert (_snapshot(root) / "DiffArtifact" / "diff.patch").read_bytes() == b""


def test_missing_explicit_intent_runs_agent_and_uses_inferred_risk_slots(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _commit_change(git_repo)
    root = tmp_path / "workspace"

    assert main(_args(git_repo, root, base, head, intent=None)) == 0

    result = json.loads(capsys.readouterr().out)
    reviewer_records = list(
        (_snapshot(root) / "Results" / "reviewers").glob("r-*.json")
    )
    intent_paths = list(root.glob("pr/p-*/Intent/current.json"))
    assert len(intent_paths) == 1
    intent_packet = json.loads(intent_paths[0].read_text("utf-8"))["packet"]
    assert intent_packet["source"] == "inferred"
    assert result["risk_level"] == "medium"
    assert len(reviewer_records) == 2


@pytest.mark.parametrize(
    "removed_option",
    (
        "--semantic-reconciler-mode=model",
        "--memory-curator-mode=model",
        "--portfolio-planner-mode=model",
        "--memory-mode=read",
        "--reviewer-mode=multi",
        "--reviewer-loop=agent-loop",
        "--risk-assessor-max-output-tokens=1024",
    ),
)
def test_cli_rejects_removed_product_stage_options(
    git_repo: Path,
    tmp_path: Path,
    removed_option: str,
) -> None:
    revision = _git(git_repo, "rev-parse", "HEAD")
    args = _args(git_repo, tmp_path / "workspace", revision, revision)
    args.append(removed_option)

    with pytest.raises(SystemExit) as raised:
        main(args)

    assert raised.value.code == 2

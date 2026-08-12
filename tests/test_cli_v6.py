from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

from review_agent.cli import main
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse


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


def _revisions(repo: Path) -> tuple[str, str]:
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "app.py").write_text(
        "def add(a, b):\n    value = a + b\n    return value\n",
        encoding="utf-8",
    )
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "change add")
    return base, _git(repo, "rev-parse", "HEAD")


def _locator(stderr: str) -> dict[str, str]:
    match = re.search(r"^Review locator: (?P<value>\{.*\})$", stderr, re.MULTILINE)
    assert match is not None, stderr
    value = json.loads(match.group("value"))
    assert set(value) == {"pr_id", "snapshot_id", "session_id"}
    return value


def _review_args(
    repo: Path,
    workspace_root: Path,
    base: str,
    head: str,
    *,
    provider: str = "fake",
    output_format: str = "json",
) -> list[str]:
    return [
        "review",
        "--repo",
        str(repo),
        "--base",
        base,
        "--head",
        head,
        "--external-review-id",
        "local-pr-17",
        "--workspace-root",
        str(workspace_root),
        "--intent",
        "Preserve add() behavior while making the intermediate value explicit.",
        "--reviewer-provider",
        provider,
        "--format",
        output_format,
    ]


def test_cli_v6_fake_review_returns_only_authoritative_json(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _revisions(git_repo)
    workspace_root = tmp_path / "workspace"

    exit_code = main(_review_args(git_repo, workspace_root, base, head))

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert set(payload) == {
        "pr_id",
        "snapshot_id",
        "status",
        "risk_level",
        "findings",
        "uncertainties",
    }
    assert payload["status"] == "completed"
    assert payload["findings"] == []
    locator = _locator(captured.err)
    assert locator["pr_id"] == payload["pr_id"]
    assert locator["snapshot_id"] == payload["snapshot_id"]
    assert not (git_repo / ".review-agent" / "runs").exists()
    assert list(workspace_root.glob("pr/p-*/Snapshots/s-*/Results/review-result.json"))
    assert list(workspace_root.glob("pr/p-*/Sessions/u-*/pipeline-state.json"))


def test_cli_v6_injects_only_assignment_rules_and_runs_no_quality_command(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    base, head = _revisions(git_repo)
    adapter = FakeToolCallingAdapter(
        [
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(
                    {
                        "findings": [],
                        "uncertainties": ["Recording-only reviewer."],
                    }
                ),
            )
        ]
    )

    class RecordingFactory:
        def create(self):
            return adapter

    def recording_factory(_config, *, stage_label):
        assert stage_label == "reviewer"
        return RecordingFactory()

    def quality_process_must_not_run(*_args, **_kwargs):
        raise AssertionError("empty LocalQualityPlan must not execute a process")

    monkeypatch.setattr(
        "review_agent.product_runtime.build_model_adapter_factory_from_config",
        recording_factory,
    )
    monkeypatch.setattr(
        "review_agent.local_quality.SubprocessQualityExecutor.run",
        quality_process_must_not_run,
    )

    exit_code = main(
        _review_args(
            git_repo,
            tmp_path / "workspace-rules",
            base,
            head,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert '<BuiltInReviewRule name="python.md">' in request.system
    assert "Mutable Default Arguments" in request.system
    assert "Go Review Principles" not in request.system
    assert "github_workflows.md" not in request.system
    assert "no_configured_checks" in request.messages[0]["content"]


def test_cli_v6_markdown_is_a_pure_review_result_render(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _revisions(git_repo)

    exit_code = main(
        _review_args(
            git_repo,
            tmp_path / "workspace",
            base,
            head,
            output_format="markdown",
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("# Code Review\n")
    assert "Recommendation" not in captured.out
    assert "Review brief" not in captured.out
    _locator(captured.err)


def test_cli_v6_authoritative_failed_result_still_returns_zero(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _revisions(git_repo)

    exit_code = main(
        _review_args(
            git_repo,
            tmp_path / "workspace",
            base,
            head,
            provider="none",
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "failed"
    _locator(captured.err)


def test_cli_v6_resume_reuses_result_without_invoking_a_provider(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    base, head = _revisions(git_repo)
    workspace_root = tmp_path / "workspace"
    assert main(_review_args(git_repo, workspace_root, base, head)) == 0
    first = capsys.readouterr()
    locator = _locator(first.err)

    def provider_must_not_be_built(*_args, **_kwargs):
        raise AssertionError("completed Resume must not build a provider")

    monkeypatch.setattr(
        "review_agent.product_runtime.build_model_adapter_factory_from_config",
        provider_must_not_be_built,
    )
    exit_code = main(
        [
            "resume",
            locator["session_id"],
            "--repo",
            str(git_repo),
            "--workspace-root",
            str(workspace_root),
            "--pr-id",
            locator["pr_id"],
            "--snapshot-id",
            locator["snapshot_id"],
            "--format",
            "json",
        ]
    )

    resumed = capsys.readouterr()
    assert exit_code == 0
    assert resumed.out == first.out
    assert _locator(resumed.err) == locator


def test_cli_v6_rejects_missing_external_review_identity_before_writing(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _revisions(git_repo)
    root = tmp_path / "workspace"

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--workspace-root",
            str(root),
        ]
    )

    assert exit_code == 2
    assert "--external-review-id is required" in capsys.readouterr().err
    assert not root.exists()

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import pytest

from review_agent.cli import main
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnResponse,
)


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


def _without_declared_intent(arguments: list[str]) -> list[str]:
    values = list(arguments)
    index = values.index("--intent")
    del values[index : index + 2]
    return values


def _intent_records(workspace_root: Path) -> tuple[dict, dict, dict]:
    current_paths = list(workspace_root.glob("pr/p-*/Intent/current.json"))
    analysis_paths = list(
        workspace_root.glob("pr/p-*/Intent/history/analysis-*.json")
    )
    request_paths = list(
        workspace_root.glob("pr/p-*/Snapshots/s-*/Requests/request-*.json")
    )
    assert len(current_paths) == len(analysis_paths) == len(request_paths) == 1
    return tuple(
        json.loads(path.read_text("utf-8"))
        for path in (current_paths[0], analysis_paths[0], request_paths[0])
    )


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


def test_cli_v6_missing_explicit_intent_must_run_intent_agent_and_stay_inferred(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _revisions(git_repo)
    workspace_root = tmp_path / "workspace-inferred-intent"
    arguments = _without_declared_intent(
        _review_args(git_repo, workspace_root, base, head)
    )
    arguments.append("--non-interactive")

    assert main(arguments) == 0

    output = json.loads(capsys.readouterr().out)
    current, analysis, request = _intent_records(workspace_root)
    assert output["risk_level"] == "medium"
    assert current["packet"]["source"] == "inferred"
    assert analysis["inference_run"]["status"] == "completed"
    assert analysis["trust_policy"] == "normal"
    assert analysis["model_inference_promoted"] is False
    assert request["intent_trust_policy"] == "normal"


def test_cli_v6_intent_agent_tools_use_v6_tool_results_without_observation_store(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    base, head = _revisions(git_repo)
    workspace_root = tmp_path / "workspace-intent-tools"

    def finish_intent(request):
        evidence_id = request.tool_results[-1].observation_ids[0]
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "candidates": [
                        {
                            "field": "goal",
                            "value": "Make the add implementation explicit.",
                            "origin": "commit_message",
                            "confidence": "high",
                            "source_refs": ["change add"],
                            "evidence_refs": [evidence_id],
                            "rationale": "The commit subject states the change.",
                            "conclusion_impact": "material",
                        }
                    ],
                    "uncertainties": [],
                    "summary": "The commit message supplied one review goal.",
                }
            ),
            provider_name="fake",
            model="fake-intent-analyst",
        )

    intent_adapter = FakeToolCallingAdapter(
        [
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[
                    ModelToolCall(
                        "intent-commit-1",
                        "read_commit_messages",
                        {"max_commits": 1},
                    )
                ],
                provider_name="fake",
                model="fake-intent-analyst",
            ),
            finish_intent,
        ]
    )

    def reviewer_response(_request):
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text='{"findings":[],"uncertainties":[]}',
            provider_name="fake",
            model="fake-reviewer-v2",
        )

    class RecordingFactory:
        def __init__(self):
            self.created = 0

        def create(self):
            self.created += 1
            if self.created == 1:
                return intent_adapter
            return FakeToolCallingAdapter([reviewer_response])

    factory = RecordingFactory()
    monkeypatch.setattr(
        "review_agent.product_runtime.build_model_adapter_factory_from_config",
        lambda _config, *, stage_label: factory,
    )

    arguments = _without_declared_intent(
        _review_args(git_repo, workspace_root, base, head)
    )
    assert main(arguments) == 0

    capsys.readouterr()
    current, analysis, _request = _intent_records(workspace_root)
    tool_indices = list(
        workspace_root.glob("pr/p-*/Snapshots/s-*/ToolResults/index.jsonl")
    )
    assert current["packet"]["source"] == "inferred"
    assert analysis["inference_run"]["trace"]["tool_call_count"] == 1
    assert len(tool_indices) == 1
    records = [
        json.loads(line)
        for line in tool_indices[0].read_text("utf-8").splitlines()
    ]
    assert records[0]["tool_name"] == "read_commit_messages"
    assert records[0]["status"] == "completed"
    assert not list(workspace_root.glob("**/observations.jsonl"))
    assert not list(workspace_root.glob("**/observations"))


def test_cli_v6_evaluation_policy_promotes_reliable_model_intent_to_explicit(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _revisions(git_repo)
    workspace_root = tmp_path / "workspace-evaluation-intent"
    arguments = _without_declared_intent(
        _review_args(git_repo, workspace_root, base, head)
    )
    arguments.append("--evaluation-trust-model-intent")

    assert main(arguments) == 0

    output = json.loads(capsys.readouterr().out)
    current, analysis, request = _intent_records(workspace_root)
    assert output["risk_level"] == "low"
    assert current["packet"]["source"] == "explicit"
    assert analysis["inference_run"]["status"] == "completed"
    assert analysis["trust_policy"] == "evaluation_trust_model"
    assert analysis["model_inference_promoted"] is True
    assert request["intent_trust_policy"] == "evaluation_trust_model"


def test_cli_v6_explicit_input_skips_intent_agent_even_in_evaluation_mode(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    base, head = _revisions(git_repo)
    workspace_root = tmp_path / "workspace-explicit-intent"

    def inference_must_not_run(*_args, **_kwargs):
        raise AssertionError("explicit Intent must bypass Intent Agent")

    monkeypatch.setattr(
        "review_agent.product_runtime.run_intent_inference",
        inference_must_not_run,
    )
    arguments = _review_args(git_repo, workspace_root, base, head)
    arguments.append("--evaluation-trust-model-intent")

    assert main(arguments) == 0

    capsys.readouterr()
    current, analysis, request = _intent_records(workspace_root)
    assert current["packet"]["source"] == "explicit"
    assert analysis["inference_run"] is None
    assert analysis["model_inference_promoted"] is False
    assert request["intent_trust_policy"] == "evaluation_trust_model"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--title", "Make add() implementation explicit"),
        ("--description", "Preserve add behavior while naming the intermediate value."),
    ],
)
def test_cli_v6_pr_metadata_is_explicit_and_skips_intent_agent(
    git_repo: Path,
    tmp_path: Path,
    capsys,
    monkeypatch,
    option: str,
    value: str,
) -> None:
    base, head = _revisions(git_repo)
    workspace_root = tmp_path / ("workspace-metadata-" + option[2:])

    def inference_must_not_run(*_args, **_kwargs):
        raise AssertionError("PR metadata must bypass Intent Agent")

    monkeypatch.setattr(
        "review_agent.product_runtime.run_intent_inference",
        inference_must_not_run,
    )
    arguments = _without_declared_intent(
        _review_args(git_repo, workspace_root, base, head)
    )
    arguments.extend([option, value])

    assert main(arguments) == 0

    capsys.readouterr()
    current, analysis, _request = _intent_records(workspace_root)
    assert current["packet"] == {
        "goal": value,
        "source": "explicit",
        "uncertainties": [],
    }
    assert analysis["inference_run"] is None


def test_cli_v6_evaluation_policy_never_promotes_failed_intent_agent(
    git_repo: Path,
    tmp_path: Path,
    capsys,
) -> None:
    base, head = _revisions(git_repo)
    workspace_root = tmp_path / "workspace-failed-intent"
    arguments = _without_declared_intent(
        _review_args(
            git_repo,
            workspace_root,
            base,
            head,
            provider="none",
        )
    )
    arguments.append("--evaluation-trust-model-intent")

    assert main(arguments) == 0

    output = json.loads(capsys.readouterr().out)
    current, analysis, request = _intent_records(workspace_root)
    assert output["risk_level"] == "high"
    assert current["packet"]["goal"] is None
    assert current["packet"]["source"] is None
    assert analysis["inference_run"]["status"] == "failed"
    assert analysis["model_inference_promoted"] is False
    assert request["intent_trust_policy"] == "evaluation_trust_model"


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

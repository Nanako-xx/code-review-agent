from pathlib import Path
import json

from conftest import run_git
from review_agent.cli import main


def test_cli_review_writes_current_schema_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return token == 'ok'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--intent",
            "Add auth token check",
            "--focus",
            "regression safety",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_root = git_repo / ".review-agent" / "runs"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "request.json").exists()
    assert (run_dirs[0] / "intent.json").exists()
    assert (run_dirs[0] / "risk.json").exists()
    assert (run_dirs[0] / "assignments.json").exists()
    assert (run_dirs[0] / "report.md").exists()

    intent = json.loads((run_dirs[0] / "intent.json").read_text(encoding="utf-8"))
    risk = json.loads((run_dirs[0] / "risk.json").read_text(encoding="utf-8"))
    assignments = json.loads((run_dirs[0] / "assignments.json").read_text(encoding="utf-8"))

    assert "uncertainties" in intent
    assert "unknowns" not in intent
    assert "signal_refs" in risk
    assert "evidence_refs" not in risk
    assert "initial_context" in assignments["assignments"][0]
    assert "provided_evidence_refs" not in assignments["assignments"][0]


def test_cli_review_with_fake_reviewer_writes_reviewer_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return token == 'ok'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--intent",
            "Add auth token check",
            "--reviewer-provider",
            "fake",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_root = git_repo / ".review-agent" / "runs"
    run_dirs = sorted(run_root.iterdir())
    run_dir = run_dirs[-1]

    assert (run_dir / "reviewer_envelope.json").exists()
    assert (run_dir / "reviewer_raw_response.json").exists()
    assert (run_dir / "reviewer_result.json").exists()

    result = json.loads((run_dir / "reviewer_result.json").read_text(encoding="utf-8"))
    raw = json.loads((run_dir / "reviewer_raw_response.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert result["status"] == "partial"
    assert raw["provider_name"] == "fake"
    assert "## Single Reviewer Result" in report
    assert "Fake reviewer executed." in report


def test_cli_openai_compatible_adapter_requires_api_key(git_repo: Path, monkeypatch, capsys):
    monkeypatch.delenv("REVIEW_AGENT_API_KEY", raising=False)
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--reviewer-provider",
            "openai-compatible",
            "--reviewer-model",
            "review-model",
            "--reviewer-base-url",
            "https://example.test/v1",
            "--non-interactive",
        ]
    )

    assert exit_code == 2
    assert "Reviewer adapter configuration error" in capsys.readouterr().out


def test_cli_agent_loop_openai_compatible_uses_adapter_factory(git_repo: Path, monkeypatch):
    from review_agent.model_adapter import FakeToolCallingAdapter
    from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse

    class ToolThenFinalFactory:
        def create(self):
            class OpenAICompatibleFakeToolCallingAdapter(FakeToolCallingAdapter):
                provider_name = "openai-compatible"

            def final_response(request):
                observation_id = request.tool_results[-1].observation_ids[0]
                return ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=json.dumps(
                        {
                            "contract_assessments": [
                                {
                                    "contract": "regression_safety",
                                    "status": "covered",
                                    "summary": "OpenAI-compatible adapter path used tools.",
                                    "evidence_refs": [observation_id],
                                }
                            ],
                            "confirmed_findings": [],
                            "rejected_hypotheses": [],
                            "uncertainties": [],
                            "observation_refs": [observation_id],
                            "investigation_summary": "OpenAI-compatible agent loop executed.",
                            "status": "completed",
                        }
                    ),
                    provider_name="openai-compatible",
                    model="review-model",
                )

            return OpenAICompatibleFakeToolCallingAdapter(
                script=[
                    ModelTurnResponse(
                        kind=ModelResponseKind.TOOL_CALLS,
                        tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "app.py"})],
                    ),
                    final_response,
                ]
            )

    def fake_build_factory(config):
        assert config.provider_name == "openai-compatible"
        assert config.model == "review-model"
        assert config.base_url == "https://example.test/v1"
        return ToolThenFinalFactory()

    monkeypatch.setattr("review_agent.cli.build_model_adapter_factory_from_config", fake_build_factory, raising=False)
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--reviewer-loop",
            "agent-loop",
            "--reviewer-provider",
            "openai-compatible",
            "--reviewer-model",
            "review-model",
            "--reviewer-base-url",
            "https://example.test/v1",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    trace = json.loads((run_dir / "reviewer_agent_trace.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "reviewer_result.json").read_text(encoding="utf-8"))
    raw = json.loads((run_dir / "reviewer_raw_response.json").read_text(encoding="utf-8"))

    assert trace["tool_call_count"] == 1
    assert result["status"] == "completed"
    assert raw["provider_name"] == "openai-compatible"


def test_cli_multi_reviewer_mode_requires_reviewer_provider(git_repo: Path, capsys):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--reviewer-mode",
            "multi",
            "--non-interactive",
        ]
    )

    assert exit_code == 2
    assert "--reviewer-mode multi requires --reviewer-provider" in capsys.readouterr().out
    assert not (git_repo / ".review-agent").exists()


def test_cli_fake_reviewer_writes_observation_store_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return token == 'ok'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--intent",
            "Add auth token check",
            "--reviewer-provider",
            "fake",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    observation_records = [
        json.loads(line) for line in (run_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert observation_records
    compare_record = next(record for record in observation_records if record["source"] == "git.compare_base_head")
    assert compare_record["path"] == "auth.py"
    assert (run_dir / compare_record["raw_artifact_ref"]).exists()

    envelope = json.loads((run_dir / "reviewer_envelope.json").read_text(encoding="utf-8"))
    assert compare_record["observation_id"] in envelope["messages"][0]["content"]
    assert "## Observations" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_cli_writes_repository_intelligence_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--reviewer-provider",
            "fake",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    payload = json.loads((run_dir / "repository_intelligence.json").read_text(encoding="utf-8"))
    observation_records = [
        json.loads(line) for line in (run_dir / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    envelope = json.loads((run_dir / "reviewer_envelope.json").read_text(encoding="utf-8"))

    assert payload["changed_symbols"][0]["qualified_name"] == "add"
    assert any(record["source"] == "repo_intelligence.snapshot" for record in observation_records)
    assert "## Repository Intelligence" in report
    assert "modified function add app.py:1-2" in report
    assert "Repository Intelligence" in envelope["messages"][0]["content"]


def test_cli_multi_reviewer_mode_writes_per_reviewer_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text(
        "def is_admin(user):\n"
        "    return True\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "change auth")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--intent",
            "Change authorization behavior",
            "--reviewer-provider",
            "fake",
            "--reviewer-mode",
            "multi",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    multi = json.loads((run_dir / "multi_reviewer_result.json").read_text(encoding="utf-8"))
    reconciliation = json.loads((run_dir / "reconciliation.json").read_text(encoding="utf-8"))
    completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert multi["reviewer_count"] >= 2
    assert "canonical_findings" in reconciliation
    assert completion["status"] in {"completed", "completed_with_uncertainties", "blocked"}
    assert {item["role"] for item in multi["executions"]} >= {"Core Reviewer", "Adversarial Reviewer"}
    assert (run_dir / "reviewer_0_envelope.json").exists()
    assert (run_dir / "reviewer_1_envelope.json").exists()
    assert (run_dir / "reviewer_0_raw_response.json").exists()
    assert (run_dir / "reviewer_1_raw_response.json").exists()
    assert (run_dir / "reviewer_0_result.json").exists()
    assert (run_dir / "reviewer_1_result.json").exists()
    assert "## Multi-Reviewer Summary" in report
    assert "## Evidence Reconciliation" in report
    assert "## Completion Status" in report


def test_cli_agent_loop_fake_reviewer_writes_trace_artifact(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--intent",
            "Review arithmetic change",
            "--reviewer-provider",
            "fake",
            "--reviewer-loop",
            "agent-loop",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    trace = json.loads((run_dir / "reviewer_agent_trace.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "reviewer_result.json").read_text(encoding="utf-8"))

    assert trace["tool_call_count"] == 1
    assert trace["turns"][0]["tool_calls"][0]["tool_name"] == "compare_base_head"
    assert result["status"] == "completed"


def test_cli_agent_loop_fake_reviewer_handles_no_changed_files(git_repo: Path):
    base = head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--intent",
            "Review unchanged repository",
            "--reviewer-provider",
            "fake",
            "--reviewer-loop",
            "agent-loop",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    trace = json.loads((run_dir / "reviewer_agent_trace.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "reviewer_result.json").read_text(encoding="utf-8"))

    assert trace["tool_call_count"] == 0
    assert result["status"] == "completed"


def test_cli_multi_agent_loop_writes_per_reviewer_trace_artifacts(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def is_admin(user):\n    return True\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "change auth")
    head = run_git(git_repo, "rev-parse", "HEAD")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            head,
            "--intent",
            "Change authorization behavior",
            "--reviewer-provider",
            "fake",
            "--reviewer-mode",
            "multi",
            "--reviewer-loop",
            "agent-loop",
            "--non-interactive",
        ]
    )

    assert exit_code == 0
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]

    assert (run_dir / "reviewer_0_agent_trace.json").exists()
    assert (run_dir / "reviewer_1_agent_trace.json").exists()
    reviewer_0_trace = json.loads((run_dir / "reviewer_0_agent_trace.json").read_text(encoding="utf-8"))
    reviewer_1_trace = json.loads((run_dir / "reviewer_1_agent_trace.json").read_text(encoding="utf-8"))
    for trace in (reviewer_0_trace, reviewer_1_trace):
        assert trace["tool_call_count"] == 1
        assert trace["final_status"] == "completed"
        assert trace["turns"][0]["tool_calls"][0]["tool_name"] == "compare_base_head"
    assert (run_dir / "multi_reviewer_result.json").exists()
    assert (run_dir / "reconciliation.json").exists()
    assert (run_dir / "completion.json").exists()

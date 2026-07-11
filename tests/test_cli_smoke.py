from pathlib import Path
import hashlib
import json
import re

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


def test_cli_review_writes_state_and_preflight_summary(git_repo: Path, monkeypatch, capsys):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return token == 'ok'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")
    secret = "session-must-not-persist-this-secret"
    monkeypatch.setenv("REVIEW_AGENT_API_KEY", secret)

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--intent",
            "Add auth token check",
            "--reviewer-provider",
            "fake",
            "--non-interactive",
        ]
    )

    output = capsys.readouterr().out
    run_dirs = sorted((git_repo / ".review-agent" / "runs").iterdir())
    state = json.loads((run_dirs[-1] / "state.json").read_text(encoding="utf-8"))
    brief = json.loads((run_dirs[-1] / "review_brief.json").read_text(encoding="utf-8"))
    final_risk = json.loads((run_dirs[-1] / "final_risk.json").read_text(encoding="utf-8"))
    session_text = (run_dirs[-1] / "session.json").read_text(encoding="utf-8")
    session = json.loads(session_text)

    assert exit_code == 0
    assert "Preflight" in output
    assert "Requested Base: HEAD~1" in output
    assert "Requested Head: HEAD" in output
    assert f"Resolved Base: {base}" in output
    assert f"Resolved Head: {head}" in output
    assert "Changed files: 1" in output
    assert "Intent status:" in output
    assert "Run directory:" in output
    assert "Final risk:" in output
    assert "Review brief:" in output
    assert "Review brief JSON:" in output
    assert "Recommendation:" in output
    assert state["status"] == "completed"
    assert state["phase"] == "completed"
    assert state["base_revision"] == "HEAD~1"
    assert state["head_revision"] == "HEAD"
    assert state["resolved_base_revision"] == base
    assert state["resolved_head_revision"] == head
    assert state["artifacts"]["request"] == "request.json"
    assert state["artifacts"]["repository_intelligence"] == "repository_intelligence.json"
    assert state["artifacts"]["report"] == "report.md"
    assert state["artifacts"]["review_brief"] == "review_brief.json"
    assert state["artifacts"]["final_risk"] == "final_risk.json"
    assert final_risk["status"] == "reassessed"
    assert brief["review_id"] == run_dirs[-1].name
    assert brief["base_revision"] == base
    assert brief["head_revision"] == head
    assert brief["change_map_and_repository_impact"]["changed_files"] == ["auth.py"]
    assert brief["initial_and_final_risk_assessment"]["final"]["status"] == "reassessed"
    assert brief["non_binding_recommendation"] == "manual_review"
    assert session["schema_version"] == 1
    assert session["status"] == "completed"
    assert session["current_phase"] == "completed"
    assert session["revisions"] == {
        "requested_base": "HEAD~1",
        "requested_head": "HEAD",
        "resolved_base_sha": base,
        "resolved_head_sha": head,
        "original_base_sha": base,
        "incremental_from_sha": None,
        "change_kind": "initial",
    }
    assert re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", base)
    assert re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head)
    assert all(phase["status"] == "completed" for phase in session["phases"].values())
    assert session["execution"] == {
        "reviewer_provider": "fake",
        "reviewer_model": None,
        "reviewer_base_url": None,
        "reviewer_api_key_env": "REVIEW_AGENT_API_KEY",
        "reviewer_mode": "single",
        "reviewer_loop": "single-shot",
        "non_interactive": True,
    }
    assert secret not in session_text

    binding = f"{base}..{head}"
    expected_artifacts = {
        "request": ("request.json", "review_request_v1", "preflight", None),
        "review_brief": ("review_brief.json", "review_brief_v1", "reporting", binding),
        "report": ("report.md", "review_report_markdown_v1", "reporting", binding),
        "final_risk": ("final_risk.json", "final_risk_assessment_v1", "final_risk", binding),
        "observations": ("observations.jsonl", "observation_log_jsonl_v1", "reporting", binding),
    }
    for artifact_name, (relative_path, schema, phase, revision_binding) in expected_artifacts.items():
        descriptor = session["artifacts"][artifact_name]
        artifact_path = run_dirs[-1] / relative_path
        assert descriptor["path"] == relative_path
        assert descriptor["schema"] == schema
        assert descriptor["phase"] == phase
        assert descriptor["revision_binding"] == revision_binding
        assert descriptor["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()


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
    envelope = json.loads((run_dir / "reviewer_envelope.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert result["status"] == "partial"
    assert raw["provider_name"] == "fake"
    assert envelope["parameters"]["model"] == "fake-reviewer"
    assert envelope["parameters"]["context"]["budget_scope"] == "messages_only"
    assert "## Uncertainties" in report
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
    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    assert session["status"] == "failed"
    assert session["phases"]["preflight"]["status"] == "completed"
    assert session["phases"]["repository_intelligence"]["status"] == "completed"
    assert session["phases"]["reviewers"]["status"] == "failed"


def test_cli_review_records_failed_state_when_collection_fails(git_repo: Path, monkeypatch, capsys):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return token == 'ok'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")

    def raise_error(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr("review_agent.cli.collect_change_summary", raise_error)

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--non-interactive",
        ]
    )

    run_dirs = sorted((git_repo / ".review-agent" / "runs").iterdir())
    state = json.loads((run_dirs[-1] / "state.json").read_text(encoding="utf-8"))
    session = json.loads((run_dirs[-1] / "session.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert state["status"] == "failed"
    assert state["phase"] == "failed"
    assert state["base_revision"] == "HEAD~1"
    assert state["head_revision"] == "HEAD"
    assert state["resolved_base_revision"] == base
    assert state["resolved_head_revision"] == head
    assert "RuntimeError: boom" in state["errors"]
    assert session["status"] == "failed"
    assert session["current_phase"] == "failed"
    assert session["phases"]["preflight"]["status"] == "failed"
    assert session["phases"]["repository_intelligence"]["status"] == "pending"
    assert session["revisions"]["resolved_base_sha"] == base
    assert session["revisions"]["resolved_head_sha"] == head
    assert "Review failed" in capsys.readouterr().err


def test_cli_invalid_revision_does_not_create_session(git_repo: Path, capsys):
    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            "missing-base",
            "--head",
            "HEAD",
            "--non-interactive",
        ]
    )

    assert exit_code == 1
    assert "unable to resolve revisions" in capsys.readouterr().err
    run_root = git_repo / ".review-agent" / "runs"
    assert not run_root.exists() or not list(run_root.iterdir())


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
    envelope = json.loads((run_dir / "reviewer_envelope.json").read_text(encoding="utf-8"))

    assert trace["tool_call_count"] == 1
    assert result["status"] == "completed"
    assert raw["provider_name"] == "openai-compatible"
    assert envelope["parameters"]["model"] == "review-model"
    assert envelope["parameters"]["context"]["budget_scope"] == "messages_only"


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
    run_dirs = sorted((git_repo / ".review-agent" / "runs").iterdir())
    state = json.loads((run_dirs[-1] / "state.json").read_text(encoding="utf-8"))
    session = json.loads((run_dirs[-1] / "session.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["phase"] == "failed"
    assert "--reviewer-mode multi requires --reviewer-provider" in state["errors"][0]
    assert session["status"] == "failed"
    assert session["phases"]["reviewers"]["status"] == "failed"


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
    assert "## Verification Evidence" in (run_dir / "report.md").read_text(encoding="utf-8")


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
    assert "## Change Map And Repository Impact" in report
    assert "Repository intelligence:" in report
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
    assert "## Change Map And Repository Impact" in report
    assert "reviewer_count:" in report
    assert "## Review Contract Coverage" in report
    assert "## Non-Binding Recommendation" in report


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

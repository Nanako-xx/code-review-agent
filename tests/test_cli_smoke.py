from pathlib import Path
import hashlib
import json
import re

from conftest import run_git
from review_agent.attempts import AttemptWorkspace
from review_agent.checkpoint import CheckpointStore
from review_agent.cli import main
from review_agent.session_store import SessionStore


def _first_reviewer_artifact(run_dir: Path, kind: str) -> Path:
    for name in (f"reviewer_{kind}.json", f"reviewer_0_{kind}.json"):
        path = run_dir / name
        if path.exists():
            return path
    raise AssertionError(f"reviewer {kind} artifact was not written")


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
    assert (run_dirs[0] / "reconciliation.json").exists()
    assert (run_dirs[0] / "completion.json").exists()
    assert (run_dirs[0] / "report.md").exists()

    intent = json.loads((run_dirs[0] / "intent.json").read_text(encoding="utf-8"))
    risk = json.loads((run_dirs[0] / "risk.json").read_text(encoding="utf-8"))
    assignments = json.loads((run_dirs[0] / "assignments.json").read_text(encoding="utf-8"))
    completion = json.loads((run_dirs[0] / "completion.json").read_text(encoding="utf-8"))

    assert "uncertainties" in intent
    assert "unknowns" not in intent
    assert "signal_refs" in risk
    assert "evidence_refs" not in risk
    assert "initial_context" in assignments["assignments"][0]
    assert "provided_evidence_refs" not in assignments["assignments"][0]
    assert completion["status"] == "blocked"
    assert completion["blockers"] == ["Core Reviewer did not run"]


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
    assert session["schema_version"] == 4
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
        "risk_assessor": {
            "mode": "local",
            "provider": "none",
            "model": None,
            "base_url": None,
            "api_key_env": "REVIEW_AGENT_API_KEY",
            "max_output_tokens": 4096,
            "max_provider_attempts": 2,
            "max_elapsed_seconds": 60.0,
        },
        "portfolio_planner": {
            "mode": "local",
            "provider": "none",
            "model": None,
            "base_url": None,
            "api_key_env": "REVIEW_AGENT_API_KEY",
            "max_output_tokens": 4096,
            "max_provider_attempts": 2,
            "max_elapsed_seconds": 60.0,
        },
        "semantic_reconciler": {
            "mode": "local",
            "provider": "none",
            "model": None,
            "base_url": None,
            "api_key_env": "REVIEW_AGENT_API_KEY",
            "max_output_tokens": 4096,
            "max_provider_attempts": 2,
            "max_elapsed_seconds": 60.0,
        },
        "supplemental_policy": {
            "version": "supplemental_policy_v1",
            "risk_level": "critical",
            "max_waves": 2,
            "max_tasks": 4,
            "max_tasks_per_wave": 2,
            "max_concurrency": 2,
            "max_turns_per_task": 10,
            "max_tool_calls_per_task": 24,
            "max_tokens_per_task": 65536,
            "max_total_tokens": 262144,
            "max_elapsed_seconds": 600.0,
        },
    }
    assert secret not in session_text
    assert "state" not in session["artifacts"]

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


def test_cli_model_stage_inherit_and_overrides_persist_concrete_config(
    git_repo: Path,
) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("value = 2\n", encoding="utf-8")
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
            "Review the app change",
            "--reviewer-provider",
            "fake",
            "--reviewer-model",
            "review-model",
            "--reviewer-base-url",
            "https://reviewer.example/v1",
            "--reviewer-api-key-env",
            "REVIEWER_API_KEY",
            "--risk-assessor-mode",
            "model",
            "--risk-assessor-provider",
            "inherit",
            "--risk-assessor-max-output-tokens",
            "2048",
            "--risk-assessor-max-provider-attempts",
            "3",
            "--risk-assessor-max-elapsed-seconds",
            "45",
            "--portfolio-planner-mode",
            "model",
            "--portfolio-planner-provider",
            "fake",
            "--portfolio-planner-model",
            "planner-model",
            "--portfolio-planner-base-url",
            "https://planner.example/v1",
            "--portfolio-planner-api-key-env",
            "PLANNER_API_KEY",
            "--portfolio-planner-max-output-tokens",
            "3072",
            "--portfolio-planner-max-provider-attempts",
            "4",
            "--portfolio-planner-max-elapsed-seconds",
            "75",
            "--semantic-reconciler-mode",
            "model",
            "--semantic-reconciler-provider",
            "inherit",
            "--semantic-reconciler-model",
            "semantic-model",
            "--semantic-reconciler-api-key-env",
            "SEMANTIC_API_KEY",
            "--semantic-reconciler-max-output-tokens",
            "3584",
            "--semantic-reconciler-max-provider-attempts",
            "5",
            "--semantic-reconciler-max-elapsed-seconds",
            "90",
            "--non-interactive",
        ]
    )

    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    session_text = (run_dir / "session.json").read_text(encoding="utf-8")
    execution = json.loads(session_text)["execution"]

    assert exit_code == 0
    assert execution["risk_assessor"] == {
        "mode": "model",
        "provider": "fake",
        "model": "review-model",
        "base_url": "https://reviewer.example/v1",
        "api_key_env": "REVIEWER_API_KEY",
        "max_output_tokens": 2048,
        "max_provider_attempts": 3,
        "max_elapsed_seconds": 45.0,
    }
    assert execution["portfolio_planner"] == {
        "mode": "model",
        "provider": "fake",
        "model": "planner-model",
        "base_url": "https://planner.example/v1",
        "api_key_env": "PLANNER_API_KEY",
        "max_output_tokens": 3072,
        "max_provider_attempts": 4,
        "max_elapsed_seconds": 75.0,
    }
    assert execution["semantic_reconciler"] == {
        "mode": "model",
        "provider": "fake",
        "model": "semantic-model",
        "base_url": "https://reviewer.example/v1",
        "api_key_env": "SEMANTIC_API_KEY",
        "max_output_tokens": 3584,
        "max_provider_attempts": 5,
        "max_elapsed_seconds": 90.0,
    }
    assert execution["risk_assessor"]["provider"] != "inherit"
    assert execution["portfolio_planner"]["provider"] != "inherit"
    assert execution["semantic_reconciler"]["provider"] != "inherit"


def test_cli_model_stage_rejects_inheriting_reviewer_provider_none(
    git_repo: Path,
    capsys,
) -> None:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("value = 2\n", encoding="utf-8")
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
            "--risk-assessor-mode",
            "model",
            "--non-interactive",
        ]
    )

    assert exit_code == 2
    assert "risk-assessor: mode=model requires" in capsys.readouterr().out
    assert not (git_repo / ".review-agent" / "runs").exists()


def test_cli_semantic_reconciler_rejects_inheriting_reviewer_provider_none(
    git_repo: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            "HEAD",
            "--head",
            "HEAD",
            "--semantic-reconciler-mode",
            "model",
            "--non-interactive",
        ]
    )

    assert exit_code == 2
    assert "semantic-reconciler: mode=model requires" in capsys.readouterr().out
    assert not (git_repo / ".review-agent" / "runs").exists()


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

    result = json.loads(
        _first_reviewer_artifact(run_dir, "result").read_text(encoding="utf-8")
    )
    raw = json.loads(
        _first_reviewer_artifact(run_dir, "raw_response").read_text(
            encoding="utf-8"
        )
    )
    envelope = json.loads(
        _first_reviewer_artifact(run_dir, "envelope").read_text(encoding="utf-8")
    )
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert result["status"] == "partial"
    assert raw["provider_name"] == "fake"
    assert envelope["parameters"]["model"] == "fake-reviewer"
    assert envelope["parameters"]["context"]["budget_scope"] == "messages_only"
    assert "## Uncertainties" in report
    assert result["investigation_summary"] == "Fake reviewer executed."


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
    assert session["phases"]["intent_discovery"]["status"] == "failed"


def test_cli_review_records_failed_state_when_collection_fails(git_repo: Path, monkeypatch, capsys):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text("def check(token):\n    return token == 'ok'\n", encoding="utf-8")
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")

    def raise_error(*args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr("review_agent.command.collect_change_summary", raise_error)

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


def test_cli_session_create_failure_returns_clear_error(git_repo: Path, monkeypatch, capsys):
    def fail_create(self, manifest):
        raise OSError("session create unavailable")

    monkeypatch.setattr(SessionStore, "create", fail_create)

    exit_code = main(
        ["review", "--repo", str(git_repo), "--base", "HEAD", "--head", "HEAD", "--non-interactive"]
    )

    assert exit_code == 1
    error_output = capsys.readouterr().err
    assert "unable to create review Session" in error_output
    assert "session create unavailable" in error_output


def test_cli_legacy_state_write_failure_does_not_override_session(git_repo: Path, monkeypatch, capsys):
    def fail_state_write(self, state):
        raise OSError("initial state unavailable")

    monkeypatch.setattr(CheckpointStore, "write_state", fail_state_write)

    exit_code = main(
        ["review", "--repo", str(git_repo), "--base", "HEAD", "--head", "HEAD", "--non-interactive"]
    )

    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert session["status"] == "completed"
    assert all(phase["status"] == "completed" for phase in session["phases"].values())
    assert session["errors"] == []
    assert not (run_dir / "state.json").exists()
    error_output = capsys.readouterr().err
    assert "Warning:" in error_output
    assert "initial state unavailable" in error_output


def test_cli_request_write_failure_marks_session_and_state_failed(git_repo: Path, monkeypatch, capsys):
    original_write_json = AttemptWorkspace.write_json

    def fail_request_write(self, relative_path, payload):
        if relative_path == "request.json":
            raise OSError("request checkpoint unavailable")
        return original_write_json(self, relative_path, payload)

    monkeypatch.setattr(AttemptWorkspace, "write_json", fail_request_write)

    exit_code = main(
        ["review", "--repo", str(git_repo), "--base", "HEAD", "--head", "HEAD", "--non-interactive"]
    )

    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert session["status"] == "failed"
    assert session["phases"]["preflight"]["status"] == "failed"
    assert "OSError: request checkpoint unavailable" in session["errors"]
    assert state["status"] == "failed"
    assert "OSError: request checkpoint unavailable" in state["errors"]
    error_output = capsys.readouterr().err
    assert "Review failed:" in error_output
    assert "request checkpoint unavailable" in error_output


def test_cli_finalization_failure_leaves_retryable_running_session(git_repo: Path, monkeypatch, capsys):
    def fail_finalization(self, now):
        raise RuntimeError("session finalization unavailable")

    monkeypatch.setattr(SessionStore, "mark_session_completed", fail_finalization)

    exit_code = main(
        ["review", "--repo", str(git_repo), "--base", "HEAD", "--head", "HEAD", "--non-interactive"]
    )

    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert session["status"] == "running"
    assert session["current_phase"] == "reporting"
    assert all(phase["status"] == "completed" for phase in session["phases"].values())
    assert session["errors"] == []
    assert state["status"] == "running"
    assert state["phase"] != "failed"
    error_output = capsys.readouterr().err
    assert "Review failed: session finalization unavailable" in error_output
    assert "retryable" in error_output


def test_cli_completed_state_write_is_best_effort(git_repo: Path, monkeypatch, capsys):
    original_write_state = CheckpointStore.write_state

    def fail_completed_state(self, state):
        if state.status.value == "completed":
            raise OSError("completed state unavailable")
        return original_write_state(self, state)

    monkeypatch.setattr(CheckpointStore, "write_state", fail_completed_state)

    exit_code = main(
        ["review", "--repo", str(git_repo), "--base", "HEAD", "--head", "HEAD", "--non-interactive"]
    )

    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert session["status"] == "completed"
    assert state["status"] == "running"
    error_output = capsys.readouterr().err
    assert "Warning:" in error_output
    assert "completed state unavailable" in error_output


def test_cli_quality_gate_uses_resolved_head_when_worktree_is_dirty(git_repo: Path):
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "valid target head")
    (git_repo / "app.py").write_text("def broken(:\n", encoding="utf-8")

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

    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    quality = json.loads((run_dir / "quality_gates.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert quality["results"][0]["status"] == "passed"


def test_cli_quality_gate_uses_non_checked_out_head_commit(git_repo: Path):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "target.py").write_text("def broken(:\n", encoding="utf-8")
    run_git(git_repo, "add", "target.py")
    run_git(git_repo, "commit", "-m", "invalid target head")
    target_head = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "target.py").write_text("def fixed():\n    return 1\n", encoding="utf-8")
    run_git(git_repo, "add", "target.py")
    run_git(git_repo, "commit", "-m", "valid current head")

    exit_code = main(
        [
            "review",
            "--repo",
            str(git_repo),
            "--base",
            base,
            "--head",
            target_head,
            "--non-interactive",
        ]
    )

    run_dir = sorted((git_repo / ".review-agent" / "runs").iterdir())[-1]
    quality = json.loads((run_dir / "quality_gates.json").read_text(encoding="utf-8"))
    session = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert quality["results"][0]["status"] == "failed"
    assert session["revisions"]["resolved_head_sha"] == target_head


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
                                    "contract": contract,
                                    "status": "covered",
                                    "summary": "OpenAI-compatible adapter path used tools.",
                                    "evidence_refs": [observation_id],
                                }
                                for contract in (
                                    "intent_alignment",
                                    "behavioral_correctness",
                                    "regression_safety",
                                    "test_adequacy",
                                    "unresolved_uncertainties",
                                )
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

    monkeypatch.setattr("review_agent.command.build_model_adapter_factory_from_config", fake_build_factory)
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
    trace = json.loads(
        _first_reviewer_artifact(run_dir, "agent_trace").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(
        _first_reviewer_artifact(run_dir, "result").read_text(encoding="utf-8")
    )
    raw = json.loads(
        _first_reviewer_artifact(run_dir, "raw_response").read_text(
            encoding="utf-8"
        )
    )
    envelope = json.loads(
        _first_reviewer_artifact(run_dir, "envelope").read_text(encoding="utf-8")
    )

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

    envelope = json.loads(
        _first_reviewer_artifact(run_dir, "envelope").read_text(encoding="utf-8")
    )
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
    envelope = json.loads(
        _first_reviewer_artifact(run_dir, "envelope").read_text(encoding="utf-8")
    )

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
    trace = json.loads(
        _first_reviewer_artifact(run_dir, "agent_trace").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(
        _first_reviewer_artifact(run_dir, "result").read_text(encoding="utf-8")
    )

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
    trace = json.loads(
        _first_reviewer_artifact(run_dir, "agent_trace").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(
        _first_reviewer_artifact(run_dir, "result").read_text(encoding="utf-8")
    )

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

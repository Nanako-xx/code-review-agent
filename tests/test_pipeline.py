from __future__ import annotations

import json
from pathlib import Path
import threading
import time

from conftest import run_git

from review_agent.checkpoint import CheckpointStore
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_adapter_factory import FakeModelAdapterFactory
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnResponse,
)
from review_agent.models import QualityGateResult, ReviewRequest
from review_agent.pipeline import PHASE_MESSAGES, ReviewPipeline
from review_agent.quality import QualityGateExecution
from review_agent.revision import RevisionResolver
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    ModelStageConfig,
    PhaseStatus,
    ReviewExecutionConfig,
    initial_session_manifest,
)
from review_agent.session_store import SessionStore


def _pipeline(
    git_repo: Path,
    *,
    review_id: str = "review-pipeline",
    reviewer_mode: str = "single",
    reviewer_loop: str = "agent-loop",
    adapter_factory_builder=None,
    changed_path: str = "app.py",
    risk_assessor: ModelStageConfig | None = None,
    portfolio_planner: ModelStageConfig | None = None,
) -> tuple[ReviewPipeline, SessionStore, CheckpointStore]:
    base = run_git(git_repo, "rev-parse", "HEAD")
    changed_file = git_repo / changed_path
    changed_file.parent.mkdir(parents=True, exist_ok=True)
    changed_file.write_text(
        (
            "def add(a, b):\n    return a - b\n"
            if changed_path == "app.py"
            else "def check(token):\n    return token == 'ok'\n"
        ),
        encoding="utf-8",
    )
    run_git(git_repo, "add", changed_path)
    run_git(git_repo, "commit", "-m", "change implementation")
    head = run_git(git_repo, "rev-parse", "HEAD")
    resolver = RevisionResolver()
    identity = resolver.repository_identity(git_repo)
    revisions = resolver.resolve_pair(git_repo, base, head)
    checkpoint_store = CheckpointStore(git_repo, review_id)
    session_store = SessionStore(checkpoint_store.run_dir)
    session_store.create(
        initial_session_manifest(
            review_id=review_id,
            repository=identity,
            revisions=revisions,
            execution=ReviewExecutionConfig(
                reviewer_provider="fake",
                reviewer_model=None,
                reviewer_base_url=None,
                reviewer_api_key_env="REVIEW_AGENT_API_KEY",
                reviewer_mode=reviewer_mode,
                reviewer_loop=reviewer_loop,
                non_interactive=True,
                risk_assessor=risk_assessor or ModelStageConfig(),
                portfolio_planner=portfolio_planner or ModelStageConfig(),
            ),
            now="2026-07-12T00:00:00Z",
        )
    )
    request = ReviewRequest(
        repository_path=identity.canonical_path,
        base_revision=base,
        head_revision=head,
        user_intent="Preserve addition semantics",
    )
    pipeline_kwargs = {}
    if adapter_factory_builder is not None:
        pipeline_kwargs["adapter_factory_builder"] = adapter_factory_builder
    return (
        ReviewPipeline(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
            request=request,
            **pipeline_kwargs,
        ),
        session_store,
        checkpoint_store,
    )


def test_review_pipeline_runs_all_phases_through_atomic_attempts(git_repo: Path) -> None:
    pipeline, session_store, checkpoint_store = _pipeline(git_repo)

    result = pipeline.execute()

    manifest = session_store.load()
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.current_phase is RunPhase.COMPLETED
    assert all(
        checkpoint.status is PhaseStatus.COMPLETED
        for checkpoint in manifest.phases.values()
    )
    reviewer = manifest.phases["reviewers"].tasks["reviewer-0"]
    assert reviewer.status is PhaseStatus.COMPLETED
    assert reviewer.attempts == 1
    assert "reviewer_0_observations" in reviewer.artifacts
    assert manifest.artifacts["quality_gate_plan"].phase is RunPhase.QUALITY_GATES
    assert manifest.artifacts["quality_gate_observations"].phase is RunPhase.QUALITY_GATES
    assert manifest.artifacts["deep_quality_gates"].phase is RunPhase.PLANNING
    assert manifest.artifacts["deep_quality_gate_observations"].phase is RunPhase.PLANNING
    assert manifest.artifacts["repository_observations"].phase is RunPhase.REPOSITORY_INTELLIGENCE
    assert manifest.artifacts["observations"].phase is RunPhase.REPORTING
    assert (checkpoint_store.run_dir / "report.md").exists()
    assert (checkpoint_store.run_dir / "observations.jsonl").exists()
    assert (checkpoint_store.run_dir / "attempts" / "preflight" / "1").is_dir()
    assert result.context.brief is not None
    assert result.context.final_risk is not None


def test_model_assisted_risk_and_portfolio_are_runtime_compiled_and_audited(
    git_repo: Path,
) -> None:
    model_stage = ModelStageConfig(mode="model", provider="fake")
    pipeline, session_store, checkpoint_store = _pipeline(
        git_repo,
        review_id="review-model-planning",
        risk_assessor=model_stage,
        portfolio_planner=model_stage,
    )

    result = pipeline.execute()

    assert result.context.risk_assessment is not None
    assert result.context.risk_assessment.level.value == "medium"
    assert len(result.context.assignments) == 2
    assert [item.role_kind for item in result.context.assignments] == [
        "core",
        "adversarial",
    ]
    assert set(result.context.assignments[0].assigned_contract) == {
        "intent_alignment",
        "behavioral_correctness",
        "regression_safety",
        "test_adequacy",
        "unresolved_uncertainties",
    }
    assert all(
        item.repository_permission == "read_only"
        and item.command_permission == "safe_checks_only"
        for item in result.context.assignments
    )
    assert len(result.context.reviewer_executions) == 2
    assert result.context.multi_run is not None

    manifest = session_store.load()
    planning_artifacts = {
        name
        for name, descriptor in manifest.artifacts.items()
        if descriptor.phase is RunPhase.PLANNING
    }
    assert {
        "risk_model_envelope",
        "risk_model_raw_response",
        "risk_model_decision",
        "portfolio_packet",
        "portfolio_model_envelope",
        "portfolio_model_raw_response",
        "portfolio_model_decision",
        "portfolio_plan",
        "planning_summary",
    }.issubset(planning_artifacts)
    risk_decision = json.loads(
        (checkpoint_store.run_dir / "risk_model_decision.json").read_text(
            encoding="utf-8"
        )
    )
    portfolio_decision = json.loads(
        (checkpoint_store.run_dir / "portfolio_model_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert risk_decision["status"] == "accepted"
    assert risk_decision["local_floor"] == "low"
    assert risk_decision["model_proposed_level"] == "medium"
    assert portfolio_decision["status"] == "accepted"
    assert portfolio_decision["final_reviewer_count"] == 2
    assert any(
        action.startswith("injected_required_role:adversarial")
        for action in portfolio_decision["policy_actions"]
    )
    assert result.context.brief is not None
    assert result.context.brief.orchestration["risk"]["status"] == "accepted"


def test_model_planning_failures_fall_back_without_hiding_uncertainty(
    git_repo: Path,
) -> None:
    class InvalidFactory:
        def create(self):
            return FakeToolCallingAdapter(
                script=[
                    ModelTurnResponse(
                        kind=ModelResponseKind.INVALID,
                        error="synthetic planning failure",
                    ),
                    ModelTurnResponse(
                        kind=ModelResponseKind.INVALID,
                        error="synthetic planning failure",
                    ),
                ]
            )

    def stage_aware_builder(config):
        if config.stage_label in {"risk-assessor", "portfolio-planner"}:
            return InvalidFactory()
        return FakeModelAdapterFactory()

    model_stage = ModelStageConfig(mode="model", provider="fake")
    pipeline, _, checkpoint_store = _pipeline(
        git_repo,
        review_id="review-model-planning-fallback",
        risk_assessor=model_stage,
        portfolio_planner=model_stage,
        adapter_factory_builder=stage_aware_builder,
    )

    result = pipeline.execute()

    risk_decision = json.loads(
        (checkpoint_store.run_dir / "risk_model_decision.json").read_text(
            encoding="utf-8"
        )
    )
    portfolio_decision = json.loads(
        (checkpoint_store.run_dir / "portfolio_model_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert risk_decision["status"] == "fallback"
    assert risk_decision["final_level"] == risk_decision["local_floor"]
    assert portfolio_decision["status"] == "fallback"
    assert len(result.context.assignments) == 1
    assert result.context.assignments[0].role_kind == "core"
    assert result.context.brief is not None
    assert any(
        "Model Risk Assessor fallback" in uncertainty
        for uncertainty in result.context.brief.uncertainties
    )
    assert any(
        "Portfolio planner fallback" in uncertainty
        for uncertainty in result.context.brief.uncertainties
    )


def test_risk_triggered_gate_reaches_observations_and_reviewer_context(
    git_repo: Path,
    monkeypatch,
) -> None:
    (git_repo / "tests").mkdir()
    (git_repo / "tests" / "test_app.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )
    (git_repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "pyproject.toml", "tests/test_app.py")
    run_git(git_repo, "commit", "-m", "configure pytest")

    calls: list[str] = []

    def fake_execute(_repo, _revision, gate):
        calls.append(gate.name)
        return QualityGateExecution(
            result=QualityGateResult(
                name=gate.name,
                status="passed",
                command=list(gate.command),
                summary="pytest passed in isolated snapshot",
                category=gate.category,
                cost=gate.cost,
                source=gate.source,
                blocking=gate.blocking,
                duration_seconds=0.01,
                sandbox="test-sandbox",
            ),
            raw_output="1 passed",
        )

    monkeypatch.setattr("review_agent.pipeline.execute_quality_gate", fake_execute)
    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-deep-quality",
        changed_path="auth.py",
    )

    result = pipeline.execute()

    assert calls == ["pytest"]
    assert [gate.name for gate in result.context.quality_results] == [
        "python_compile",
        "pytest",
    ]
    pytest_result = result.context.quality_results[1]
    assert pytest_result.observation_ref is not None
    assert result.context.assignments[0].initial_context.quality_gate_summary == {
        "python_compile": "passed",
        "pytest": "passed",
    }
    assert pytest_result.observation_ref in (
        result.context.assignments[0].initial_context.observation_refs
    )
    assert session_store.load().status is RunStatus.COMPLETED


def test_quality_discovery_observation_reaches_reviewer_context(
    git_repo: Path,
) -> None:
    (git_repo / "pyproject.toml").write_text(
        """
[[tool.review-agent.quality-gates]]
name = "unsafe_gate"
category = "security"
command = ["bash", "-c", "echo unsafe"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "pyproject.toml")
    run_git(git_repo, "commit", "-m", "add invalid quality gate")
    pipeline, _session_store, _checkpoint_store = _pipeline(
        git_repo,
        review_id="review-quality-discovery",
    )

    result = pipeline.execute()

    assert result.context.quality_gate_plan is not None
    assert result.context.quality_gate_plan.discovery_issues
    assert result.context.quality_gate_observations is not None
    discovery = next(
        observation
        for observation in result.context.quality_gate_observations.list_observations()
        if observation.source == "quality_gate.discovery"
    )
    assert all(
        discovery.observation_id in assignment.initial_context.observation_refs
        for assignment in result.context.assignments
    )
    assert result.context.completion is not None
    assert any(
        "Quality gate discovery issue" in uncertainty
        for uncertainty in result.context.completion.uncertainties
    )


def test_blocking_gate_unavailable_does_not_abort_reviewers_but_blocks_completion(
    git_repo: Path,
    monkeypatch,
) -> None:
    (git_repo / "pyproject.toml").write_text(
        """
[[tool.review-agent.quality-gates]]
name = "required_security"
category = "security"
cost = "expensive"
command = ["python", "-m", "bandit"]
blocking = true
trigger_risks = ["high", "critical"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "pyproject.toml")
    run_git(git_repo, "commit", "-m", "require security gate")

    def unavailable(_repo, _revision, gate):
        return QualityGateExecution(
            result=QualityGateResult(
                name=gate.name,
                status="unavailable",
                command=list(gate.command),
                summary="bandit is unavailable",
                category=gate.category,
                cost=gate.cost,
                source=gate.source,
                blocking=gate.blocking,
                reason="bandit is not installed",
                sandbox="test-sandbox",
            ),
            raw_output="bandit is not installed",
        )

    monkeypatch.setattr("review_agent.pipeline.execute_quality_gate", unavailable)
    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-blocking-quality",
        changed_path="auth.py",
    )

    result = pipeline.execute()

    gate = next(
        item
        for item in result.context.quality_results
        if item.name == "required_security"
    )
    assert gate.status == "unavailable"
    assert len(result.context.reviewer_executions) == 3
    assert result.context.multi_run is not None
    assert result.context.completion is not None
    assert result.context.completion.status == "blocked"
    assert any("required_security" in item for item in result.context.completion.blockers)
    manifest = session_store.load()
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.phases["reviewers"].tasks["reviewer-0"].status is PhaseStatus.COMPLETED


def test_completed_pipeline_hydrates_every_phase_without_provider_execution(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _pipeline(git_repo)
    pipeline.execute()

    def provider_must_not_run(_config):
        raise AssertionError("provider must not be rebuilt while hydrating")

    hydrated = ReviewPipeline(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
        adapter_factory_builder=provider_must_not_run,
    )
    for phase in PHASE_MESSAGES:
        hydrated.load_phase(phase)

    assert hydrated.context.request is not None
    assert hydrated.context.repository_intelligence is not None
    assert hydrated.context.reviewer_result is not None
    assert hydrated.context.final_risk is not None
    assert hydrated.context.brief is not None


def test_agent_loop_persists_head_bound_tool_observation(git_repo: Path) -> None:
    class ReadRangeFactory:
        def create(self):
            def final_response(request):
                observation_id = request.tool_results[-1].observation_ids[0]
                contracts = (
                    "intent_alignment",
                    "behavioral_correctness",
                    "regression_safety",
                    "test_adequacy",
                )
                return ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=json.dumps(
                        {
                            "contract_assessments": [
                                {
                                    "contract": contract,
                                    "status": "covered",
                                    "summary": "Inspected the reviewed head revision.",
                                    "evidence_refs": [observation_id],
                                }
                                for contract in contracts
                            ],
                            "confirmed_findings": [],
                            "rejected_hypotheses": [],
                            "uncertainties": [],
                            "observation_refs": [observation_id],
                            "investigation_summary": "Read the head revision through ToolGateway.",
                            "status": "completed",
                        }
                    ),
                )

            return FakeToolCallingAdapter(
                script=[
                    ModelTurnResponse(
                        kind=ModelResponseKind.TOOL_CALLS,
                        tool_calls=[
                            ModelToolCall(
                                "call-read",
                                "read_range",
                                {
                                    "path": "app.py",
                                    "revision": "head",
                                    "line_start": 1,
                                    "line_end": 2,
                                },
                            )
                        ],
                    ),
                    final_response,
                ]
            )

    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-head-observation",
        adapter_factory_builder=lambda _config: ReadRangeFactory(),
    )

    result = pipeline.execute()

    revisions = result.context.manifest.revisions
    reviewer_store = result.context.reviewer_observations[0]
    assert session_store.load().status is RunStatus.COMPLETED
    assert any(
        observation.revision == f"head@{revisions.resolved_head_sha}"
        for observation in reviewer_store.list_observations()
    )


def test_single_reviewer_findings_flow_through_reconciliation_and_brief(
    git_repo: Path,
) -> None:
    class FindingFactory:
        def create(self):
            def final_response(request):
                content = str(request.messages[0]["content"])
                observation_id = next(
                    line.split(":", 1)[0]
                    for line in content.splitlines()
                    if line.startswith("O-")
                )
                return ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    final_text=json.dumps(
                        {
                            "contract_assessments": [],
                            "confirmed_findings": [
                                {
                                    "claim": "The implementation subtracts instead of adding.",
                                    "severity": "high",
                                    "confidence": "high",
                                    "path": "app.py",
                                    "line": 2,
                                    "evidence_refs": [observation_id],
                                    "impact": "Callers receive incorrect arithmetic results.",
                                    "suggested_action": "Restore addition semantics.",
                                    "verification_performed": [
                                        "Compared app.py between base and head."
                                    ],
                                }
                            ],
                            "rejected_hypotheses": [],
                            "uncertainties": ["Single-shot review did not inspect callers."],
                            "observation_refs": [observation_id],
                            "investigation_summary": "Confirmed the changed arithmetic behavior.",
                            "status": "partial",
                        }
                    ),
                )

            return FakeToolCallingAdapter(script=[final_response])

    pipeline, _, _ = _pipeline(
        git_repo,
        review_id="review-single-finding",
        reviewer_loop="single-shot",
        adapter_factory_builder=lambda _config: FindingFactory(),
    )

    result = pipeline.execute()

    assert result.context.reconciliation is not None
    assert len(result.context.reconciliation.canonical_findings) == 1
    assert result.context.brief is not None
    finding = result.context.brief.verified_findings[0]
    assert finding.path == "app.py"
    assert finding.line == 2
    assert finding.impact == "Callers receive incorrect arithmetic results."


def test_pipeline_overlaps_reviewer_calls_and_commits_in_stable_order(
    git_repo: Path,
) -> None:
    class State:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.release = threading.Event()
            self.active = 0
            self.max_active = 0

    state = State()

    class OverlapAdapter:
        provider_name = "overlap-fake"

        def __init__(self) -> None:
            self.delegate = FakeModelAdapterFactory().create()

        def complete_turn(self, request):
            is_reviewer = (
                request.parameters.get("response_schema")
                == "reviewer_assignment_result_v2"
            )
            if not is_reviewer:
                return self.delegate.complete_turn(request)
            with state.lock:
                state.active += 1
                state.max_active = max(state.max_active, state.active)
                if state.active >= 2:
                    state.release.set()
            try:
                state.release.wait(timeout=1)
                time.sleep(0.02)
                return self.delegate.complete_turn(request)
            finally:
                with state.lock:
                    state.active -= 1

    class OverlapFactory:
        def create(self):
            return OverlapAdapter()

    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-parallel-reviewers",
        reviewer_mode="multi",
        changed_path="auth.py",
        adapter_factory_builder=lambda _config: OverlapFactory(),
    )

    result = pipeline.execute()

    assert state.max_active >= 2
    assert [
        execution.reviewer_index
        for execution in result.context.reviewer_executions
    ] == [0, 1, 2]
    tasks = session_store.load().phases["reviewers"].tasks
    assert list(tasks) == ["reviewer-0", "reviewer-1", "reviewer-2"]
    assert all(task.status is PhaseStatus.COMPLETED for task in tasks.values())


def test_pipeline_persists_provider_failure_without_aborting_other_reviewers(
    git_repo: Path,
) -> None:
    class FailSecondReviewerFactory:
        def __init__(self) -> None:
            self.created = 0

        def create(self):
            self.created += 1
            # Intent inference owns adapter 1. Reviewer indices 0, 1, 2 own
            # adapters 2, 3, 4 respectively.
            if self.created == 3:
                raise RuntimeError("reviewer provider creation failed")
            return FakeModelAdapterFactory().create()

    factory = FailSecondReviewerFactory()
    pipeline, session_store, checkpoint_store = _pipeline(
        git_repo,
        review_id="review-isolated-provider-failure",
        reviewer_mode="multi",
        changed_path="auth.py",
        adapter_factory_builder=lambda _config: factory,
    )

    result = pipeline.execute()

    executions = result.context.reviewer_executions
    assert [item.result.status.value for item in executions] == [
        "completed",
        "failed",
        "completed",
    ]
    assert executions[1].runtime.termination_reason.value == "runtime_failure"
    assert "reviewer provider creation failed" in executions[1].result.investigation_summary
    tasks = session_store.load().phases["reviewers"].tasks
    assert all(task.status is PhaseStatus.COMPLETED for task in tasks.values())

    raw_response = json.loads(
        (checkpoint_store.run_dir / "reviewer_1_raw_response.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw_response["runtime"]["termination_reason"] == "runtime_failure"
    assert result.context.brief is not None
    reviewer_summary = result.context.brief.change_map_and_repository_impact[
        "reviewer_summary"
    ]
    assert reviewer_summary["termination_counts"]["runtime_failure"] == 1
    assert reviewer_summary["executions"][1]["status"] == "failed"

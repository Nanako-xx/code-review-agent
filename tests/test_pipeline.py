from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import pytest

from conftest import run_git

from review_agent.checkpoint import CheckpointStore
from review_agent.memory_models import MemoryExecutionConfig, MemoryMode
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_adapter_factory import FakeModelAdapterFactory
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnResponse,
)
from review_agent.models import QualityGateResult, ReviewRequest
from review_agent.observations import ObservationStore
from review_agent.pipeline import PHASE_MESSAGES, PipelineStageError, ReviewPipeline
from review_agent.quality import QualityGateExecution
from review_agent.resume import ResumeAction, ReviewSessionResumer
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
    semantic_reconciler: ModelStageConfig | None = None,
    existing_ci_evidence: tuple[str, ...] = (),
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
                semantic_reconciler=semantic_reconciler or ModelStageConfig(),
                memory=MemoryExecutionConfig(
                    mode=MemoryMode.OFF,
                    root_path=str((git_repo / ".memory-test").resolve()),
                ),
            ),
            now="2026-07-12T00:00:00Z",
        )
    )
    request = ReviewRequest(
        repository_path=identity.canonical_path,
        base_revision=base,
        head_revision=head,
        user_intent="Preserve addition semantics",
        existing_ci_evidence=existing_ci_evidence,
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


def test_ci_evidence_reaches_each_reviewer_without_cross_reviewer_leakage(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-ci-evidence-reviewers",
        reviewer_mode="multi",
        changed_path="auth.py",
        existing_ci_evidence=("security suite passed", ""),
    )

    result = pipeline.execute()

    assert session_store.load().status is RunStatus.COMPLETED
    assert len(result.context.reviewer_executions) >= 2
    assert result.context.intent_observations is not None
    ci_observations = [
        item
        for item in result.context.intent_observations.list_observations()
        if item.source.startswith("review_request.existing_ci_evidence:")
    ]
    assert len(ci_observations) == 2
    ci_summaries = {
        item.observation_id: result.context.intent_observations.summaries_by_id()[
            item.observation_id
        ]
        for item in ci_observations
    }
    for execution in result.context.reviewer_executions:
        model_context = "\n".join(
            str(message["content"]) for message in execution.envelope.messages
        )
        assert all(
            observation_id in model_context and summary in model_context
            for observation_id, summary in ci_summaries.items()
        )

    head = result.context.manifest.revisions.resolved_head_sha
    private_stores: dict[int, ObservationStore] = {}
    private_ids: dict[int, str] = {}
    for index in range(len(result.context.reviewer_executions)):
        store = ObservationStore(tmp_path / f"private-reviewer-{index}")
        observation = store.record(
            source=f"reviewer.private:{index}",
            revision=f"head@{head}",
            path=None,
            line_start=None,
            line_end=None,
            raw_content=f"private reviewer observation {index}",
            context_view=f"Private reviewer observation {index}",
        )
        private_stores[index] = store
        private_ids[index] = observation.observation_id

    pipeline.context.reviewer_observations = private_stores
    for index, current_store in private_stores.items():
        authorized = pipeline._reviewer_authorized_observation_summaries(
            current_store
        )
        assert set(ci_summaries).issubset(authorized)
        assert private_ids[index] in authorized
        assert all(
            private_id not in authorized
            for other_index, private_id in private_ids.items()
            if other_index != index
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
                observation_id = [
                    line.split(":", 1)[0]
                    for line in content.splitlines()
                    if line.startswith("O-")
                ][-1]
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


def test_model_semantic_reconciliation_without_requests_runs_zero_supplemental_tasks(
    git_repo: Path,
) -> None:
    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-semantic-no-supplemental",
        semantic_reconciler=ModelStageConfig(mode="model", provider="fake"),
    )

    result = pipeline.execute()

    manifest = session_store.load()
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.supplemental_waves == {}
    assert result.context.supplemental_executions == []
    assert result.context.semantic_reconciliation is not None
    assert result.context.semantic_reconciliation.status == "accepted"
    assert result.context.semantic_reconciliation.supplemental.status == "not_needed"
    assert result.context.semantic_reconciliation.supplemental.stop_reason == "no_requests"
    assert result.context.supplemental_plan is not None
    effective_limits = result.context.supplemental_plan.limits.budget_limits
    assert result.context.semantic_reconciliation.supplemental.budget["limits"] == {
        "tasks": effective_limits.tasks,
        "tool_calls": effective_limits.tool_calls,
        "tokens": effective_limits.tokens,
        "elapsed_seconds": effective_limits.elapsed_seconds,
    }


def test_semantic_provider_failure_falls_back_and_requires_manual_review(
    git_repo: Path,
) -> None:
    class FindingFactory:
        def create(self):
            def final_response(request):
                content = str(request.messages[0]["content"])
                observation_id = [
                    line.split(":", 1)[0]
                    for line in content.splitlines()
                    if line.startswith("O-")
                ][-1]
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
                                    "impact": "Callers receive incorrect results.",
                                    "suggested_action": "Restore addition semantics.",
                                    "verification_performed": ["Compared base and head."],
                                }
                            ],
                            "rejected_hypotheses": [],
                            "uncertainties": [],
                            "observation_refs": [observation_id],
                            "investigation_summary": "Confirmed the arithmetic regression.",
                            "status": "partial",
                        }
                    ),
                )

            return FakeToolCallingAdapter(script=[final_response])

    class UnavailableSemanticFactory:
        def create(self):
            raise RuntimeError("semantic provider unavailable")

    def build_factory(config):
        if config.stage_label == "semantic-reconciler":
            return UnavailableSemanticFactory()
        return FindingFactory()

    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-semantic-provider-fallback",
        reviewer_loop="single-shot",
        semantic_reconciler=ModelStageConfig(mode="model", provider="fake"),
        adapter_factory_builder=build_factory,
    )

    result = pipeline.execute()

    manifest = session_store.load()
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.supplemental_waves == {}
    assert result.context.semantic_reconciliation is not None
    assert result.context.semantic_reconciliation.status == "fallback"
    assert result.context.semantic_reconciliation.model.status == "fallback"
    assert result.context.completion is not None
    assert result.context.completion.status == "completed_with_uncertainties"
    assert result.context.completion.recommendation == "manual_review"
    assert any(
        "Semantic reconciliation used deterministic fallback" in item
        for item in result.context.completion.uncertainties
    )


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        ("unavailable", "unavailable"),
        ("partial", "partial"),
        ("failed", "failed"),
        ("max_waves", "budget_exhausted"),
    ],
)
def test_supplemental_provider_outcomes_retain_disagreement_and_require_manual_review(
    git_repo: Path,
    outcome: str,
    expected_status: str,
) -> None:
    semantic_calls = 0

    class SemanticAdapter:
        provider_name = "semantic-outcome-test"

        def complete_turn(self, request):
            nonlocal semantic_calls
            semantic_calls += 1
            packet = json.loads(request.messages[0]["content"])
            candidate_id, candidate = next(iter(packet["candidate_catalog"].items()))
            observation_id = candidate["evidence_refs"][0]
            first_pass = semantic_calls == 1
            requires_investigation = first_pass or outcome == "max_waves"
            return ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                provider_name=self.provider_name,
                final_text=json.dumps(
                    {
                        "canonical_groups": [
                            {
                                "member_ids": [candidate_id],
                                "representative_id": candidate_id,
                                "canonical_claim": candidate["claim"],
                                "rationale": "The authorized Observation supports the candidate.",
                                "supporting_refs": [observation_id],
                                "proposed_confidence": candidate["confidence"],
                            }
                        ],
                        "rejections": [],
                        "disagreements": [
                            {
                                "disagreement_id": "D-provider-outcome",
                                "candidate_ids": [candidate_id],
                                "status": (
                                    "needs_investigation"
                                    if requires_investigation
                                    else "resolved"
                                ),
                                "issue": "A targeted provider check is required.",
                                "resolution": (
                                    ""
                                    if requires_investigation
                                    else "The model proposed resolution."
                                ),
                                "decision_refs": [observation_id],
                            }
                        ],
                        "supplemental_requests": (
                            [
                                {
                                    "disagreement_id": "D-provider-outcome",
                                    "question": "Does HEAD preserve the intended arithmetic behavior?",
                                    "required_evidence": ["Inspect the changed return expression."],
                                    "preferred_perspective": "behavioral correctness",
                                    "related_candidate_ids": [candidate_id],
                                    "reason_refs": [observation_id],
                                }
                            ]
                            if requires_investigation
                            else []
                        ),
                        "uncertainties": [],
                        "summary": "Semantic provider-outcome scenario.",
                    }
                ),
            )

    class SemanticFactory:
        def create(self):
            return SemanticAdapter()

    class ReviewerAdapter:
        provider_name = "reviewer-outcome-test"

        def complete_turn(self, request):
            content = str(request.messages[0]["content"])
            observation_id = [
                line.split(":", 1)[0]
                for line in content.splitlines()
                if line.startswith("O-")
            ][-1]
            contract_line = next(
                line for line in content.splitlines() if line.startswith("Assigned Contract:")
            )
            contracts = [
                item.strip()
                for item in contract_line.split(":", 1)[1].split(",")
                if item.strip()
            ]
            is_supplemental = any(
                contract.startswith("supplemental_investigation:")
                for contract in contracts
            )
            if is_supplemental and outcome == "failed":
                raise RuntimeError("supplemental provider call failed")
            payload = {
                "contract_assessments": (
                    [
                        {
                            "contract": contract,
                            "status": "partial" if outcome == "partial" else "covered",
                            "summary": "The targeted provider returned incomplete evidence.",
                            "evidence_refs": [observation_id],
                        }
                        for contract in contracts
                    ]
                    if is_supplemental
                    else []
                ),
                "confirmed_findings": (
                    []
                    if is_supplemental
                    else [
                        {
                            "claim": "The implementation subtracts instead of adding.",
                            "severity": "high",
                            "confidence": "high",
                            "path": "app.py",
                            "line": 2,
                            "evidence_refs": [observation_id],
                            "impact": "Callers receive incorrect arithmetic results.",
                            "suggested_action": "Restore addition semantics.",
                            "verification_performed": ["Compared base and head."],
                        }
                    ]
                ),
                "rejected_hypotheses": [],
                "uncertainties": (
                    ["The targeted provider returned incomplete evidence."]
                    if is_supplemental and outcome == "partial"
                    else []
                ),
                "observation_refs": [observation_id],
                "investigation_summary": "Provider outcome scenario completed.",
                "status": "partial",
            }
            if is_supplemental and outcome == "max_waves":
                payload["status"] = "completed"
            return ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                provider_name=self.provider_name,
                final_text=json.dumps(payload),
                raw={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            )

    class ReviewerFactory:
        def create(self):
            return ReviewerAdapter()

    def build_factory(config):
        if config.stage_label == "semantic-reconciler":
            return SemanticFactory()
        if (
            outcome == "unavailable"
            and pipeline.context.manifest.current_phase
            is RunPhase.SUPPLEMENTAL_INVESTIGATION
        ):
            return None
        return ReviewerFactory()

    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id=f"review-supplemental-{outcome}",
        reviewer_loop="single-shot",
        semantic_reconciler=ModelStageConfig(mode="model", provider="fake"),
        adapter_factory_builder=build_factory,
    )

    result = pipeline.execute()

    manifest = session_store.load()
    semantic = result.context.semantic_reconciliation
    assert semantic is not None
    assert pipeline.validate_completed_supplemental_state() == ()
    assert semantic.supplemental.status == expected_status
    assert semantic.remaining_disagreements
    assert result.context.completion is not None
    assert result.context.completion.recommendation == "manual_review"
    if outcome == "unavailable":
        assert semantic_calls == 1
        assert manifest.supplemental_waves == {}
        assert semantic.supplemental.unavailable == 1
        return

    assert semantic_calls == 2
    wave = next(iter(manifest.supplemental_waves.values()))
    task = next(iter(wave.tasks.values()))
    if outcome == "max_waves":
        assert wave.stop_reason == "max_waves"
        assert task.status.value == "completed"
        assert result.context.completion.status == "budget_exhausted"
    else:
        assert wave.stop_reason == "task_failure"
        assert task.status.value == outcome
    assert task.artifacts
    if outcome == "failed":
        assert result.context.supplemental_executions == []
        assert result.context.supplemental_observations == {}
        assert semantic.supplemental.failed == 1
    elif outcome == "partial":
        assert len(result.context.supplemental_executions) == 1
        assert semantic.supplemental.partial == 1
    else:
        assert len(result.context.supplemental_executions) == 1
        assert semantic.supplemental.completed == 1


@pytest.mark.parametrize(
    ("reviewer_mode", "expected_max_active"),
    [("single", 1), ("multi", 2)],
)
def test_supplemental_scheduler_honors_mode_limit_and_stable_commit_order(
    git_repo: Path,
    reviewer_mode: str,
    expected_max_active: int,
) -> None:
    class State:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.barrier = threading.Barrier(2) if reviewer_mode == "multi" else None
            self.active = 0
            self.max_active = 0
            self.started: list[str] = []
            self.completed: list[str] = []

    state = State()
    semantic_calls = 0

    class SemanticAdapter:
        provider_name = "semantic-scheduler-test"

        def complete_turn(self, request):
            nonlocal semantic_calls
            semantic_calls += 1
            packet = json.loads(request.messages[0]["content"])
            candidates = sorted(packet["candidate_catalog"].items())
            candidate_ids = [candidate_id for candidate_id, _ in candidates]
            observation_ids = sorted(
                {
                    observation_id
                    for _, candidate in candidates
                    for observation_id in candidate["evidence_refs"]
                }
            )
            representative_id, representative = candidates[0]
            first_pass = semantic_calls == 1
            disagreements = []
            requests = []
            for suffix in ("a", "b"):
                disagreement_id = f"D-scheduler-{suffix}"
                disagreements.append(
                    {
                        "disagreement_id": disagreement_id,
                        "candidate_ids": candidate_ids,
                        "status": "needs_investigation" if first_pass else "resolved",
                        "issue": f"Targeted scheduler question {suffix} needs evidence.",
                        "resolution": (
                            "" if first_pass else f"Question {suffix} was resolved."
                        ),
                        "decision_refs": observation_ids,
                    }
                )
                if first_pass:
                    requests.append(
                        {
                            "disagreement_id": disagreement_id,
                            "question": f"Inspect targeted scheduler behavior {suffix}.",
                            "required_evidence": [f"Check behavior {suffix} at HEAD."],
                            "preferred_perspective": "behavioral correctness",
                            "related_candidate_ids": candidate_ids,
                            "reason_refs": observation_ids,
                        }
                    )
            return ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                provider_name=self.provider_name,
                final_text=json.dumps(
                    {
                        "canonical_groups": [
                            {
                                "member_ids": candidate_ids,
                                "representative_id": representative_id,
                                "canonical_claim": representative["claim"],
                                "rationale": "The candidates describe the same observed behavior.",
                                "supporting_refs": observation_ids,
                                "proposed_confidence": representative["confidence"],
                            }
                        ],
                        "rejections": [],
                        "disagreements": disagreements,
                        "supplemental_requests": requests,
                        "uncertainties": [],
                        "summary": "Scheduler scenario reconciled.",
                    }
                ),
            )

    class SemanticFactory:
        def create(self):
            return SemanticAdapter()

    class ReviewerAdapter:
        provider_name = "reviewer-scheduler-test"

        def __init__(self) -> None:
            self.delegate = FakeModelAdapterFactory().create()

        def complete_turn(self, request):
            response_schema = request.parameters.get("response_schema")
            if response_schema == "risk_proposal_v1":
                return ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    provider_name=self.provider_name,
                    final_text=json.dumps(
                        {
                            "level": "high",
                            "dimensions": {
                                "impact": "Scheduler behavior affects review completeness.",
                                "blast_radius": "Two targeted tasks are required.",
                                "reversibility": "The review can be resumed.",
                                "uncertainty": "A crash window is injected by the test.",
                                "verification_strength": "Checkpoint state is auditable.",
                            },
                            "reasons": ["Exercise bounded multi-task recovery."],
                            "signal_refs": [],
                            "uncertainties": [],
                            "suggested_focus": ["Verify stable supplemental scheduling."],
                        }
                    ),
                )
            if response_schema != "reviewer_assignment_result_v2":
                return self.delegate.complete_turn(request)
            content = str(request.messages[0]["content"])
            observation_id = [
                line.split(":", 1)[0]
                for line in content.splitlines()
                if line.startswith("O-")
            ][-1]
            contract_line = next(
                line for line in content.splitlines() if line.startswith("Assigned Contract:")
            )
            contracts = [
                item.strip()
                for item in contract_line.split(":", 1)[1].split(",")
                if item.strip()
            ]
            supplemental_contracts = [
                contract
                for contract in contracts
                if contract.startswith("supplemental_investigation:")
            ]
            if supplemental_contracts:
                disagreement_id = supplemental_contracts[0].split(":", 1)[1]
                with state.lock:
                    start_position = len(state.started)
                    state.started.append(disagreement_id)
                    state.active += 1
                    state.max_active = max(state.max_active, state.active)
                if state.barrier is not None:
                    state.barrier.wait(timeout=5)
                    if start_position == 0:
                        time.sleep(0.05)
                with state.lock:
                    state.completed.append(disagreement_id)
                    state.active -= 1
            payload = {
                "contract_assessments": [
                    {
                        "contract": contract,
                        "status": "covered",
                        "summary": "Inspected the targeted behavior.",
                        "evidence_refs": [observation_id],
                    }
                    for contract in supplemental_contracts
                ],
                "confirmed_findings": (
                    []
                    if supplemental_contracts
                    else [
                        {
                            "claim": "The implementation subtracts instead of adding.",
                            "severity": "high",
                            "confidence": "high",
                            "path": "app.py",
                            "line": 2,
                            "evidence_refs": [observation_id],
                            "impact": "Callers receive incorrect arithmetic results.",
                            "suggested_action": "Restore addition semantics.",
                            "verification_performed": ["Compared base and head."],
                        }
                    ]
                ),
                "rejected_hypotheses": [],
                "uncertainties": [],
                "observation_refs": [observation_id],
                "investigation_summary": "Scheduler task completed.",
                "status": "completed" if supplemental_contracts else "partial",
            }
            return ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                provider_name=self.provider_name,
                final_text=json.dumps(payload),
                raw={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
            )

    class ReviewerFactory:
        def create(self):
            return ReviewerAdapter()

    def build_factory(config):
        if config.stage_label == "semantic-reconciler":
            return SemanticFactory()
        return ReviewerFactory()

    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id=f"review-supplemental-scheduler-{reviewer_mode}",
        reviewer_mode=reviewer_mode,
        reviewer_loop="single-shot",
        risk_assessor=ModelStageConfig(mode="model", provider="fake"),
        semantic_reconciler=ModelStageConfig(mode="model", provider="fake"),
        adapter_factory_builder=build_factory,
    )

    preserved_task_id: str | None = None
    preserved_task = None
    if reviewer_mode == "single":
        real_commit = pipeline._commit_supplemental_attempt
        commit_calls = 0

        def fail_second_supplemental_commit(attempt):
            nonlocal commit_calls
            commit_calls += 1
            if commit_calls == 2:
                raise RuntimeError("injected crash after one supplemental task committed")
            return real_commit(attempt)

        pipeline._commit_supplemental_attempt = fail_second_supplemental_commit
        with pytest.raises(PipelineStageError, match="after one supplemental task"):
            pipeline.execute()
        interrupted_wave = next(
            iter(session_store.load().supplemental_waves.values())
        )
        completed = [
            (task_id, task)
            for task_id, task in interrupted_wave.tasks.items()
            if task.status.value == "completed"
        ]
        running = [
            task
            for task in interrupted_wave.tasks.values()
            if task.status.value == "running"
        ]
        assert len(completed) == 1
        assert len(running) == 1
        preserved_task_id, preserved_task = completed[0]
        pipeline._commit_supplemental_attempt = real_commit
        result = pipeline.execute(
            starting_phase=RunPhase.SUPPLEMENTAL_INVESTIGATION,
            resuming=True,
        )
    else:
        result = pipeline.execute()

    assert semantic_calls == 2
    assert state.max_active == expected_max_active
    plan = result.context.supplemental_plan
    assert plan is not None
    expected_disagreements = [
        spec.source_disagreement_id for spec in plan.tasks
    ]
    expected_task_ids = [spec.task_id for spec in plan.tasks]
    wave = next(iter(session_store.load().supplemental_waves.values()))
    assert list(wave.tasks) == expected_task_ids
    committed_task_ids = [
        result.context.supplemental_task_ids_by_trace[execution.trace_id]
        for execution in result.context.supplemental_executions
    ]
    assert committed_task_ids == expected_task_ids
    if reviewer_mode == "single":
        assert preserved_task_id is not None
        assert preserved_task is not None
        assert wave.tasks[preserved_task_id] == preserved_task
        retried_task = next(
            task
            for task_id, task in wave.tasks.items()
            if task_id != preserved_task_id
        )
        assert retried_task.attempts == 2
        assert retried_task.unknown_consumed.tasks == 1
        assert state.started == [
            expected_disagreements[0],
            expected_disagreements[1],
            expected_disagreements[1],
        ]
        assert state.completed == state.started
    else:
        assert len(state.started) == 2
        assert state.completed == list(reversed(state.started))
        completed_manifest = session_store.load()
        completed_wave = next(iter(completed_manifest.supplemental_waves.values()))
        target_id, unaffected_id = expected_task_ids
        target_before = completed_wave.tasks[target_id]
        unaffected_before = completed_wave.tasks[unaffected_id]
        observation_artifact = next(
            name
            for name in target_before.artifacts
            if name.endswith("_observations")
        )
        observation_path = (
            pipeline.context.checkpoint_store.run_dir
            / completed_manifest.artifacts[observation_artifact].path
        )
        observation_path.write_text('{"tampered":true}\n', encoding="utf-8")

        resumed = ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=pipeline.context.checkpoint_store,
            session_store=session_store,
        ).resume()

        repaired_manifest = session_store.load()
        repaired_wave = next(iter(repaired_manifest.supplemental_waves.values()))
        assert resumed.action is ResumeAction.CONTINUE_SESSION
        assert resumed.starting_phase is RunPhase.SUPPLEMENTAL_INVESTIGATION
        assert repaired_wave.tasks[unaffected_id] == unaffected_before
        assert repaired_wave.tasks[target_id].attempts == target_before.attempts + 1
        assert (
            repaired_manifest.phases[RunPhase.REVIEWERS.value].attempts
            == completed_manifest.phases[RunPhase.REVIEWERS.value].attempts
        )
        assert (
            repaired_manifest.phases[RunPhase.RECONCILIATION_ANALYSIS.value].attempts
            == completed_manifest.phases[
                RunPhase.RECONCILIATION_ANALYSIS.value
            ].attempts
        )


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


def test_two_reviewers_conflict_triggers_one_bounded_supplemental_wave(
    git_repo: Path,
) -> None:
    semantic_calls: list[dict[str, object]] = []

    class SemanticAdapter:
        provider_name = "semantic-test"

        def complete_turn(self, request):
            packet = json.loads(request.messages[0]["content"])
            semantic_calls.append(packet)
            candidates = sorted(packet["candidate_catalog"].items())
            candidate_ids = [candidate_id for candidate_id, _ in candidates]
            observation_ids = sorted(
                {
                    observation_id
                    for _, candidate in candidates
                    for observation_id in candidate["evidence_refs"]
                }
            )
            first_pass = len(semantic_calls) == 1
            representative_id, representative = max(
                candidates,
                key=lambda item: (
                    {"low": 0, "medium": 1, "high": 2, "blocker": 3}[
                        item[1]["severity"]
                    ],
                    item[0],
                ),
            )
            canonical_groups = (
                [
                    {
                        "member_ids": [candidate_id],
                        "representative_id": candidate_id,
                        "canonical_claim": candidate["claim"],
                        "rationale": "Each Reviewer made a distinct claim at this location.",
                        "supporting_refs": candidate["evidence_refs"],
                        "proposed_confidence": candidate["confidence"],
                    }
                    for candidate_id, candidate in candidates
                ]
                if first_pass
                else [
                    {
                        "member_ids": candidate_ids,
                        "representative_id": representative_id,
                        "canonical_claim": representative["claim"],
                        "rationale": "The targeted investigation resolved both views into the supported behavior.",
                        "supporting_refs": observation_ids,
                        "proposed_confidence": representative["confidence"],
                    }
                ]
            )
            disagreement = {
                "disagreement_id": "D-behavior",
                "candidate_ids": candidate_ids,
                "status": "needs_investigation" if first_pass else "resolved",
                "issue": "The changed arithmetic behavior needs a targeted check.",
                "resolution": (
                    "" if first_pass else "The targeted reviewer confirmed the cited behavior."
                ),
                "decision_refs": observation_ids,
            }
            supplemental_requests = []
            if first_pass:
                supplemental_requests.append(
                    {
                        "disagreement_id": "D-behavior",
                        "question": "Does app.py preserve addition semantics at HEAD?",
                        "required_evidence": ["Inspect the changed implementation."],
                        "preferred_perspective": "behavioral correctness",
                        "related_candidate_ids": candidate_ids,
                        "reason_refs": observation_ids,
                    }
                )
            return ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                provider_name=self.provider_name,
                final_text=json.dumps(
                    {
                        "canonical_groups": canonical_groups,
                        "rejections": [],
                        "disagreements": [disagreement],
                        "supplemental_requests": supplemental_requests,
                        "uncertainties": [],
                        "summary": "Semantic reconciliation completed.",
                    }
                ),
            )

    class SemanticFactory:
        def create(self):
            return SemanticAdapter()

    class ReviewerAdapter:
        provider_name = "reviewer-test"

        def __init__(self) -> None:
            self.delegate = FakeModelAdapterFactory().create()

        def complete_turn(self, request):
            response_schema = request.parameters.get("response_schema")
            if response_schema == "risk_proposal_v1":
                return ModelTurnResponse(
                    kind=ModelResponseKind.FINAL,
                    provider_name=self.provider_name,
                    final_text=json.dumps(
                        {
                            "level": "high",
                            "dimensions": {
                                "impact": "Arithmetic behavior affects callers.",
                                "blast_radius": "The helper can have multiple callers.",
                                "reversibility": "The revision can be reverted.",
                                "uncertainty": "Intent requires targeted confirmation.",
                                "verification_strength": "Repository evidence is available.",
                            },
                            "reasons": ["Exercise high-risk durable recovery."],
                            "signal_refs": [],
                            "uncertainties": [],
                            "suggested_focus": ["Verify arithmetic semantics."],
                        }
                    ),
                )
            if response_schema != "reviewer_assignment_result_v2":
                return self.delegate.complete_turn(request)
            content = str(request.messages[0]["content"])
            observation_id = [
                line.split(":", 1)[0]
                for line in content.splitlines()
                if line.startswith("O-")
            ][-1]
            contract_line = next(
                line for line in content.splitlines() if line.startswith("Assigned Contract:")
            )
            contracts = [
                item.strip()
                for item in contract_line.split(":", 1)[1].split(",")
                if item.strip()
            ]
            is_supplemental = any(
                contract.startswith("supplemental_investigation:")
                for contract in contracts
            )
            is_second_initial_reviewer = str(
                request.parameters.get("trace_id", "")
            ).endswith("reviewer-1")
            initial_finding = {
                "claim": (
                    "The subtraction may be an intentional arithmetic behavior change."
                    if is_second_initial_reviewer
                    else "The implementation subtracts instead of adding."
                ),
                "severity": "medium" if is_second_initial_reviewer else "high",
                "confidence": "medium" if is_second_initial_reviewer else "high",
                "path": "app.py",
                "line": 2,
                "evidence_refs": [observation_id],
                "impact": (
                    "The intended behavior is ambiguous without a targeted check."
                    if is_second_initial_reviewer
                    else "Callers receive incorrect arithmetic results."
                ),
                "suggested_action": "Verify and document the intended arithmetic semantics.",
                "verification_performed": ["Compared base and head."],
            }
            payload = {
                "contract_assessments": (
                    [
                        {
                            "contract": contract,
                            "status": "covered",
                            "summary": "Inspected the targeted behavior.",
                            "evidence_refs": [observation_id],
                        }
                        for contract in contracts
                    ]
                    if is_supplemental
                    else []
                ),
                "confirmed_findings": (
                    []
                    if is_supplemental
                    else [initial_finding]
                ),
                "rejected_hypotheses": [],
                "uncertainties": [],
                "observation_refs": [observation_id],
                "investigation_summary": "Targeted review completed.",
                "status": "completed" if is_supplemental else "partial",
            }
            return ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                provider_name=self.provider_name,
                final_text=json.dumps(payload),
                raw={"usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}},
            )

    class ReviewerFactory:
        def create(self):
            return ReviewerAdapter()

    def build_factory(config):
        if config.stage_label == "semantic-reconciler":
            return SemanticFactory()
        return ReviewerFactory()

    pipeline, session_store, _ = _pipeline(
        git_repo,
        review_id="review-semantic-supplemental",
        reviewer_loop="single-shot",
        risk_assessor=ModelStageConfig(mode="model", provider="fake"),
        semantic_reconciler=ModelStageConfig(mode="model", provider="fake"),
        adapter_factory_builder=build_factory,
    )

    real_commit = pipeline._commit_supplemental_attempt
    crashed_after_provider_return = False

    def fail_before_first_supplemental_commit(attempt):
        nonlocal crashed_after_provider_return
        if not crashed_after_provider_return:
            crashed_after_provider_return = True
            raise RuntimeError("injected crash before supplemental checkpoint commit")
        return real_commit(attempt)

    pipeline._commit_supplemental_attempt = fail_before_first_supplemental_commit
    with pytest.raises(PipelineStageError, match="injected crash"):
        pipeline.execute()

    interrupted = session_store.load()
    interrupted_wave = next(iter(interrupted.supplemental_waves.values()))
    interrupted_task = next(iter(interrupted_wave.tasks.values()))
    assert interrupted_task.status.value == "running"
    assert interrupted_task.reservation.tasks == 1

    pipeline._commit_supplemental_attempt = real_commit
    result = pipeline.execute(
        starting_phase=RunPhase.SUPPLEMENTAL_INVESTIGATION,
        resuming=True,
    )

    manifest = session_store.load()
    assert manifest.status is RunStatus.COMPLETED
    assert len(semantic_calls) == 2
    assert len(result.context.reviewer_executions) >= 2
    assert len(manifest.supplemental_waves) == 1
    wave = next(iter(manifest.supplemental_waves.values()))
    assert wave.status is PhaseStatus.COMPLETED
    assert wave.stop_reason == "resolved"
    assert [task.status.value for task in wave.tasks.values()] == ["completed"]
    completed_after_resume = next(iter(wave.tasks.values()))
    assert completed_after_resume.attempts == 2
    assert completed_after_resume.unknown_consumed.tasks == 1
    assert completed_after_resume.unknown_invocation_ids
    assert len(result.context.supplemental_executions) == 1
    assert result.context.semantic_reconciliation is not None
    assert result.context.semantic_reconciliation.supplemental.status == "completed"
    assert result.context.semantic_reconciliation.supplemental.stop_reason == "resolved"
    assert result.context.semantic_reconciliation.remaining_disagreements == ()
    assert result.context.brief is not None
    assert result.context.brief.semantic_reconciliation["supplemental"][
        "stop_reason"
    ] == "resolved"
    assert "semantic_reconciliation" in manifest.artifacts
    assert "supplemental_summary" in manifest.artifacts
    run_dir = pipeline.context.checkpoint_store.run_dir
    semantic_payload = json.loads(
        (
            run_dir / manifest.artifacts["semantic_reconciliation"].path
        ).read_text(encoding="utf-8")
    )
    completion_payload = json.loads(
        (run_dir / manifest.artifacts["completion"].path).read_text(
            encoding="utf-8"
        )
    )
    final_risk_payload = json.loads(
        (run_dir / manifest.artifacts["final_risk"].path).read_text(
            encoding="utf-8"
        )
    )
    brief_payload = json.loads(
        (run_dir / manifest.artifacts["review_brief"].path).read_text(
            encoding="utf-8"
        )
    )
    assert semantic_payload["supplemental"]["stop_reason"] == "resolved"
    assert semantic_payload["remaining_disagreements"] == []
    assert "semantic_reconciliation:accepted" in final_risk_payload["signal_refs"]
    assert brief_payload["semantic_reconciliation"] == semantic_payload
    assert (
        brief_payload["initial_and_final_risk_assessment"]["final"]
        == final_risk_payload
    )
    assert (
        brief_payload["non_binding_recommendation"]
        == completion_payload["recommendation"]
    )
    report_path = pipeline.context.checkpoint_store.run_dir / manifest.artifacts[
        "report"
    ].path
    report = report_path.read_text(encoding="utf-8")
    assert "## Semantic Reconciliation And Supplemental Investigation" in report
    assert "status=completed, stop_reason=resolved" in report
    assert pipeline.validate_completed_supplemental_state() == ()

    original_task = next(iter(wave.tasks.values()))
    result_artifact = next(
        name for name in original_task.artifacts if name.endswith("_result")
    )
    result_path = pipeline.context.checkpoint_store.run_dir / manifest.artifacts[
        result_artifact
    ].path
    result_path.write_text('{"tampered":true}', encoding="utf-8")

    resumed = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=pipeline.context.checkpoint_store,
        session_store=session_store,
    ).resume()

    repaired = session_store.load()
    repaired_wave = next(iter(repaired.supplemental_waves.values()))
    repaired_task = next(iter(repaired_wave.tasks.values()))
    assert resumed.action is ResumeAction.CONTINUE_SESSION
    assert resumed.starting_phase is RunPhase.SUPPLEMENTAL_INVESTIGATION
    assert repaired.status is RunStatus.COMPLETED
    assert repaired.phases[RunPhase.REVIEWERS.value].attempts == 1
    assert repaired.phases[RunPhase.RECONCILIATION_ANALYSIS.value].attempts == 1
    assert (
        repaired.phases[RunPhase.SUPPLEMENTAL_INVESTIGATION.value].attempts
        == manifest.phases[RunPhase.SUPPLEMENTAL_INVESTIGATION.value].attempts + 1
    )
    assert repaired_task.attempts == original_task.attempts + 1
    assert repaired_task.charged.tasks == original_task.charged.tasks + 1

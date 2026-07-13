from __future__ import annotations

import json
from pathlib import Path

from conftest import run_git

from review_agent.checkpoint import CheckpointStore
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnResponse,
)
from review_agent.models import ReviewRequest
from review_agent.pipeline import PHASE_MESSAGES, ReviewPipeline
from review_agent.revision import RevisionResolver
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import PhaseStatus, ReviewExecutionConfig, initial_session_manifest
from review_agent.session_store import SessionStore


def _pipeline(
    git_repo: Path,
    *,
    review_id: str = "review-pipeline",
    reviewer_mode: str = "single",
    reviewer_loop: str = "agent-loop",
    adapter_factory_builder=None,
) -> tuple[ReviewPipeline, SessionStore, CheckpointStore]:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
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
    assert manifest.artifacts["repository_observations"].phase is RunPhase.REPOSITORY_INTELLIGENCE
    assert manifest.artifacts["observations"].phase is RunPhase.REPORTING
    assert (checkpoint_store.run_dir / "report.md").exists()
    assert (checkpoint_store.run_dir / "observations.jsonl").exists()
    assert (checkpoint_store.run_dir / "attempts" / "preflight" / "1").is_dir()
    assert result.context.brief is not None
    assert result.context.final_risk is not None


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

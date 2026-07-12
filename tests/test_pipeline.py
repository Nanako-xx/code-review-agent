from __future__ import annotations

from pathlib import Path

from conftest import run_git

from review_agent.checkpoint import CheckpointStore
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
    return (
        ReviewPipeline(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
            request=request,
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

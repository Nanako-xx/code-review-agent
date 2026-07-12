from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import run_git

import review_agent.pipeline as pipeline_module
from review_agent.checkpoint import CheckpointStore
from review_agent.model_adapter_factory import FakeModelAdapterFactory
from review_agent.models import ReviewRequest
from review_agent.pipeline import PipelineStageError, ReviewPipeline
from review_agent.resume import ResumeAction, ResumeBlockedError, ReviewSessionResumer
from review_agent.revision import RevisionResolver
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import PhaseStatus, ReviewExecutionConfig, initial_session_manifest
from review_agent.session_store import SessionStore


def _session_pipeline(
    git_repo: Path,
    *,
    review_id: str,
    reviewer_mode: str = "single",
    reviewer_loop: str = "single-shot",
    adapter_factory_builder=None,
    symbolic_head: bool = False,
) -> tuple[ReviewPipeline, SessionStore, CheckpointStore]:
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "auth.py").write_text(
        "def check(token):\n    return token == 'ok'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")
    resolver = RevisionResolver()
    identity = resolver.repository_identity(git_repo)
    requested_head = "HEAD" if symbolic_head else head
    revisions = resolver.resolve_pair(git_repo, base, requested_head)
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
    kwargs = {}
    if adapter_factory_builder is not None:
        kwargs["adapter_factory_builder"] = adapter_factory_builder
    pipeline = ReviewPipeline(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
        request=ReviewRequest(
            repository_path=identity.canonical_path,
            base_revision=base,
            head_revision=requested_head,
            user_intent="Add authentication token check",
        ),
        **kwargs,
    )
    return pipeline, session_store, checkpoint_store


def test_resume_restarts_stale_running_phase_without_repeating_preflight(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-stale-running",
    )

    with monkeypatch.context() as scoped:
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt("simulated process exit")

        scoped.setattr(pipeline_module, "build_repository_intelligence", interrupt)
        with pytest.raises(KeyboardInterrupt, match="simulated process exit"):
            pipeline.execute()

    interrupted = session_store.load()
    preflight_hashes = {
        name: interrupted.artifacts[name].sha256
        for name in interrupted.phases["preflight"].artifacts
    }
    assert interrupted.phases["repository_intelligence"].status is PhaseStatus.RUNNING
    assert interrupted.phases["repository_intelligence"].attempts == 1

    with monkeypatch.context() as scoped:
        scoped.setattr(
            pipeline_module,
            "run_python_compile_gate",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("preflight quality gate must not rerun")
            ),
        )
        result = ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()

    completed = session_store.load()
    assert result.action is ResumeAction.CONTINUE_SESSION
    assert result.starting_phase is RunPhase.REPOSITORY_INTELLIGENCE
    assert completed.status is RunStatus.COMPLETED
    assert completed.phases["repository_intelligence"].attempts == 2
    assert {
        name: completed.artifacts[name].sha256
        for name in completed.phases["preflight"].artifacts
    } == preflight_hashes


def test_resume_reuses_completed_reviewer_and_retries_only_failed_task(
    git_repo: Path,
) -> None:
    class FailSecondFactory:
        def __init__(self) -> None:
            self.created = 0

        def create(self):
            self.created += 1
            if self.created == 2:
                raise RuntimeError("provider timeout")
            return FakeModelAdapterFactory().create()

    failing_factory = FailSecondFactory()
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-partial-reviewers",
        reviewer_mode="multi",
        adapter_factory_builder=lambda _config: failing_factory,
    )

    with pytest.raises(PipelineStageError, match="provider timeout"):
        pipeline.execute()

    failed = session_store.load()
    tasks = failed.phases["reviewers"].tasks
    assert tasks["reviewer-0"].status is PhaseStatus.COMPLETED
    assert tasks["reviewer-1"].status is PhaseStatus.FAILED
    reviewer_zero_hashes = {
        name: failed.artifacts[name].sha256
        for name in tasks["reviewer-0"].artifacts
    }

    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    completed = session_store.load()
    completed_tasks = completed.phases["reviewers"].tasks
    assert result.starting_phase is RunPhase.REVIEWERS
    assert completed.status is RunStatus.COMPLETED
    assert completed_tasks["reviewer-0"].attempts == 1
    assert completed_tasks["reviewer-1"].attempts == 2
    assert all(task.status is PhaseStatus.COMPLETED for task in completed_tasks.values())
    assert {
        name: completed.artifacts[name].sha256
        for name in completed_tasks["reviewer-0"].artifacts
    } == reviewer_zero_hashes


def test_resume_invalidates_only_tampered_reviewer_task(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-tampered-reviewer",
        reviewer_mode="multi",
    )
    pipeline.execute()
    original = session_store.load()
    tasks = original.phases["reviewers"].tasks
    reviewer_zero_hashes = {
        name: original.artifacts[name].sha256
        for name in tasks["reviewer-0"].artifacts
    }
    tampered_name = "reviewer_1_result"
    tampered_path = checkpoint_store.run_dir / original.artifacts[tampered_name].path
    tampered_path.write_text('{"tampered":true}', encoding="utf-8")

    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    repaired = session_store.load()
    repaired_tasks = repaired.phases["reviewers"].tasks
    assert result.starting_phase is RunPhase.REVIEWERS
    assert repaired_tasks["reviewer-0"].attempts == 1
    assert repaired_tasks["reviewer-1"].attempts == 2
    assert {
        name: repaired.artifacts[name].sha256
        for name in repaired_tasks["reviewer-0"].artifacts
    } == reviewer_zero_hashes
    assert repaired.artifacts[tampered_name].sha256 != original.artifacts[tampered_name].sha256 or tampered_path.read_text(encoding="utf-8") != '{"tampered":true}'


def test_completed_resume_is_audit_only(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-audit-only",
    )
    pipeline.execute()

    with monkeypatch.context() as scoped:
        def forbidden(*args, **kwargs):
            raise AssertionError("completed audit must not execute stages")

        scoped.setattr(pipeline_module, "run_python_compile_gate", forbidden)
        scoped.setattr(pipeline_module, "build_repository_intelligence", forbidden)
        scoped.setattr(pipeline_module, "run_single_reviewer", forbidden)
        result = ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()

    assert result.action is ResumeAction.AUDIT_COMPLETED
    assert result.starting_phase is None


def test_resume_blocks_revision_drift_until_incremental_child_support(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-drift-blocked",
        symbolic_head=True,
    )
    pipeline.execute()
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move requested head")

    with pytest.raises(ResumeBlockedError, match="drifted|child Session"):
        ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()


def test_resume_audits_batch_a_session_without_new_observation_partitions(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-batch-a-compatible",
    )
    pipeline.execute()
    session_path = checkpoint_store.run_dir / "session.json"
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    for artifact_name, phase_name in (
        ("repository_observations", "repository_intelligence"),
        ("reviewer_0_observations", "reviewers"),
    ):
        payload["artifacts"].pop(artifact_name)
        payload["phases"][phase_name]["artifacts"].remove(artifact_name)
    payload["phases"]["reviewers"]["tasks"] = {}
    session_path.write_text(json.dumps(payload), encoding="utf-8")

    with monkeypatch.context() as scoped:
        def forbidden(*args, **kwargs):
            raise AssertionError("Batch A audit must not rerun stages")

        scoped.setattr(pipeline_module, "run_python_compile_gate", forbidden)
        scoped.setattr(pipeline_module, "build_repository_intelligence", forbidden)
        scoped.setattr(pipeline_module, "run_single_reviewer", forbidden)
        result = ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()

    assert result.action is ResumeAction.AUDIT_COMPLETED

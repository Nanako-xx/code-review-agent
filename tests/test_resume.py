from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import run_git

import review_agent.pipeline as pipeline_module
from review_agent.checkpoint import CheckpointStore
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
    symbolic_base: bool = False,
) -> tuple[ReviewPipeline, SessionStore, CheckpointStore]:
    base = run_git(git_repo, "rev-parse", "HEAD")
    if symbolic_base:
        run_git(git_repo, "branch", "review-base", base)
    (git_repo / "auth.py").write_text(
        "def check(token):\n    return token == 'ok'\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "add auth check")
    head = run_git(git_repo, "rev-parse", "HEAD")
    resolver = RevisionResolver()
    identity = resolver.repository_identity(git_repo)
    requested_base = "review-base" if symbolic_base else base
    requested_head = "HEAD" if symbolic_head else head
    revisions = resolver.resolve_pair(git_repo, requested_base, requested_head)
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
            base_revision=requested_base,
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


def test_resume_reuses_completed_reviewer_and_retries_control_layer_failure(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-partial-reviewers",
        reviewer_mode="multi",
    )

    original_commit = pipeline._commit_reviewer_attempt
    with monkeypatch.context() as scoped:
        def fail_second_commit(attempt):
            if attempt.index == 1:
                raise RuntimeError("artifact promotion failed")
            return original_commit(attempt)

        scoped.setattr(pipeline, "_commit_reviewer_attempt", fail_second_commit)
        with pytest.raises(PipelineStageError, match="artifact promotion failed"):
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


def test_head_drift_creates_idempotent_child_with_incremental_priority_map(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-head-drift",
        symbolic_head=True,
    )
    pipeline.execute()
    parent = session_store.load()
    parent_session_bytes = session_store.session_path.read_bytes()
    parent_observation_ids = {
        json.loads(line)["observation_id"]
        for line in (checkpoint_store.run_dir / "observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move requested head")

    first = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()
    repeated = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    assert first.action is ResumeAction.CREATE_INCREMENTAL_SESSION
    assert first.change_kind.value == "head_moved"
    assert first.child_created is True
    assert repeated.new_review_id == first.new_review_id
    assert repeated.child_created is False
    assert session_store.session_path.read_bytes() == parent_session_bytes
    child_store = SessionStore(
        git_repo / ".review-agent" / "runs" / str(first.new_review_id)
    )
    child = child_store.load()
    assert child.status is RunStatus.COMPLETED
    assert child.parent_review_id == parent.review_id
    assert child.incremental_from_sha == parent.revisions.resolved_head_sha
    assert "incremental_priority" in child.artifacts
    priority = json.loads(
        (child_store.run_dir / "incremental_priority.json").read_text(encoding="utf-8")
    )
    assert priority["from_revision"] == parent.revisions.resolved_head_sha
    assert priority["to_revision"] == child.revisions.resolved_head_sha
    assert priority["changed_files"] == ["later.py"]
    assignments = json.loads(
        (child_store.run_dir / "assignments.json").read_text(encoding="utf-8")
    )
    diff_ranges = set(
        assignments["assignments"][0]["initial_context"]["diff_ranges"]
    )
    assert "incremental:later.py" in diff_ranges
    assert {"full:auth.py", "full:later.py"}.issubset(diff_ranges)
    brief = json.loads(
        (child_store.run_dir / "review_brief.json").read_text(encoding="utf-8")
    )
    assert (
        brief["change_map_and_repository_impact"]["incremental_priority"]
        == priority
    )
    assert "Incremental priority map:" in (
        child_store.run_dir / "report.md"
    ).read_text(encoding="utf-8")
    child_observation_ids = {
        json.loads(line)["observation_id"]
        for line in (child_store.run_dir / "observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert child_observation_ids.isdisjoint(parent_observation_ids)
    assert len(list((git_repo / ".review-agent" / "runs").iterdir())) == 2


def test_base_drift_creates_full_review_child_without_incremental_map(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-base-drift",
        symbolic_base=True,
    )
    pipeline.execute()
    parent = session_store.load()
    run_git(
        git_repo,
        "branch",
        "-f",
        "review-base",
        parent.revisions.resolved_head_sha,
    )

    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    assert result.change_kind.value == "base_moved"
    assert result.incremental_range is None
    assert "incremental_priority" not in result.manifest.artifacts
    assert result.manifest.original_base_sha == parent.original_base_sha
    assert result.manifest.incremental_from_sha is None


def test_head_child_missing_incremental_map_rebuilds_preflight(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-missing-incremental-map",
        symbolic_head=True,
    )
    pipeline.execute()
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move requested head")
    created = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()
    child_store = SessionStore(
        git_repo / ".review-agent" / "runs" / str(created.new_review_id)
    )
    payload = json.loads(child_store.session_path.read_text(encoding="utf-8"))
    payload["artifacts"].pop("incremental_priority")
    payload["phases"]["planning"]["artifacts"].remove(
        "incremental_priority"
    )
    child_store.session_path.write_text(json.dumps(payload), encoding="utf-8")

    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    repaired = child_store.load()
    assert result.starting_phase is RunPhase.PLANNING
    assert repaired.phases["planning"].attempts == 2
    assert "incremental_priority" in repaired.artifacts


def test_detached_revisions_do_not_drift_when_repository_head_moves(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-detached-revisions",
    )
    pipeline.execute()
    parent = session_store.load()
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move repository head")

    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    assert result.action is ResumeAction.AUDIT_COMPLETED
    assert result.manifest.review_id == parent.review_id
    assert len(list((git_repo / ".review-agent" / "runs").iterdir())) == 1


def test_running_child_without_request_restarts_preflight_attempt(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-running-child",
        symbolic_head=True,
    )
    pipeline.execute()
    parent = session_store.load()
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move requested head")

    with monkeypatch.context() as scoped:
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt("simulated child preflight interruption")

        scoped.setattr(pipeline_module, "run_python_compile_gate", interrupt)
        with pytest.raises(KeyboardInterrupt, match="child preflight interruption"):
            ReviewSessionResumer(
                repository=git_repo,
                checkpoint_store=checkpoint_store,
                session_store=session_store,
            ).resume()

    child_dirs = [
        path
        for path in (git_repo / ".review-agent" / "runs").iterdir()
        if path.name != parent.review_id
    ]
    assert len(child_dirs) == 1
    child_store = SessionStore(child_dirs[0])
    interrupted = child_store.load()
    assert interrupted.phases["preflight"].status is PhaseStatus.COMPLETED
    assert interrupted.phases["quality_gates"].status is PhaseStatus.RUNNING
    assert interrupted.phases["quality_gates"].attempts == 1
    assert "request" in interrupted.artifacts

    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    completed = child_store.load()
    assert result.action is ResumeAction.CREATE_INCREMENTAL_SESSION
    assert result.child_created is False
    assert result.new_review_id == child_dirs[0].name
    assert completed.status is RunStatus.COMPLETED
    assert completed.phases["preflight"].attempts == 1
    assert completed.phases["quality_gates"].attempts == 2


def test_failed_child_without_request_retries_preflight_attempt(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-failed-child",
        symbolic_head=True,
    )
    pipeline.execute()
    parent = session_store.load()
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move requested head")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            pipeline_module,
            "run_python_compile_gate",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("simulated child preflight failure")
            ),
        )
        with pytest.raises(PipelineStageError, match="child preflight failure"):
            ReviewSessionResumer(
                repository=git_repo,
                checkpoint_store=checkpoint_store,
                session_store=session_store,
            ).resume()

    child_dir = next(
        path
        for path in (git_repo / ".review-agent" / "runs").iterdir()
        if path.name != parent.review_id
    )
    child_store = SessionStore(child_dir)
    failed = child_store.load()
    assert failed.phases["preflight"].status is PhaseStatus.COMPLETED
    assert failed.phases["quality_gates"].status is PhaseStatus.FAILED
    assert failed.phases["quality_gates"].attempts == 1
    assert "request" in failed.artifacts

    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    assert result.child_created is False
    assert child_store.load().phases["preflight"].attempts == 1
    assert child_store.load().phases["quality_gates"].attempts == 2
    assert child_store.load().status is RunStatus.COMPLETED


def test_existing_child_with_mismatched_lineage_blocks_parent_resume(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-mismatched-child",
        symbolic_head=True,
    )
    pipeline.execute()
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move requested head")
    created = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()
    child_store = SessionStore(
        git_repo / ".review-agent" / "runs" / str(created.new_review_id)
    )
    payload = json.loads(child_store.session_path.read_text(encoding="utf-8"))
    payload["parent_review_id"] = "review-unrelated"
    child_store.session_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResumeBlockedError, match="mismatched lineage"):
        ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()


def test_existing_corrupt_child_blocks_parent_resume(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-corrupt-child",
        symbolic_head=True,
    )
    pipeline.execute()
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "move requested head")
    created = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()
    child_store = SessionStore(
        git_repo / ".review-agent" / "runs" / str(created.new_review_id)
    )
    child_store.session_path.write_text("{", encoding="utf-8")

    with pytest.raises(ResumeBlockedError, match="child Session is invalid"):
        ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()


def test_base_and_head_drift_creates_full_review_child(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-base-head-drift",
        symbolic_base=True,
        symbolic_head=True,
    )
    pipeline.execute()
    parent = session_store.load()
    (git_repo / "next.py").write_text("next_value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "next.py")
    run_git(git_repo, "commit", "-m", "move head again")
    run_git(
        git_repo,
        "branch",
        "-f",
        "review-base",
        parent.revisions.resolved_head_sha,
    )

    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    assert result.change_kind.value == "base_and_head_moved"
    assert result.incremental_range is None
    assert result.manifest.incremental_from_sha is None
    assert "incremental_priority" not in result.manifest.artifacts


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

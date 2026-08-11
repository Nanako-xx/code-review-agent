from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import run_git

import review_agent.pipeline as pipeline_module
from review_agent.artifacts import artifact_schema
from review_agent.checkpoint import CheckpointStore
from review_agent.memory_models import MemoryExecutionConfig, MemoryMode
from review_agent.models import ReviewRequest
from review_agent.pipeline import PipelineStageError, ReviewPipeline
from review_agent.resume import (
    LegacySessionUnsupportedError,
    ResumeAction,
    ResumeBlockedError,
    ReviewSessionResumer,
    diagnose_legacy_session,
    require_v6_resume_from_legacy,
)
from review_agent.revision import RevisionResolver
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    LEGACY_SESSION_SCHEMA_VERSION,
    MODEL_STAGE_SESSION_SCHEMA_VERSION,
    PREVIOUS_SESSION_SCHEMA_VERSION,
    SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
    PhaseStatus,
    ReviewExecutionConfig,
    SessionManifest,
    SupplementalBudget,
    initial_session_manifest,
    session_phases_for_schema,
)
from review_agent.session_store import SessionStore


@pytest.mark.parametrize("version", ("v1", "v2", "v3", "v4"))
def test_legacy_session_is_diagnostic_only_for_v6_product_resume(version: str) -> None:
    run_dir = Path(__file__).parent / "fixtures" / "sessions" / version

    diagnostic = diagnose_legacy_session(run_dir)

    assert diagnostic.schema_version == int(version[1:])
    assert diagnostic.status
    assert diagnostic.current_phase
    assert all(item.name and item.schema and item.path for item in diagnostic.artifacts)
    with pytest.raises(LegacySessionUnsupportedError) as caught:
        require_v6_resume_from_legacy(run_dir)
    assert caught.value.diagnostic == diagnostic
    assert "read-only" in str(caught.value)


def _session_pipeline(
    git_repo: Path,
    *,
    review_id: str,
    reviewer_mode: str = "single",
    reviewer_loop: str = "single-shot",
    adapter_factory_builder=None,
    symbolic_head: bool = False,
    symbolic_base: bool = False,
    memory_mode: MemoryMode = MemoryMode.OFF,
    project_rules: tuple[str, ...] = (),
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
                memory=MemoryExecutionConfig(
                    mode=memory_mode,
                    root_path=str((git_repo / ".memory-test").resolve()),
                ),
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
            project_rules=project_rules,
        ),
        **kwargs,
    )
    return pipeline, session_store, checkpoint_store


def _observation_payloads(
    run_dir: Path,
    manifest: SessionManifest,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    observation_paths = {
        descriptor.path
        for descriptor in manifest.artifacts.values()
        if descriptor.schema == "observation_log_jsonl_v1"
    }
    for relative_path in sorted(observation_paths):
        for line in (run_dir / relative_path).read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            assert isinstance(payload, dict)
            payloads.append(payload)
    return payloads


def _downgrade_session_schema(
    session_store: SessionStore,
    schema_version: int,
) -> None:
    payload = json.loads(session_store.session_path.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    execution = payload["execution"]
    execution.pop("memory")
    execution.pop("memory_curator")
    if schema_version < SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION:
        execution.pop("semantic_reconciler")
        execution.pop("supplemental_policy")
    if schema_version < MODEL_STAGE_SESSION_SCHEMA_VERSION:
        execution.pop("risk_assessor")
        execution.pop("portfolio_planner")
    if schema_version < SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION:
        payload.pop("supplemental_waves")
    legacy_phases = {
        phase.value for phase in session_phases_for_schema(schema_version)
    }
    payload["phases"] = {
        phase_name: checkpoint
        for phase_name, checkpoint in payload["phases"].items()
        if phase_name in legacy_phases
    }
    if schema_version < PREVIOUS_SESSION_SCHEMA_VERSION:
        for checkpoint in payload["phases"].values():
            checkpoint.pop("user_decisions", None)
    payload["artifacts"] = {
        artifact_name: descriptor
        for artifact_name, descriptor in payload["artifacts"].items()
        if descriptor["phase"] in legacy_phases
    }
    session_store.session_path.write_text(json.dumps(payload), encoding="utf-8")


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


@pytest.mark.parametrize(
    "schema_version",
    [PREVIOUS_SESSION_SCHEMA_VERSION, MODEL_STAGE_SESSION_SCHEMA_VERSION],
    ids=["session-v2", "session-v3"],
)
def test_legacy_resume_does_not_construct_or_call_semantic_reconciler(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_version: int,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id=f"review-resume-v{schema_version}",
    )
    _downgrade_session_schema(
        session_store,
        schema_version,
    )
    with monkeypatch.context() as scoped:
        def interrupt_legacy_reconciliation(*args, **kwargs):
            raise KeyboardInterrupt("interrupt legacy reconciliation")

        scoped.setattr(
            pipeline,
            "_run_reconciliation",
            interrupt_legacy_reconciliation,
        )
        with pytest.raises(KeyboardInterrupt, match="legacy reconciliation"):
            pipeline.execute()

    interrupted = session_store.load()
    assert (
        interrupted.phases[RunPhase.RECONCILIATION.value].status
        is PhaseStatus.RUNNING
    )
    assert interrupted.supplemental_waves == {}

    class SemanticReconcilerSentinel:
        def create(self, *args, **kwargs):
            raise AssertionError(
                "schema v2/v3 resume must not construct a Semantic Reconciler"
            )

        def __call__(self, *args, **kwargs):
            raise AssertionError(
                "schema v2/v3 resume must not call a Semantic Reconciler"
            )

    sentinel = SemanticReconcilerSentinel()
    original_model_stage_adapter = ReviewPipeline._model_stage_adapter

    def guarded_model_stage_adapter(self, stage, *, stage_label):
        if stage_label == "semantic-reconciler":
            return sentinel.create(stage)
        return original_model_stage_adapter(
            self,
            stage,
            stage_label=stage_label,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline_module, "MemoryStore", sentinel)
        scoped.setattr(pipeline_module, "run_local_memory_curator", sentinel)
        scoped.setattr(pipeline_module, "run_model_memory_curator", sentinel)
        scoped.setattr(
            ReviewPipeline,
            "_model_stage_adapter",
            guarded_model_stage_adapter,
        )
        scoped.setattr(pipeline_module, "reconcile_semantically", sentinel)
        result = ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()

    completed = session_store.load()
    assert result.action is ResumeAction.CONTINUE_SESSION
    assert result.starting_phase is RunPhase.RECONCILIATION
    assert completed.status is RunStatus.COMPLETED
    assert completed.schema_version == schema_version
    assert list(completed.phases) == [
        phase.value for phase in session_phases_for_schema(schema_version)
    ]
    assert completed.supplemental_waves == {}
    assert "semantic_reconciliation" not in completed.artifacts
    assert "supplemental_summary" not in completed.artifacts


@pytest.mark.parametrize(
    "schema_version",
    [LEGACY_SESSION_SCHEMA_VERSION, SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION],
    ids=["session-v1", "session-v4"],
)
def test_completed_legacy_resume_never_opens_memory_or_runs_curator(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_version: int,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id=f"review-v{schema_version}-no-memory-resume",
    )
    pipeline.execute()
    _downgrade_session_schema(
        session_store,
        schema_version,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy same-revision resume must not use Memory")

    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline_module, "MemoryStore", forbidden)
        scoped.setattr(pipeline_module, "run_local_memory_curator", forbidden)
        scoped.setattr(pipeline_module, "run_model_memory_curator", forbidden)
        result = ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()

    assert result.action is ResumeAction.AUDIT_COMPLETED
    assert result.starting_phase is None
    assert session_store.load().schema_version == schema_version


def test_legacy_revision_drift_requires_explicit_v5_memory_config(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-v4-drift-upgrade",
        symbolic_head=True,
    )
    pipeline.execute()
    _downgrade_session_schema(
        session_store,
        SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION,
    )
    parent = session_store.load()
    (git_repo / "legacy-later.py").write_text("value = 2\n", encoding="utf-8")
    run_git(git_repo, "add", "legacy-later.py")
    run_git(git_repo, "commit", "-m", "move legacy requested head")

    with pytest.raises(
        ResumeBlockedError,
        match="explicit compatible v5 ReviewExecutionConfig",
    ):
        ReviewSessionResumer(
            repository=git_repo,
            checkpoint_store=checkpoint_store,
            session_store=session_store,
        ).resume()

    legacy = parent.execution
    upgrade = ReviewExecutionConfig(
        reviewer_provider=legacy.reviewer_provider,
        reviewer_model=legacy.reviewer_model,
        reviewer_base_url=legacy.reviewer_base_url,
        reviewer_api_key_env=legacy.reviewer_api_key_env,
        reviewer_mode=legacy.reviewer_mode,
        reviewer_loop=legacy.reviewer_loop,
        non_interactive=legacy.non_interactive,
        risk_assessor=legacy.risk_assessor,
        portfolio_planner=legacy.portfolio_planner,
        semantic_reconciler=legacy.semantic_reconciler,
        supplemental_policy=legacy.supplemental_policy,
        memory=MemoryExecutionConfig(
            mode=MemoryMode.OFF,
            root_path=str((git_repo / ".memory-v5-upgrade").resolve()),
        ),
    )
    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
        upgrade_execution=upgrade,
    ).resume()

    assert result.action is ResumeAction.CREATE_INCREMENTAL_SESSION
    child_store = SessionStore(
        git_repo / ".review-agent" / "runs" / str(result.new_review_id)
    )
    child = child_store.load()
    assert child.schema_version == SESSION_SCHEMA_VERSION
    assert child.execution == upgrade
    assert child.status is RunStatus.COMPLETED
    assert "memory_snapshot" in child.artifacts
    assert "memory_candidates" in child.artifacts


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
    parent_memory_snapshot = json.loads(
        (checkpoint_store.run_dir / "memory_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    parent_observations = _observation_payloads(checkpoint_store.run_dir, parent)
    parent_observation_ids = {
        str(observation["observation_id"])
        for observation in parent_observations
    }
    parent_supplemental_plan = json.loads(
        (checkpoint_store.run_dir / "supplemental_initial_plan.json").read_text(
            encoding="utf-8"
        )
    )
    parent_wave_id = str(parent_supplemental_plan["wave_id"])
    parent_task_id = f"STASK-{'a' * 64}"
    seeded_parent_budget = SupplementalBudget(
        tasks=1,
        tool_calls=2,
        tokens=1024,
        elapsed_seconds=5,
    )
    session_store.invalidate_from(
        RunPhase.SUPPLEMENTAL_INVESTIGATION,
        "seed resumable parent supplemental state",
        "2026-07-12T00:01:00Z",
    )
    session_store.mark_phase_running(
        RunPhase.SUPPLEMENTAL_INVESTIGATION,
        "2026-07-12T00:02:00Z",
    )
    parent_budget_artifact = f"supplemental_wave_{parent_wave_id}_budget"
    parent_budget_path = checkpoint_store.run_dir / "seeded_parent_budget.json"
    parent_budget_path.write_text(
        json.dumps(
            {
                "unknown_consumed": {
                    "tasks": seeded_parent_budget.tasks,
                    "tool_calls": seeded_parent_budget.tool_calls,
                    "tokens": seeded_parent_budget.tokens,
                    "elapsed_seconds": seeded_parent_budget.elapsed_seconds,
                }
            }
        ),
        encoding="utf-8",
    )
    session_store.register_existing_artifact(
        name=parent_budget_artifact,
        relative_path=parent_budget_path.name,
        schema=artifact_schema(parent_budget_artifact),
        phase=RunPhase.SUPPLEMENTAL_INVESTIGATION,
        revision_binding=(
            f"{parent.revisions.resolved_base_sha}.."
            f"{parent.revisions.resolved_head_sha}"
        ),
        now="2026-07-12T00:03:00Z",
    )
    session_store.initialize_wave(
        parent_wave_id,
        {parent_task_id: "b" * 64},
        "2026-07-12T00:04:00Z",
        trigger_digest=str(parent_supplemental_plan["trigger_digest"]),
    )
    session_store.reserve_task_budget(
        parent_task_id,
        seeded_parent_budget,
        "2026-07-12T00:05:00Z",
    )
    session_store.mark_task_running(
        parent_task_id,
        "2026-07-12T00:06:00Z",
    )
    session_store.mark_task_unknown(
        parent_task_id,
        f"INV-{'c' * 64}",
        "provider returned before parent checkpoint commit",
        "2026-07-12T00:07:00Z",
    )
    parent = session_store.load()
    parent_wave = parent.supplemental_waves[parent_wave_id]
    assert parent_wave.tasks[parent_task_id].unknown_consumed == seeded_parent_budget
    parent_session_bytes = session_store.session_path.read_bytes()
    parent_supplemental_artifacts = {
        name
        for name, descriptor in parent.artifacts.items()
        if descriptor.phase is RunPhase.SUPPLEMENTAL_INVESTIGATION
    }
    assert parent_supplemental_artifacts == {parent_budget_artifact}
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
    child_memory_snapshot = json.loads(
        (child_store.run_dir / "memory_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    assert child_memory_snapshot["head_sha"] == child.revisions.resolved_head_sha
    assert child_memory_snapshot["snapshot_id"] != parent_memory_snapshot[
        "snapshot_id"
    ]
    assert child_memory_snapshot["memory_generation"] == 0
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
    child_observations = _observation_payloads(child_store.run_dir, child)
    child_observation_ids = {
        str(observation["observation_id"])
        for observation in child_observations
    }
    assert child_observations
    assert all(
        child.revisions.resolved_head_sha in str(observation["revision"])
        for observation in child_observations
    )
    assert all(
        parent.revisions.resolved_head_sha not in str(observation["revision"])
        for observation in child_observations
    )
    assert child_observation_ids.isdisjoint(parent_observation_ids)

    child_supplemental_plan = json.loads(
        (child_store.run_dir / "supplemental_initial_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert child.supplemental_waves == {}
    assert child_supplemental_plan["review_id"] == child.review_id
    assert child_supplemental_plan["base_sha"] == child.revisions.resolved_base_sha
    assert child_supplemental_plan["head_sha"] == child.revisions.resolved_head_sha
    assert (
        child_supplemental_plan["wave_id"]
        != parent_supplemental_plan["wave_id"]
    )
    child_supplemental_artifacts = {
        name
        for name, descriptor in child.artifacts.items()
        if descriptor.phase is RunPhase.SUPPLEMENTAL_INVESTIGATION
    }
    assert child_supplemental_artifacts.isdisjoint(
        parent_supplemental_artifacts
    )
    assert (
        f"supplemental_wave_{child_supplemental_plan['wave_id']}_summary"
        in child_supplemental_artifacts
    )
    child_semantic = json.loads(
        (child_store.run_dir / child.artifacts["semantic_reconciliation"].path)
        .read_text(encoding="utf-8")
    )
    child_budget = child_semantic["supplemental"]["budget"]
    empty_budget = {
        "tasks": 0,
        "tool_calls": 0,
        "tokens": 0,
        "elapsed_seconds": 0.0,
    }
    assert child_budget["charged"] == empty_budget
    assert child_budget["unknown_consumed"] == empty_budget
    assert child_budget["reserved"] == empty_budget
    assert child_budget["remaining"] == child_budget["limits"]


def test_read_write_drift_child_reselects_generation_without_parent_proposal(
    git_repo: Path,
) -> None:
    pipeline, session_store, checkpoint_store = _session_pipeline(
        git_repo,
        review_id="review-memory-head-drift",
        symbolic_head=True,
        memory_mode=MemoryMode.READ_WRITE,
        project_rules=("Preserve the authentication token contract.",),
    )
    pipeline.execute()
    parent = session_store.load()
    parent_snapshot = json.loads(
        (checkpoint_store.run_dir / "memory_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    parent_candidates = json.loads(
        (checkpoint_store.run_dir / "memory_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    assert parent_candidates["candidates"]
    assert "memory_outbox" in parent.artifacts

    (git_repo / "auth.py").write_text(
        "def check(token):\n    return token in {'ok', 'legacy'}\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "auth.py")
    run_git(git_repo, "commit", "-m", "move memory target head")

    result = ReviewSessionResumer(
        repository=git_repo,
        checkpoint_store=checkpoint_store,
        session_store=session_store,
    ).resume()

    child_store = SessionStore(
        git_repo / ".review-agent" / "runs" / str(result.new_review_id)
    )
    child = child_store.load()
    child_snapshot = json.loads(
        (child_store.run_dir / "memory_snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    child_candidates = json.loads(
        (child_store.run_dir / "memory_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.action is ResumeAction.CREATE_INCREMENTAL_SESSION
    assert child_snapshot["head_sha"] == child.revisions.resolved_head_sha
    assert child_snapshot["snapshot_id"] != parent_snapshot["snapshot_id"]
    assert child_snapshot["memory_generation"] >= parent_snapshot[
        "memory_generation"
    ]
    assert {
        item["candidate_id"] for item in child_candidates["candidates"]
    }.isdisjoint(
        item["candidate_id"] for item in parent_candidates["candidates"]
    )
    parent_outbox = json.loads(
        (checkpoint_store.run_dir / "memory_outbox.json").read_text(
            encoding="utf-8"
        )
    )
    child_outbox = json.loads(
        (child_store.run_dir / "memory_outbox.json").read_text(
            encoding="utf-8"
        )
    )
    assert child_outbox["review_id"] == child.review_id
    assert child_outbox["snapshot_id"] == child_snapshot["snapshot_id"]
    assert child_outbox["outbox_digest"] != parent_outbox["outbox_digest"]
    if "memory_persistence_receipt" in child.artifacts:
        child_receipt = json.loads(
            (child_store.run_dir / "memory_persistence_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        assert child_receipt["outbox_digest"] == child_outbox["outbox_digest"]
    else:
        child_brief = json.loads(
            (child_store.run_dir / "review_brief.json").read_text(
                encoding="utf-8"
            )
        )
        assert child_brief["memory_audit"]["status"]["outbox_pending"] is True
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

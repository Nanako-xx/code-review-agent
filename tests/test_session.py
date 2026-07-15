from __future__ import annotations

from dataclasses import replace

import pytest

from review_agent.memory_models import MemoryExecutionConfig, MemoryMode
from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.run_state import (
    RunPhase,
    RunStatus,
    await_user_run_state,
    initial_run_state,
    run_state_from_dict,
    run_state_to_dict,
)
from review_agent.session import (
    LEGACY_SESSION_PHASES,
    LEGACY_SESSION_SCHEMA_VERSION,
    MODEL_STAGE_SESSION_SCHEMA_VERSION,
    PREVIOUS_SESSION_SCHEMA_VERSION,
    PREVIOUS_SESSION_PHASES,
    SEMANTIC_RECONCILIATION_SESSION_PHASES,
    SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION,
    SESSION_PHASES,
    SESSION_SCHEMA_VERSION,
    ArtifactDescriptor,
    ModelStageConfig,
    PhaseCheckpoint,
    PhaseStatus,
    ReviewWaveCheckpoint,
    ReviewerTaskCheckpoint,
    ReviewExecutionConfig,
    RevisionChangeKind,
    SessionManifest,
    SupplementalBudget,
    SupplementalPolicy,
    SupplementalTaskCheckpoint,
    SupplementalTaskStatus,
    child_session_manifest,
    initial_session_manifest,
    session_phases_for_schema,
    session_manifest_from_dict,
    session_manifest_to_dict,
)


NOW = "2026-07-10T00:00:00Z"
WAVE_ID = f"W-{'1' * 64}"
TASK_ID = f"STASK-{'2' * 64}"
ASSIGNMENT_DIGEST = "3" * 64
TRIGGER_DIGEST = "4" * 64


def execution_config() -> ReviewExecutionConfig:
    return ReviewExecutionConfig(
        reviewer_provider="openai-compatible",
        reviewer_model="review-model",
        reviewer_base_url="https://provider.example/v1",
        reviewer_api_key_env="REVIEW_AGENT_API_KEY",
        reviewer_mode="multi",
        reviewer_loop="agent-loop",
        non_interactive=True,
        risk_assessor=ModelStageConfig(
            mode="model",
            provider="openai-compatible",
            model="risk-model",
            base_url="https://risk.example/v1",
            api_key_env="RISK_API_KEY",
            max_output_tokens=2048,
            max_provider_attempts=3,
            max_elapsed_seconds=45,
        ),
        portfolio_planner=ModelStageConfig(
            mode="model",
            provider="fake",
            api_key_env="PLANNER_API_KEY",
            max_output_tokens=3072,
            max_provider_attempts=2,
            max_elapsed_seconds=60,
        ),
        semantic_reconciler=ModelStageConfig(
            mode="model",
            provider="fake",
            model="reconciler-model",
            api_key_env="RECONCILER_API_KEY",
            max_output_tokens=3584,
            max_provider_attempts=4,
            max_elapsed_seconds=75,
        ),
        supplemental_policy=SupplementalPolicy.for_risk("high"),
        memory=MemoryExecutionConfig(
            mode=MemoryMode.READ_WRITE,
            root_path="C:/review-agent-memory",
            required=False,
            max_snapshot_records=1500,
            max_snapshot_bytes=4_194_304,
            max_context_records=10,
            max_query_results=6,
        ),
        memory_curator=ModelStageConfig(
            mode="model",
            provider="fake",
            model="memory-curator-model",
            api_key_env="MEMORY_CURATOR_API_KEY",
            max_output_tokens=2048,
            max_provider_attempts=3,
            max_elapsed_seconds=40,
        ),
    )


def manifest() -> SessionManifest:
    return initial_session_manifest(
        review_id="review-1",
        repository=RepositoryIdentity("C:/repo", "C:/repo/.git", None),
        revisions=ResolvedRevisions("main", "HEAD", "a" * 40, "b" * 40),
        execution=execution_config(),
        now=NOW,
    )


def artifact_descriptor(**overrides: object) -> ArtifactDescriptor:
    values = {
        "name": "request",
        "path": "artifacts/request.json",
        "sha256": "c" * 64,
        "schema": "review_request_v1",
        "phase": RunPhase.PREFLIGHT,
        "revision_binding": None,
    }
    values.update(overrides)
    return ArtifactDescriptor(**values)


def child_manifest(
    change_kind: RevisionChangeKind = RevisionChangeKind.HEAD_MOVED,
) -> SessionManifest:
    incremental_from_sha = (
        "c" * 40 if change_kind is RevisionChangeKind.HEAD_MOVED else None
    )
    return replace(
        manifest(),
        review_id="review-child",
        parent_review_id="review-parent",
        root_review_id="review-root",
        incremental_from_sha=incremental_from_sha,
        revision_change_kind=change_kind,
    )


def test_session_manifest_round_trips_with_pending_phases() -> None:
    original = manifest()

    payload = session_manifest_to_dict(original)
    loaded = session_manifest_from_dict(payload)

    assert loaded == original
    assert payload["schema_version"] == SESSION_SCHEMA_VERSION
    assert payload["status"] == "created"
    assert payload["current_phase"] == "created"
    assert payload["revisions"]["resolved_head_sha"] == "b" * 40
    assert list(payload["phases"]) == [phase.value for phase in SESSION_PHASES]
    assert payload["phases"]["preflight"]["status"] == "pending"
    assert payload["phases"]["reviewers"]["tasks"] == {}
    assert payload["phases"]["intent_resolution"]["user_decisions"] == {}
    assert payload["execution"]["risk_assessor"] == {
        "mode": "model",
        "provider": "openai-compatible",
        "model": "risk-model",
        "base_url": "https://risk.example/v1",
        "api_key_env": "RISK_API_KEY",
        "max_output_tokens": 2048,
        "max_provider_attempts": 3,
        "max_elapsed_seconds": 45.0,
    }
    assert payload["execution"]["portfolio_planner"]["provider"] == "fake"
    assert payload["execution"]["semantic_reconciler"] == {
        "mode": "model",
        "provider": "fake",
        "model": "reconciler-model",
        "base_url": None,
        "api_key_env": "RECONCILER_API_KEY",
        "max_output_tokens": 3584,
        "max_provider_attempts": 4,
        "max_elapsed_seconds": 75.0,
    }
    assert payload["execution"]["supplemental_policy"] == {
        "version": "supplemental_policy_v1",
        "risk_level": "high",
        "max_waves": 2,
        "max_tasks": 3,
        "max_tasks_per_wave": 2,
        "max_concurrency": 2,
        "max_turns_per_task": 8,
        "max_tool_calls_per_task": 16,
        "max_tokens_per_task": 49152,
        "max_total_tokens": 147456,
        "max_elapsed_seconds": 480.0,
    }
    assert payload["execution"]["memory"] == {
        "schema_version": 1,
        "mode": "read-write",
        "root_path": "C:/review-agent-memory",
        "required": False,
        "selection_policy_version": "memory_selection_v2",
        "feedback_policy_version": "feedback_aggregation_v1",
        "max_snapshot_records": 1500,
        "max_snapshot_bytes": 4_194_304,
        "max_context_records": 10,
        "max_query_results": 6,
    }
    assert payload["execution"]["memory_curator"]["provider"] == "fake"
    assert payload["supplemental_waves"] == {}
    assert SESSION_PHASES == (
        RunPhase.PREFLIGHT,
        RunPhase.QUALITY_GATES,
        RunPhase.REPOSITORY_INTELLIGENCE,
        RunPhase.MEMORY_SELECTION,
        RunPhase.INTENT_DISCOVERY,
        RunPhase.INTENT_RESOLUTION,
        RunPhase.PLANNING,
        RunPhase.REVIEWERS,
        RunPhase.RECONCILIATION_ANALYSIS,
        RunPhase.SUPPLEMENTAL_INVESTIGATION,
        RunPhase.RECONCILIATION,
        RunPhase.COMPLETION,
        RunPhase.FINAL_RISK,
        RunPhase.MEMORY_PROPOSAL,
        RunPhase.REPORTING,
    )


def test_session_manifest_round_trips_reviewer_task_checkpoints() -> None:
    original = manifest()
    reviewer_checkpoint = PhaseCheckpoint(
        status=PhaseStatus.RUNNING,
        attempts=1,
        started_at=NOW,
        tasks={
            "reviewer-0": ReviewerTaskCheckpoint(
                status=PhaseStatus.COMPLETED,
                attempts=1,
                started_at=NOW,
                completed_at="2026-07-10T00:01:00Z",
                artifacts=("reviewer_0_result",),
            ),
            "reviewer-1": ReviewerTaskCheckpoint(
                status=PhaseStatus.FAILED,
                attempts=2,
                started_at="2026-07-10T00:02:00Z",
                error="provider unavailable",
            ),
        },
    )
    original = replace(
        original,
        status=RunStatus.RUNNING,
        current_phase=RunPhase.REVIEWERS,
        phases={
            **original.phases,
            RunPhase.REVIEWERS.value: reviewer_checkpoint,
        },
    )

    payload = session_manifest_to_dict(original)
    loaded = session_manifest_from_dict(payload)

    assert loaded == original
    assert loaded.phases["reviewers"].tasks["reviewer-0"].artifacts == (
        "reviewer_0_result",
    )
    assert payload["phases"]["reviewers"]["tasks"]["reviewer-1"]["attempts"] == 2


def test_session_manifest_v5_round_trips_supplemental_wave_checkpoints() -> None:
    budget = SupplementalBudget(
        tasks=1,
        tool_calls=8,
        tokens=8192,
        elapsed_seconds=30,
    )
    task = SupplementalTaskCheckpoint(
        task_id=TASK_ID,
        assignment_digest=ASSIGNMENT_DIGEST,
        status=SupplementalTaskStatus.RUNNING,
        attempts=1,
        started_at=NOW,
        reservation=budget,
        unknown_consumed=SupplementalBudget(tokens=256, elapsed_seconds=1.5),
        unknown_invocation_ids=(f"INV-{'5' * 64}",),
    )
    wave = ReviewWaveCheckpoint(
        wave_id=WAVE_ID,
        wave_index=1,
        trigger_digest=TRIGGER_DIGEST,
        effective_policy=SupplementalPolicy.for_risk("high"),
        status=PhaseStatus.RUNNING,
        attempts=1,
        started_at=NOW,
        tasks={TASK_ID: task},
    )
    original = replace(
        manifest(),
        status=RunStatus.RUNNING,
        current_phase=RunPhase.SUPPLEMENTAL_INVESTIGATION,
        supplemental_waves={WAVE_ID: wave},
    )

    payload = session_manifest_to_dict(original)
    loaded = session_manifest_from_dict(payload)

    assert loaded == original
    assert loaded.supplemental_waves[WAVE_ID].tasks[TASK_ID] == task
    assert payload["supplemental_waves"][WAVE_ID]["tasks"][TASK_ID][
        "reservation"
    ] == {
        "tasks": 1,
        "tool_calls": 8,
        "tokens": 8192,
        "elapsed_seconds": 30.0,
    }
    assert payload["supplemental_waves"][WAVE_ID]["effective_policy"][
        "risk_level"
    ] == "high"
    with pytest.raises(TypeError):
        loaded.supplemental_waves[WAVE_ID].tasks[TASK_ID] = task


@pytest.mark.parametrize(
    ("risk_level", "expected"),
    [
        ("low", (1, 1, 1, 1, 4, 8, 16384, 16384, 120.0)),
        ("medium", (1, 2, 2, 2, 6, 12, 32768, 65536, 240.0)),
        ("high", (2, 3, 2, 2, 8, 16, 49152, 147456, 480.0)),
        ("critical", (2, 4, 2, 2, 10, 24, 65536, 262144, 600.0)),
    ],
)
def test_supplemental_policy_carries_exact_risk_upper_bounds(
    risk_level: str,
    expected: tuple[int, int, int, int, int, int, int, int, float],
) -> None:
    policy = SupplementalPolicy.for_risk(risk_level)

    assert (
        policy.max_waves,
        policy.max_tasks,
        policy.max_tasks_per_wave,
        policy.max_concurrency,
        policy.max_turns_per_task,
        policy.max_tool_calls_per_task,
        policy.max_tokens_per_task,
        policy.max_total_tokens,
        policy.max_elapsed_seconds,
    ) == expected


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_waves": 0}, "max_waves"),
        ({"max_tasks": True}, "max_tasks"),
        ({"max_tasks_per_wave": 5}, "max_tasks_per_wave"),
        ({"max_concurrency": 3}, "max_concurrency"),
        ({"max_total_tokens": 1}, "max_total_tokens"),
        ({"max_elapsed_seconds": float("nan")}, "max_elapsed_seconds"),
        ({"risk_level": "extreme"}, "risk_level"),
    ],
)
def test_supplemental_policy_strictly_rejects_invalid_limits(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "risk_level": "critical",
        "max_waves": 2,
        "max_tasks": 4,
        "max_tasks_per_wave": 2,
        "max_concurrency": 2,
        "max_turns_per_task": 10,
        "max_tool_calls_per_task": 24,
        "max_tokens_per_task": 65536,
        "max_total_tokens": 262144,
        "max_elapsed_seconds": 600,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        SupplementalPolicy(**values)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tasks": -1}, "tasks"),
        ({"tool_calls": True}, "tool_calls"),
        ({"tokens": 1.5}, "tokens"),
        ({"elapsed_seconds": float("inf")}, "elapsed_seconds"),
    ],
)
def test_supplemental_budget_requires_finite_non_negative_usage(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SupplementalBudget(**changes)


def test_supplemental_task_checkpoint_validates_state_and_budget_transitions() -> None:
    reservation = SupplementalBudget(
        tasks=1,
        tool_calls=4,
        tokens=4096,
        elapsed_seconds=20,
    )

    with pytest.raises(ValueError, match="pending supplemental task"):
        SupplementalTaskCheckpoint(
            task_id=TASK_ID,
            assignment_digest=ASSIGNMENT_DIGEST,
            reservation=reservation,
        )
    with pytest.raises(ValueError, match="reserved supplemental task"):
        SupplementalTaskCheckpoint(
            task_id=TASK_ID,
            assignment_digest=ASSIGNMENT_DIGEST,
            status=SupplementalTaskStatus.RESERVED,
        )
    with pytest.raises(ValueError, match="running supplemental task"):
        SupplementalTaskCheckpoint(
            task_id=TASK_ID,
            assignment_digest=ASSIGNMENT_DIGEST,
            status=SupplementalTaskStatus.RUNNING,
            attempts=1,
            started_at=NOW,
        )
    with pytest.raises(ValueError, match="completed supplemental task"):
        SupplementalTaskCheckpoint(
            task_id=TASK_ID,
            assignment_digest=ASSIGNMENT_DIGEST,
            status=SupplementalTaskStatus.COMPLETED,
            attempts=1,
            started_at=NOW,
            completed_at=NOW,
            charged=SupplementalBudget(tasks=1),
        )


def test_review_wave_checkpoint_rejects_key_mismatch_and_nonterminal_completion() -> None:
    task = SupplementalTaskCheckpoint(
        task_id=TASK_ID,
        assignment_digest=ASSIGNMENT_DIGEST,
    )
    with pytest.raises(ValueError, match="task registry key"):
        ReviewWaveCheckpoint(
            wave_id=WAVE_ID,
            wave_index=1,
            trigger_digest=TRIGGER_DIGEST,
            effective_policy=SupplementalPolicy.for_risk("high"),
            tasks={f"STASK-{'6' * 64}": task},
        )
    with pytest.raises(ValueError, match="nonterminal tasks"):
        ReviewWaveCheckpoint(
            wave_id=WAVE_ID,
            wave_index=1,
            trigger_digest=TRIGGER_DIGEST,
            effective_policy=SupplementalPolicy.for_risk("high"),
            status=PhaseStatus.COMPLETED,
            attempts=1,
            started_at=NOW,
            completed_at=NOW,
            artifacts=("wave_summary",),
            stop_reason="no_requests",
            tasks={TASK_ID: task},
        )


def test_session_manifest_loads_v1_layout_without_synthesizing_new_results() -> None:
    payload = session_manifest_to_dict(manifest())
    payload["schema_version"] = LEGACY_SESSION_SCHEMA_VERSION
    payload["execution"].pop("risk_assessor")
    payload["execution"].pop("portfolio_planner")
    payload["execution"].pop("semantic_reconciler")
    payload["execution"].pop("supplemental_policy")
    payload["execution"].pop("memory")
    payload["execution"].pop("memory_curator")
    payload.pop("supplemental_waves")
    payload["phases"] = {
        phase.value: payload["phases"][phase.value]
        for phase in LEGACY_SESSION_PHASES
    }
    for checkpoint in payload["phases"].values():
        checkpoint.pop("tasks")
        checkpoint.pop("user_decisions")

    loaded = session_manifest_from_dict(payload)

    assert loaded.schema_version == LEGACY_SESSION_SCHEMA_VERSION
    assert list(loaded.phases) == [phase.value for phase in LEGACY_SESSION_PHASES]
    assert "quality_gates" not in loaded.phases
    assert "intent_discovery" not in loaded.phases
    assert "intent_resolution" not in loaded.phases
    assert "planning" not in loaded.phases
    assert all(not checkpoint.tasks for checkpoint in loaded.phases.values())
    assert all(not checkpoint.user_decisions for checkpoint in loaded.phases.values())
    assert loaded.execution.risk_assessor == ModelStageConfig()
    assert loaded.execution.portfolio_planner == ModelStageConfig()
    assert loaded.execution.semantic_reconciler == ModelStageConfig()
    assert loaded.execution.supplemental_policy == SupplementalPolicy()
    assert loaded.execution.memory is None
    assert loaded.execution.memory_curator == ModelStageConfig()
    assert loaded.supplemental_waves == {}
    assert session_manifest_to_dict(loaded)["schema_version"] == 1


def test_session_manifest_loads_v2_model_stages_as_local_for_safe_resume() -> None:
    payload = session_manifest_to_dict(manifest())
    payload["schema_version"] = PREVIOUS_SESSION_SCHEMA_VERSION
    payload["execution"].pop("risk_assessor")
    payload["execution"].pop("portfolio_planner")
    payload["execution"].pop("semantic_reconciler")
    payload["execution"].pop("supplemental_policy")
    payload["execution"].pop("memory")
    payload["execution"].pop("memory_curator")
    payload.pop("supplemental_waves")
    payload["phases"] = {
        phase.value: payload["phases"][phase.value]
        for phase in PREVIOUS_SESSION_PHASES
    }

    loaded = session_manifest_from_dict(payload)
    serialized = session_manifest_to_dict(loaded)

    assert loaded.schema_version == PREVIOUS_SESSION_SCHEMA_VERSION
    assert loaded.execution.reviewer_provider == "openai-compatible"
    assert loaded.execution.reviewer_mode == "multi"
    assert loaded.execution.risk_assessor == ModelStageConfig()
    assert loaded.execution.portfolio_planner == ModelStageConfig()
    assert loaded.execution.semantic_reconciler == ModelStageConfig()
    assert loaded.execution.supplemental_policy == SupplementalPolicy()
    assert loaded.execution.memory is None
    assert loaded.execution.memory_curator == ModelStageConfig()
    assert "risk_assessor" not in serialized["execution"]
    assert "portfolio_planner" not in serialized["execution"]
    assert "semantic_reconciler" not in serialized["execution"]
    assert "supplemental_policy" not in serialized["execution"]
    assert "supplemental_waves" not in serialized


def test_session_manifest_loads_v3_with_original_layout_and_model_stages() -> None:
    payload = session_manifest_to_dict(manifest())
    payload["schema_version"] = MODEL_STAGE_SESSION_SCHEMA_VERSION
    payload["execution"].pop("semantic_reconciler")
    payload["execution"].pop("supplemental_policy")
    payload["execution"].pop("memory")
    payload["execution"].pop("memory_curator")
    payload.pop("supplemental_waves")
    payload["phases"] = {
        phase.value: payload["phases"][phase.value]
        for phase in PREVIOUS_SESSION_PHASES
    }

    loaded = session_manifest_from_dict(payload)
    serialized = session_manifest_to_dict(loaded)

    assert loaded.schema_version == MODEL_STAGE_SESSION_SCHEMA_VERSION
    assert list(loaded.phases) == [phase.value for phase in PREVIOUS_SESSION_PHASES]
    assert loaded.execution.risk_assessor.mode == "model"
    assert loaded.execution.portfolio_planner.mode == "model"
    assert loaded.execution.semantic_reconciler == ModelStageConfig()
    assert loaded.execution.supplemental_policy == SupplementalPolicy()
    assert loaded.execution.memory is None
    assert loaded.execution.memory_curator == ModelStageConfig()
    assert "reconciliation_analysis" not in loaded.phases
    assert "supplemental_investigation" not in loaded.phases
    assert "semantic_reconciler" not in serialized["execution"]
    assert "supplemental_policy" not in serialized["execution"]
    assert "supplemental_waves" not in serialized


def test_session_manifest_loads_v4_without_memory_or_phase_upgrade() -> None:
    payload = session_manifest_to_dict(manifest())
    payload["schema_version"] = SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION
    payload["execution"].pop("memory")
    payload["execution"].pop("memory_curator")
    payload["phases"] = {
        phase.value: payload["phases"][phase.value]
        for phase in SEMANTIC_RECONCILIATION_SESSION_PHASES
    }

    loaded = session_manifest_from_dict(payload)
    serialized = session_manifest_to_dict(loaded)

    assert loaded.schema_version == SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION
    assert list(loaded.phases) == [
        phase.value for phase in SEMANTIC_RECONCILIATION_SESSION_PHASES
    ]
    assert loaded.execution.semantic_reconciler.mode == "model"
    assert loaded.execution.supplemental_policy == SupplementalPolicy.for_risk("high")
    assert loaded.execution.memory is None
    assert loaded.execution.memory_curator == ModelStageConfig()
    assert RunPhase.MEMORY_SELECTION.value not in loaded.phases
    assert RunPhase.MEMORY_PROPOSAL.value not in loaded.phases
    assert "memory" not in serialized["execution"]
    assert "memory_curator" not in serialized["execution"]


@pytest.mark.parametrize(
    ("schema_version", "expected_phases"),
    [
        (LEGACY_SESSION_SCHEMA_VERSION, LEGACY_SESSION_PHASES),
        (PREVIOUS_SESSION_SCHEMA_VERSION, PREVIOUS_SESSION_PHASES),
        (MODEL_STAGE_SESSION_SCHEMA_VERSION, PREVIOUS_SESSION_PHASES),
        (
            SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION,
            SEMANTIC_RECONCILIATION_SESSION_PHASES,
        ),
    ],
)
def test_legacy_session_phase_layouts_remain_exact(
    schema_version: int,
    expected_phases: tuple[RunPhase, ...],
) -> None:
    assert session_phases_for_schema(schema_version) == expected_phases
    assert RunPhase.MEMORY_SELECTION not in expected_phases
    assert RunPhase.MEMORY_PROPOSAL not in expected_phases


def test_session_v5_requires_fixed_memory_execution_config() -> None:
    original = manifest()

    with pytest.raises(ValueError, match="MemoryExecutionConfig"):
        replace(
            original,
            execution=replace(original.execution, memory=None),
        )


def test_session_v5_strictly_hydrates_embedded_memory_config() -> None:
    payload = session_manifest_to_dict(manifest())
    payload["execution"]["memory"]["unexpected"] = True

    with pytest.raises(ValueError, match="unexpected"):
        session_manifest_from_dict(payload)

    payload = session_manifest_to_dict(manifest())
    payload["execution"]["memory"]["mode"] = "off"
    payload["execution"]["memory"]["required"] = True

    with pytest.raises(ValueError, match="required=true"):
        session_manifest_from_dict(payload)


def test_legacy_session_rejects_v5_memory_execution_fields() -> None:
    payload = session_manifest_to_dict(manifest())
    payload["schema_version"] = SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION
    payload["phases"] = {
        phase.value: payload["phases"][phase.value]
        for phase in SEMANTIC_RECONCILIATION_SESSION_PHASES
    }

    with pytest.raises(ValueError, match="memory"):
        session_manifest_from_dict(payload)


def test_awaiting_user_manifest_round_trips_with_committed_artifacts() -> None:
    original = manifest()
    descriptors = {
        name: artifact_descriptor(
            name=name,
            path=f"{name}.json",
            schema=f"{name}_v1",
            phase=RunPhase.INTENT_RESOLUTION,
            revision_binding=f"{'a' * 40}..{'b' * 40}",
        )
        for name in ("intent_candidates", "intent_questions")
    }
    awaiting = PhaseCheckpoint(
        status=PhaseStatus.AWAITING_USER,
        attempts=1,
        started_at=NOW,
        artifacts=tuple(descriptors),
    )
    original = replace(
        original,
        status=RunStatus.AWAITING_USER,
        current_phase=RunPhase.INTENT_RESOLUTION,
        phases={
            **original.phases,
            RunPhase.INTENT_RESOLUTION.value: awaiting,
        },
        artifacts=descriptors,
    )

    loaded = session_manifest_from_dict(session_manifest_to_dict(original))

    assert loaded == original
    assert loaded.status is RunStatus.AWAITING_USER
    assert loaded.phases["intent_resolution"].completed_at is None
    assert loaded.phases["intent_resolution"].artifacts == (
        "intent_candidates",
        "intent_questions",
    )


def test_awaiting_user_invariants_reject_wrong_phase_or_session_status() -> None:
    original = manifest()
    descriptor = artifact_descriptor(
        name="intent_questions",
        path="intent_questions.json",
        schema="intent_questions_v1",
        phase=RunPhase.INTENT_RESOLUTION,
        revision_binding=f"{'a' * 40}..{'b' * 40}",
    )
    awaiting = PhaseCheckpoint(
        status=PhaseStatus.AWAITING_USER,
        attempts=1,
        started_at=NOW,
        artifacts=("intent_questions",),
    )

    with pytest.raises(ValueError, match="only on intent_resolution"):
        replace(
            original,
            status=RunStatus.AWAITING_USER,
            current_phase=RunPhase.INTENT_RESOLUTION,
            phases={**original.phases, "planning": awaiting},
            artifacts={"intent_questions": descriptor},
        )
    with pytest.raises(ValueError, match="requires the Session status"):
        replace(
            original,
            status=RunStatus.RUNNING,
            current_phase=RunPhase.INTENT_RESOLUTION,
            phases={**original.phases, "intent_resolution": awaiting},
            artifacts={"intent_questions": descriptor},
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"attempts": 0},
        {"started_at": None},
        {"completed_at": "2026-07-10T00:01:00Z"},
        {"artifacts": ()},
        {"error": "not a wait"},
    ],
)
def test_awaiting_user_checkpoint_requires_nonterminal_committed_state(
    changes: dict[str, object],
) -> None:
    values = {
        "status": PhaseStatus.AWAITING_USER,
        "attempts": 1,
        "started_at": NOW,
        "completed_at": None,
        "artifacts": ("intent_questions",),
        "error": None,
    }
    values.update(changes)

    with pytest.raises(ValueError, match="awaiting_user phase"):
        PhaseCheckpoint(**values)


def test_reviewer_task_explicitly_rejects_awaiting_user() -> None:
    with pytest.raises(ValueError, match="reviewer task cannot"):
        ReviewerTaskCheckpoint(status=PhaseStatus.AWAITING_USER)


def test_run_state_round_trips_awaiting_user_and_new_phases() -> None:
    state = initial_run_state(
        review_id="review-1",
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
    )
    waiting = await_user_run_state(
        state,
        message="Clarification required",
        artifacts={"intent_questions": "intent_questions.json"},
    )

    assert run_state_from_dict(run_state_to_dict(waiting)) == waiting
    assert waiting.status is RunStatus.AWAITING_USER
    assert waiting.phase is RunPhase.INTENT_RESOLUTION

    payload = run_state_to_dict(waiting)
    payload["phase"] = RunPhase.PLANNING.value
    with pytest.raises(ValueError, match="only during intent_resolution"):
        run_state_from_dict(payload)


def test_session_manifest_rejects_reviewer_tasks_on_other_phases() -> None:
    original = manifest()
    invalid = PhaseCheckpoint(
        tasks={"reviewer-0": ReviewerTaskCheckpoint()},
    )

    with pytest.raises(ValueError, match="only on the reviewers phase"):
        replace(
            original,
            phases={**original.phases, RunPhase.PREFLIGHT.value: invalid},
        )


def test_completed_reviewer_phase_rejects_incomplete_task_checkpoint() -> None:
    original = manifest()
    invalid = PhaseCheckpoint(
        status=PhaseStatus.COMPLETED,
        attempts=1,
        started_at=NOW,
        completed_at="2026-07-10T00:01:00Z",
        tasks={"reviewer-0": ReviewerTaskCheckpoint()},
    )

    with pytest.raises(ValueError, match="incomplete tasks"):
        replace(
            original,
            phases={**original.phases, RunPhase.REVIEWERS.value: invalid},
        )


@pytest.mark.parametrize("task_name", ["reviewer", "reviewer--1", "reviewer-x", 0])
def test_phase_checkpoint_rejects_invalid_reviewer_task_names(task_name) -> None:
    with pytest.raises(ValueError, match="reviewer-<index>"):
        PhaseCheckpoint(tasks={task_name: ReviewerTaskCheckpoint()})


def test_session_manifest_explicitly_round_trips_nested_models_and_enums() -> None:
    checkpoint = PhaseCheckpoint(
        status=PhaseStatus.COMPLETED,
        attempts=1,
        started_at=NOW,
        completed_at="2026-07-10T00:01:00Z",
        artifacts=["request"],
        error=None,
    )
    descriptor = ArtifactDescriptor(
        name="request",
        path="request.json",
        sha256="c" * 64,
        schema="review_request_v1",
        phase=RunPhase.PREFLIGHT,
        revision_binding=None,
    )
    original = manifest()
    original = replace(
        original,
        status=RunStatus.RUNNING,
        current_phase=RunPhase.REPOSITORY_INTELLIGENCE,
        last_successful_phase=RunPhase.PREFLIGHT,
        phases={**original.phases, RunPhase.PREFLIGHT.value: checkpoint},
        artifacts={"request": descriptor},
    )

    payload = session_manifest_to_dict(original)
    loaded = session_manifest_from_dict(payload)

    assert payload["phases"]["preflight"]["status"] == "completed"
    assert payload["artifacts"]["request"]["phase"] == "preflight"
    assert loaded == original
    assert loaded.status is RunStatus.RUNNING
    assert loaded.current_phase is RunPhase.REPOSITORY_INTELLIGENCE
    assert loaded.last_successful_phase is RunPhase.PREFLIGHT
    assert loaded.phases["preflight"].status is PhaseStatus.COMPLETED
    assert loaded.artifacts["request"].phase is RunPhase.PREFLIGHT


def test_session_manifest_never_serializes_api_key_values() -> None:
    payload = session_manifest_to_dict(manifest())
    execution = payload["execution"]

    assert execution == {
        "reviewer_provider": "openai-compatible",
        "reviewer_model": "review-model",
        "reviewer_base_url": "https://provider.example/v1",
        "reviewer_api_key_env": "REVIEW_AGENT_API_KEY",
        "reviewer_mode": "multi",
        "reviewer_loop": "agent-loop",
        "non_interactive": True,
        "risk_assessor": {
            "mode": "model",
            "provider": "openai-compatible",
            "model": "risk-model",
            "base_url": "https://risk.example/v1",
            "api_key_env": "RISK_API_KEY",
            "max_output_tokens": 2048,
            "max_provider_attempts": 3,
            "max_elapsed_seconds": 45.0,
        },
        "portfolio_planner": {
            "mode": "model",
            "provider": "fake",
            "model": None,
            "base_url": None,
            "api_key_env": "PLANNER_API_KEY",
            "max_output_tokens": 3072,
            "max_provider_attempts": 2,
            "max_elapsed_seconds": 60.0,
        },
        "semantic_reconciler": {
            "mode": "model",
            "provider": "fake",
            "model": "reconciler-model",
            "base_url": None,
            "api_key_env": "RECONCILER_API_KEY",
            "max_output_tokens": 3584,
            "max_provider_attempts": 4,
            "max_elapsed_seconds": 75.0,
        },
        "supplemental_policy": {
            "version": "supplemental_policy_v1",
            "risk_level": "high",
            "max_waves": 2,
            "max_tasks": 3,
            "max_tasks_per_wave": 2,
            "max_concurrency": 2,
            "max_turns_per_task": 8,
            "max_tool_calls_per_task": 16,
            "max_tokens_per_task": 49152,
            "max_total_tokens": 147456,
            "max_elapsed_seconds": 480.0,
        },
        "memory": {
            "schema_version": 1,
            "mode": "read-write",
            "root_path": "C:/review-agent-memory",
            "required": False,
            "selection_policy_version": "memory_selection_v2",
            "feedback_policy_version": "feedback_aggregation_v1",
            "max_snapshot_records": 1500,
            "max_snapshot_bytes": 4_194_304,
            "max_context_records": 10,
            "max_query_results": 6,
        },
        "memory_curator": {
            "mode": "model",
            "provider": "fake",
            "model": "memory-curator-model",
            "base_url": None,
            "api_key_env": "MEMORY_CURATOR_API_KEY",
            "max_output_tokens": 2048,
            "max_provider_attempts": 3,
            "max_elapsed_seconds": 40.0,
        },
    }
    assert "api_key" not in execution
    assert "authorization" not in str(payload).casefold()


@pytest.mark.parametrize("secret_field", ["api_key", "Authorization"])
def test_session_manifest_rejects_secret_execution_fields(secret_field: str) -> None:
    payload = session_manifest_to_dict(manifest())
    payload["execution"][secret_field] = "secret-value"

    with pytest.raises(ValueError, match=secret_field.casefold()):
        session_manifest_from_dict(payload)


@pytest.mark.parametrize(
    "stage",
    [
        "risk_assessor",
        "portfolio_planner",
        "semantic_reconciler",
        "memory_curator",
    ],
)
def test_session_manifest_rejects_secret_model_stage_fields(stage: str) -> None:
    payload = session_manifest_to_dict(manifest())
    payload["execution"][stage]["api_key"] = "secret-value"

    with pytest.raises(ValueError, match="api_key"):
        session_manifest_from_dict(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"mode": "automatic"}, "mode"),
        ({"provider": "other"}, "provider"),
        ({"mode": "local", "provider": "fake"}, "mode=local"),
        ({"mode": "model", "provider": "none"}, "mode=model"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"max_provider_attempts": True}, "max_provider_attempts"),
        ({"max_elapsed_seconds": float("inf")}, "max_elapsed_seconds"),
        ({"api_key_env": "sk-secret"}, "api_key_env"),
    ],
)
def test_model_stage_config_rejects_invalid_values(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "mode": "model",
        "provider": "fake",
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        ModelStageConfig(**values)


def test_model_stage_config_rejects_incomplete_openai_configuration() -> None:
    with pytest.raises(ValueError, match="model"):
        ModelStageConfig(
            mode="model",
            provider="openai-compatible",
            base_url="https://provider.example/v1",
        )
    with pytest.raises(ValueError, match="base_url"):
        ModelStageConfig(
            mode="model",
            provider="openai-compatible",
            model="stage-model",
        )


def test_execution_config_rejects_api_key_value_instead_of_environment_name() -> None:
    with pytest.raises(ValueError, match="reviewer_api_key_env"):
        ReviewExecutionConfig(
            reviewer_provider="openai-compatible",
            reviewer_model="review-model",
            reviewer_base_url="https://provider.example/v1",
            reviewer_api_key_env="sk-secret-value",
            reviewer_mode="multi",
            reviewer_loop="agent-loop",
            non_interactive=True,
        )


def test_execution_config_rejects_credentials_embedded_in_base_url() -> None:
    with pytest.raises(ValueError, match="reviewer_base_url"):
        ReviewExecutionConfig(
            reviewer_provider="openai-compatible",
            reviewer_model="review-model",
            reviewer_base_url="https://user:secret@provider.example/v1",
            reviewer_api_key_env="REVIEW_AGENT_API_KEY",
            reviewer_mode="multi",
            reviewer_loop="agent-loop",
            non_interactive=True,
        )


def test_session_manifest_rejects_unsupported_schema_version() -> None:
    payload = session_manifest_to_dict(manifest())
    payload["schema_version"] = SESSION_SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema_version"):
        session_manifest_from_dict(payload)


@pytest.mark.parametrize(
    ("path", "missing_field"),
    [
        ((), "review_id"),
        ((), "last_successful_phase"),
        ((), "errors"),
        (("repository",), "origin_url"),
        (("revisions",), "requested_base"),
        (("revisions",), "incremental_from_sha"),
        (("execution",), "reviewer_model"),
        (("phases", "preflight"), "error"),
    ],
)
def test_session_manifest_rejects_missing_semantic_fields(
    path: tuple[str, ...],
    missing_field: str,
) -> None:
    payload = session_manifest_to_dict(manifest())
    target = payload
    for component in path:
        target = target[component]
    del target[missing_field]

    with pytest.raises(ValueError, match=missing_field):
        session_manifest_from_dict(payload)


def test_initial_session_has_initial_lineage_and_pending_phases() -> None:
    initial = manifest()

    assert initial.parent_review_id is None
    assert initial.root_review_id == "review-1"
    assert initial.original_base_sha == "a" * 40
    assert initial.incremental_from_sha is None
    assert initial.revision_change_kind is RevisionChangeKind.INITIAL
    assert initial.status is RunStatus.CREATED
    assert initial.current_phase is RunPhase.CREATED
    assert initial.last_successful_phase is None
    assert list(initial.phases) == [phase.value for phase in SESSION_PHASES]
    assert all(item.status is PhaseStatus.PENDING for item in initial.phases.values())
    assert all(item.attempts == 0 for item in initial.phases.values())


@pytest.mark.parametrize(
    ("change_kind", "incremental_from_sha"),
    [
        (RevisionChangeKind.HEAD_MOVED, "b" * 40),
        (RevisionChangeKind.BASE_MOVED, None),
        (RevisionChangeKind.BASE_AND_HEAD_MOVED, None),
    ],
)
def test_child_session_manifest_starts_isolated_and_preserves_root_lineage(
    change_kind: RevisionChangeKind,
    incremental_from_sha: str | None,
) -> None:
    parent = manifest()
    child = child_session_manifest(
        review_id="review-child",
        parent=parent,
        repository=parent.repository,
        revisions=ResolvedRevisions("main", "HEAD", "c" * 40, "d" * 40),
        change_kind=change_kind,
        now="2026-07-12T01:00:00Z",
    )

    assert child.parent_review_id == parent.review_id
    assert child.root_review_id == parent.root_review_id
    assert child.original_base_sha == parent.original_base_sha
    assert child.incremental_from_sha == incremental_from_sha
    assert child.execution == parent.execution
    assert child.artifacts == {}
    assert child.errors == ()
    assert child.status is RunStatus.CREATED
    assert all(
        checkpoint.status is PhaseStatus.PENDING
        for checkpoint in child.phases.values()
    )


def test_revision_drift_child_requires_explicit_v5_config_for_legacy_parent() -> None:
    payload = session_manifest_to_dict(manifest())
    payload["schema_version"] = SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION
    payload["execution"].pop("memory")
    payload["execution"].pop("memory_curator")
    payload["phases"] = {
        phase.value: payload["phases"][phase.value]
        for phase in SEMANTIC_RECONCILIATION_SESSION_PHASES
    }
    legacy_parent = session_manifest_from_dict(payload)
    revisions = ResolvedRevisions("main", "HEAD", "c" * 40, "d" * 40)

    with pytest.raises(ValueError, match="explicit v5 execution config"):
        child_session_manifest(
            review_id="review-child",
            parent=legacy_parent,
            repository=legacy_parent.repository,
            revisions=revisions,
            change_kind=RevisionChangeKind.BASE_AND_HEAD_MOVED,
            now="2026-07-12T01:00:00Z",
        )
    with pytest.raises(ValueError, match="preserve.*non-memory execution config"):
        child_session_manifest(
            review_id="review-child",
            parent=legacy_parent,
            repository=legacy_parent.repository,
            revisions=revisions,
            change_kind=RevisionChangeKind.BASE_AND_HEAD_MOVED,
            now="2026-07-12T01:00:00Z",
            execution=replace(execution_config(), reviewer_mode="single"),
        )

    child = child_session_manifest(
        review_id="review-child",
        parent=legacy_parent,
        repository=legacy_parent.repository,
        revisions=revisions,
        change_kind=RevisionChangeKind.BASE_AND_HEAD_MOVED,
        now="2026-07-12T01:00:00Z",
        execution=execution_config(),
    )

    assert child.schema_version == SESSION_SCHEMA_VERSION
    assert child.execution.memory == execution_config().memory
    assert child.artifacts == {}
    assert child.supplemental_waves == {}


def test_v5_revision_drift_child_cannot_change_pinned_execution_config() -> None:
    parent = manifest()
    changed_execution = replace(
        parent.execution,
        memory=MemoryExecutionConfig(
            mode=MemoryMode.READ,
            root_path="C:/review-agent-memory",
        ),
    )

    with pytest.raises(ValueError, match="preserve.*fixed execution config"):
        child_session_manifest(
            review_id="review-child",
            parent=parent,
            repository=parent.repository,
            revisions=ResolvedRevisions("main", "HEAD", "c" * 40, "d" * 40),
            change_kind=RevisionChangeKind.BASE_AND_HEAD_MOVED,
            now="2026-07-12T01:00:00Z",
            execution=changed_execution,
        )


@pytest.mark.parametrize(
    "model_type",
    [
        ModelStageConfig,
        ReviewExecutionConfig,
        SupplementalPolicy,
        SupplementalBudget,
        ArtifactDescriptor,
        PhaseCheckpoint,
        ReviewWaveCheckpoint,
        SupplementalTaskCheckpoint,
        SessionManifest,
    ],
)
def test_session_dataclasses_are_frozen(model_type: type[object]) -> None:
    assert model_type.__dataclass_params__.frozen is True


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("resolved_base_sha", "a" * 39),
        ("resolved_head_sha", "not-an-object-id"),
        ("original_base_sha", "g" * 40),
        ("incremental_from_sha", "z" * 40),
    ],
)
def test_session_manifest_constructor_rejects_invalid_object_ids(
    field_name: str,
    invalid_value: str,
) -> None:
    original = child_manifest() if field_name == "incremental_from_sha" else manifest()
    if field_name.startswith("resolved_"):
        revisions = replace(original.revisions, **{field_name: invalid_value})
        changes = {"revisions": revisions}
    else:
        changes = {field_name: invalid_value}

    with pytest.raises(ValueError, match=field_name):
        replace(original, **changes)


def test_session_manifest_constructor_rejects_mixed_object_id_formats() -> None:
    original = manifest()
    revisions = replace(original.revisions, resolved_head_sha="b" * 64)

    with pytest.raises(ValueError, match="same object ID format"):
        replace(original, revisions=revisions)


def test_session_manifest_accepts_consistent_sha256_git_object_ids() -> None:
    original = manifest()
    sha256_manifest = replace(
        original,
        revisions=ResolvedRevisions("main", "HEAD", "a" * 64, "b" * 64),
        original_base_sha="a" * 64,
    )

    assert session_manifest_from_dict(session_manifest_to_dict(sha256_manifest)) == (
        sha256_manifest
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("resolved_base_sha", "a" * 39),
        ("resolved_head_sha", "b" * 64),
        ("original_base_sha", "not-an-object-id"),
        ("incremental_from_sha", "c" * 64),
    ],
)
def test_session_manifest_from_dict_rejects_invalid_or_mixed_object_ids(
    field_name: str,
    invalid_value: str,
) -> None:
    payload = session_manifest_to_dict(
        child_manifest() if field_name == "incremental_from_sha" else manifest()
    )
    payload["revisions"][field_name] = invalid_value

    expected_message = (
        field_name
        if len(invalid_value) not in {40, 64}
        else "same object ID format"
    )
    with pytest.raises(ValueError, match=expected_message):
        session_manifest_from_dict(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"parent_review_id": "review-parent"}, "parent_review_id"),
        ({"root_review_id": "another-root"}, "root_review_id"),
        ({"original_base_sha": "d" * 40}, "original_base_sha"),
        ({"incremental_from_sha": "c" * 40}, "incremental_from_sha"),
    ],
)
def test_initial_manifest_constructor_rejects_invalid_lineage(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(manifest(), **changes)


def test_initial_manifest_from_dict_rejects_invalid_lineage() -> None:
    payload = session_manifest_to_dict(manifest())
    payload["parent_review_id"] = "review-parent"

    with pytest.raises(ValueError, match="parent_review_id"):
        session_manifest_from_dict(payload)


def test_head_moved_child_round_trips_with_incremental_lineage() -> None:
    original = child_manifest()

    loaded = session_manifest_from_dict(session_manifest_to_dict(original))

    assert loaded == original
    assert loaded.parent_review_id == "review-parent"
    assert loaded.root_review_id == "review-root"
    assert loaded.incremental_from_sha == "c" * 40


def test_head_moved_grandchild_after_base_drift_preserves_root_original_base() -> None:
    root = manifest()
    base_moved_child = replace(
        root,
        review_id="review-base-moved",
        parent_review_id=root.review_id,
        root_review_id=root.review_id,
        revisions=ResolvedRevisions("new-base", "HEAD", "d" * 40, "e" * 40),
        original_base_sha=root.original_base_sha,
        incremental_from_sha=None,
        revision_change_kind=RevisionChangeKind.BASE_MOVED,
    )
    head_moved_grandchild = replace(
        base_moved_child,
        review_id="review-head-moved",
        parent_review_id=base_moved_child.review_id,
        revisions=replace(
            base_moved_child.revisions,
            resolved_head_sha="f" * 40,
        ),
        incremental_from_sha=base_moved_child.revisions.resolved_head_sha,
        revision_change_kind=RevisionChangeKind.HEAD_MOVED,
    )

    loaded = session_manifest_from_dict(
        session_manifest_to_dict(head_moved_grandchild)
    )

    assert loaded == head_moved_grandchild
    assert loaded.original_base_sha == root.revisions.resolved_base_sha
    assert (
        loaded.revisions.resolved_base_sha
        == base_moved_child.revisions.resolved_base_sha
    )
    assert loaded.incremental_from_sha == base_moved_child.revisions.resolved_head_sha


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"parent_review_id": None}, "parent_review_id"),
        ({"parent_review_id": "review-child"}, "parent_review_id"),
        ({"root_review_id": "review-child"}, "root_review_id"),
        ({"incremental_from_sha": None}, "incremental_from_sha"),
    ],
)
def test_head_moved_child_constructor_rejects_invalid_lineage(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(child_manifest(), **changes)


def test_head_moved_child_from_dict_requires_incremental_revision() -> None:
    payload = session_manifest_to_dict(child_manifest())
    payload["revisions"]["incremental_from_sha"] = None

    with pytest.raises(ValueError, match="incremental_from_sha"):
        session_manifest_from_dict(payload)


def test_child_manifest_from_dict_rejects_self_referencing_root() -> None:
    payload = session_manifest_to_dict(child_manifest())
    payload["root_review_id"] = "review-child"

    with pytest.raises(ValueError, match="root_review_id"):
        session_manifest_from_dict(payload)


@pytest.mark.parametrize(
    "change_kind",
    [RevisionChangeKind.BASE_MOVED, RevisionChangeKind.BASE_AND_HEAD_MOVED],
)
def test_base_drift_child_rejects_incremental_revision(
    change_kind: RevisionChangeKind,
) -> None:
    base_drift = child_manifest(change_kind)

    with pytest.raises(ValueError, match="incremental_from_sha"):
        replace(base_drift, incremental_from_sha="c" * 40)

    payload = session_manifest_to_dict(base_drift)
    payload["revisions"]["incremental_from_sha"] = "c" * 40
    with pytest.raises(ValueError, match="incremental_from_sha"):
        session_manifest_from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("name", ""),
        ("name", "   "),
        ("schema", ""),
        ("path", ""),
        ("path", "/absolute/request.json"),
        ("path", "C:/absolute/request.json"),
        ("path", "../request.json"),
        ("path", "artifacts/../request.json"),
        ("path", "./request.json"),
        ("path", "artifacts//request.json"),
        ("path", "artifacts\\request.json"),
        ("sha256", "c" * 63),
        ("sha256", "z" * 64),
        ("phase", RunPhase.CREATED),
    ],
)
def test_artifact_descriptor_constructor_rejects_invalid_shape(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        artifact_descriptor(**{field_name: invalid_value})


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("name", ""),
        ("schema", ""),
        ("path", "../request.json"),
        ("path", "./request.json"),
        ("sha256", "c" * 63),
        ("phase", "created"),
    ],
)
def test_session_manifest_from_dict_rejects_invalid_artifact_shape(
    field_name: str,
    invalid_value: object,
) -> None:
    original = replace(
        manifest(),
        artifacts={"request": artifact_descriptor()},
    )
    payload = session_manifest_to_dict(original)
    payload["artifacts"]["request"][field_name] = invalid_value

    with pytest.raises(ValueError, match=field_name):
        session_manifest_from_dict(payload)


def test_session_manifest_constructor_rejects_artifact_registry_name_mismatch(
) -> None:
    with pytest.raises(ValueError, match="registry key"):
        replace(manifest(), artifacts={"different": artifact_descriptor()})


def test_session_manifest_from_dict_rejects_artifact_registry_name_mismatch() -> None:
    original = replace(manifest(), artifacts={"request": artifact_descriptor()})
    payload = session_manifest_to_dict(original)
    payload["artifacts"]["different"] = payload["artifacts"].pop("request")

    with pytest.raises(ValueError, match="registry key"):
        session_manifest_from_dict(payload)


def test_phase_checkpoint_defensively_freezes_artifact_names() -> None:
    source = ["request"]

    checkpoint = PhaseCheckpoint(artifacts=source)
    source.append("risk")

    assert checkpoint.artifacts == ("request",)
    assert isinstance(checkpoint.artifacts, tuple)


def test_session_manifest_defensively_freezes_internal_collections() -> None:
    original = manifest()
    phases = dict(original.phases)
    artifacts = {"request": artifact_descriptor()}
    errors = ["first"]

    frozen = replace(original, phases=phases, artifacts=artifacts, errors=errors)
    phases.clear()
    artifacts.clear()
    errors.append("second")

    assert list(frozen.phases) == [phase.value for phase in SESSION_PHASES]
    assert list(frozen.artifacts) == ["request"]
    assert frozen.errors == ("first",)
    with pytest.raises(TypeError):
        frozen.phases["preflight"] = PhaseCheckpoint()
    with pytest.raises(TypeError):
        frozen.artifacts["other"] = artifact_descriptor(name="other")


def test_immutable_manifest_still_round_trips_and_supports_replace() -> None:
    original = replace(
        manifest(),
        artifacts={"request": artifact_descriptor()},
        errors=["recoverable"],
    )

    updated = replace(original, status=RunStatus.RUNNING)
    payload = session_manifest_to_dict(updated)
    loaded = session_manifest_from_dict(payload)

    assert isinstance(payload["phases"], dict)
    assert isinstance(payload["artifacts"], dict)
    assert isinstance(payload["errors"], list)
    assert loaded == updated
    assert loaded.errors == ("recoverable",)

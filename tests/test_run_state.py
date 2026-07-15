import pytest

from review_agent.run_state import (
    RunPhase,
    RunState,
    RunStatus,
    advance_run_state,
    await_user_run_state,
    fail_run_state,
    initial_run_state,
    run_state_from_dict,
    run_state_to_dict,
)


@pytest.mark.parametrize(
    ("phase", "wire_value"),
    [
        (RunPhase.MEMORY_SELECTION, "memory_selection"),
        (RunPhase.MEMORY_PROPOSAL, "memory_proposal"),
    ],
)
def test_memory_phase_run_state_round_trip_uses_exact_wire_value(
    phase: RunPhase,
    wire_value: str,
) -> None:
    state = RunState(
        review_id="review-memory",
        status=RunStatus.RUNNING,
        phase=phase,
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
        resolved_base_revision="1111111111111111111111111111111111111111",
        resolved_head_revision="2222222222222222222222222222222222222222",
        message=f"Running {wire_value}",
        artifacts={"memory": f"{wire_value}.json"},
        errors=["prior warning"],
    )

    payload = run_state_to_dict(state)

    assert payload == {
        "review_id": "review-memory",
        "status": "running",
        "phase": wire_value,
        "repository_path": "repo",
        "base_revision": "main",
        "head_revision": "HEAD",
        "resolved_base_revision": "1111111111111111111111111111111111111111",
        "resolved_head_revision": "2222222222222222222222222222222222222222",
        "message": f"Running {wire_value}",
        "artifacts": {"memory": f"{wire_value}.json"},
        "errors": ["prior warning"],
    }
    loaded = run_state_from_dict(payload)

    assert loaded == state
    assert loaded.phase is phase


@pytest.mark.parametrize(
    "phase_value",
    [
        "MEMORY_SELECTION",
        "memory-selection",
        " memory_selection",
        "memory_proposal ",
    ],
)
def test_run_state_from_dict_rejects_noncanonical_memory_phase_values(
    phase_value: str,
) -> None:
    state = initial_run_state(
        review_id="review-memory",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
    )
    payload = run_state_to_dict(state)
    payload["phase"] = phase_value

    with pytest.raises(ValueError):
        run_state_from_dict(payload)


@pytest.mark.parametrize(
    "phase",
    [RunPhase.MEMORY_SELECTION, RunPhase.MEMORY_PROPOSAL],
)
def test_advance_to_memory_phase_preserves_existing_state_behavior(
    phase: RunPhase,
) -> None:
    state = initial_run_state(
        review_id="review-memory",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
        resolved_base_revision="1111111111111111111111111111111111111111",
        resolved_head_revision="2222222222222222222222222222222222222222",
    )
    state = advance_run_state(
        state,
        phase=RunPhase.REPOSITORY_INTELLIGENCE,
        message="Repository intelligence completed",
        artifacts={"existing": "repository_observations.json"},
    )

    advanced = advance_run_state(
        state,
        phase=phase,
        message=f"Running {phase.value}",
        artifacts={"memory": f"{phase.value}.json"},
    )

    assert advanced.status is RunStatus.RUNNING
    assert advanced.phase is phase
    assert advanced.artifacts == {
        "existing": "repository_observations.json",
        "memory": f"{phase.value}.json",
    }
    assert advanced.resolved_base_revision == state.resolved_base_revision
    assert advanced.resolved_head_revision == state.resolved_head_revision
    assert state.artifacts == {"existing": "repository_observations.json"}


@pytest.mark.parametrize(
    "phase",
    [RunPhase.MEMORY_SELECTION, RunPhase.MEMORY_PROPOSAL],
)
def test_awaiting_user_remains_invalid_during_memory_phases(phase: RunPhase) -> None:
    with pytest.raises(
        ValueError,
        match="awaiting_user RunState is allowed only during intent_resolution",
    ):
        RunState(
            review_id="review-memory",
            status=RunStatus.AWAITING_USER,
            phase=phase,
            repository_path="repo",
            base_revision="main",
            head_revision="HEAD",
            message="Waiting",
        )


def test_await_user_run_state_behavior_is_unchanged() -> None:
    state = initial_run_state(
        review_id="review-1",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
    )

    waiting = await_user_run_state(
        state,
        message="Clarification required",
        artifacts={"questions": "clarification_questions.json"},
    )

    assert waiting.status is RunStatus.AWAITING_USER
    assert waiting.phase is RunPhase.INTENT_RESOLUTION
    assert waiting.artifacts == {"questions": "clarification_questions.json"}
    assert run_state_from_dict(run_state_to_dict(waiting)) == waiting


def test_run_state_advances_and_serializes_artifacts() -> None:
    state = initial_run_state(
        review_id="review-1",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
        resolved_base_revision="1111111111111111111111111111111111111111",
        resolved_head_revision="2222222222222222222222222222222222222222",
    )

    state = advance_run_state(
        state,
        phase=RunPhase.PREFLIGHT,
        message="Preflight completed",
        artifacts={"request": "request.json"},
    )

    payload = run_state_to_dict(state)
    loaded = run_state_from_dict(payload)

    assert loaded.status is RunStatus.RUNNING
    assert loaded.phase is RunPhase.PREFLIGHT
    assert loaded.artifacts == {"request": "request.json"}
    assert payload["base_revision"] == "main"
    assert payload["head_revision"] == "HEAD"
    assert payload["resolved_base_revision"] == "1111111111111111111111111111111111111111"
    assert payload["resolved_head_revision"] == "2222222222222222222222222222222222222222"
    assert loaded.base_revision == "main"
    assert loaded.head_revision == "HEAD"
    assert loaded.resolved_base_revision == "1111111111111111111111111111111111111111"
    assert loaded.resolved_head_revision == "2222222222222222222222222222222222222222"


def test_run_state_completed_phase_marks_status_completed() -> None:
    state = initial_run_state(
        review_id="review-1",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
    )

    state = advance_run_state(state, phase=RunPhase.COMPLETED, message="Review completed")

    assert state.status is RunStatus.COMPLETED
    assert state.phase is RunPhase.COMPLETED


def test_run_state_failed_preserves_artifacts_and_appends_error() -> None:
    state = initial_run_state(
        review_id="review-1",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
        resolved_base_revision="1111111111111111111111111111111111111111",
        resolved_head_revision="2222222222222222222222222222222222222222",
    )
    state = advance_run_state(
        state,
        phase=RunPhase.PREFLIGHT,
        message="Preflight completed",
        artifacts={"request": "request.json"},
    )

    failed = fail_run_state(state, message="Review failed", error="RuntimeError: boom")

    assert failed.status is RunStatus.FAILED
    assert failed.phase is RunPhase.FAILED
    assert failed.artifacts == {"request": "request.json"}
    assert failed.errors == ["RuntimeError: boom"]
    assert failed.resolved_base_revision == "1111111111111111111111111111111111111111"
    assert failed.resolved_head_revision == "2222222222222222222222222222222222222222"


def test_advance_run_state_preserves_resolved_revisions() -> None:
    state = initial_run_state(
        review_id="review-1",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
        resolved_base_revision="1111111111111111111111111111111111111111",
        resolved_head_revision="2222222222222222222222222222222222222222",
    )

    advanced = advance_run_state(
        state,
        phase=RunPhase.REPOSITORY_INTELLIGENCE,
        message="Repository intelligence completed",
    )

    assert advanced.resolved_base_revision == "1111111111111111111111111111111111111111"
    assert advanced.resolved_head_revision == "2222222222222222222222222222222222222222"


def test_run_state_from_legacy_payload_defaults_resolved_revisions_to_none() -> None:
    payload = {
        "review_id": "review-1",
        "status": "running",
        "phase": "preflight",
        "repository_path": "repo",
        "base_revision": "main",
        "head_revision": "HEAD",
        "message": "Preflight completed",
        "artifacts": {"request": "request.json"},
        "errors": [],
    }

    loaded = run_state_from_dict(payload)

    assert loaded.resolved_base_revision is None
    assert loaded.resolved_head_revision is None


def test_run_state_from_dict_preserves_json_null_resolved_revisions() -> None:
    payload = {
        "review_id": "review-1",
        "status": "running",
        "phase": "preflight",
        "repository_path": "repo",
        "base_revision": "main",
        "head_revision": "HEAD",
        "resolved_base_revision": None,
        "resolved_head_revision": None,
        "message": "Preflight completed",
        "artifacts": {},
        "errors": [],
    }

    loaded = run_state_from_dict(payload)

    assert loaded.resolved_base_revision is None
    assert loaded.resolved_head_revision is None
    assert run_state_to_dict(loaded)["resolved_base_revision"] is None
    assert run_state_to_dict(loaded)["resolved_head_revision"] is None

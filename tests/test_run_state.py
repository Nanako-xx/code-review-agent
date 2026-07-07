from review_agent.run_state import (
    RunPhase,
    RunStatus,
    advance_run_state,
    fail_run_state,
    initial_run_state,
    run_state_from_dict,
    run_state_to_dict,
)


def test_run_state_advances_and_serializes_artifacts() -> None:
    state = initial_run_state(
        review_id="review-1",
        repository_path="repo",
        base_revision="main",
        head_revision="HEAD",
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

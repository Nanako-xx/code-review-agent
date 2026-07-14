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

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    SESSION_PHASES,
    PhaseStatus,
    ReviewExecutionConfig,
    SessionManifest,
    initial_session_manifest,
)
from review_agent.session_store import SessionStore


NOW = "2026-07-10T00:00:00Z"
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
REVISION_BINDING = f"{BASE_SHA}..{HEAD_SHA}"


def manifest(review_id: str = "review-1") -> SessionManifest:
    return initial_session_manifest(
        review_id=review_id,
        repository=RepositoryIdentity("C:/repo", "C:/repo/.git", None),
        revisions=ResolvedRevisions("main", "HEAD", BASE_SHA, HEAD_SHA),
        execution=ReviewExecutionConfig(
            reviewer_provider="fake",
            reviewer_model=None,
            reviewer_base_url=None,
            reviewer_api_key_env="REVIEW_AGENT_API_KEY",
            reviewer_mode="single",
            reviewer_loop="single-shot",
            non_interactive=True,
        ),
        now=NOW,
    )


def create_store(run_dir: Path) -> SessionStore:
    store = SessionStore(run_dir)
    store.create(manifest())
    return store


def register_artifact(
    store: SessionStore,
    *,
    name: str = "request",
    relative_path: str = "request.json",
    phase: RunPhase = RunPhase.PREFLIGHT,
    revision_binding: str | None = None,
    now: str = "2026-07-10T00:01:00Z",
) -> SessionManifest:
    return store.register_existing_artifact(
        name=name,
        relative_path=relative_path,
        schema=f"{name}_v1",
        phase=phase,
        revision_binding=revision_binding,
        now=now,
    )


def mark_all_phases(
    store: SessionStore,
    *,
    preflight_artifacts: list[str] | None = None,
) -> SessionManifest:
    updated = store.load()
    for index, phase in enumerate(SESSION_PHASES, start=1):
        artifact_names = (
            preflight_artifacts
            if phase is RunPhase.PREFLIGHT and preflight_artifacts is not None
            else []
        )
        updated = store.mark_phase_completed(
            phase,
            artifact_names,
            f"2026-07-10T00:{index:02d}:00Z",
        )
    return updated


def test_session_store_atomically_round_trips_explicit_manifest_schema(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    store = SessionStore(run_dir)
    original = manifest()

    session_path = store.create(original)

    payload = json.loads(session_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["review_id"] == "review-1"
    assert payload["phases"]["preflight"]["status"] == "pending"
    assert store.load() == original
    assert [path for path in run_dir.iterdir() if path.name.endswith(".tmp")] == []


def test_session_store_create_never_overwrites_existing_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    original = manifest()
    store.create(original)

    with pytest.raises(FileExistsError, match="session.json"):
        store.create(replace(original, updated_at="2026-07-10T00:01:00Z"))

    assert store.load() == original


def test_session_store_write_updates_only_the_same_review_id(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    original = manifest()
    store.create(original)
    updated = replace(
        original,
        status=RunStatus.RUNNING,
        current_phase=RunPhase.PREFLIGHT,
        updated_at="2026-07-10T00:01:00Z",
    )

    store.write(updated)

    assert store.load() == updated


def test_session_store_write_cannot_replace_another_review_session(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    original = manifest()
    store.create(original)

    with pytest.raises(ValueError, match="review_id"):
        store.write(manifest("review-2"))

    assert store.load() == original


def test_session_store_load_applies_strict_session_schema_validation(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    store.create(manifest())
    payload = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    (tmp_path / "session.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported session schema_version"):
        store.load()


def test_session_store_registers_raw_byte_hash_and_exact_revision_binding(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    artifact_path = tmp_path / "request.bin"
    raw_bytes = b"\xff\x00review\r\n"
    artifact_path.write_bytes(raw_bytes)

    updated = register_artifact(
        store,
        relative_path="request.bin",
        revision_binding=REVISION_BINDING,
    )

    descriptor = updated.artifacts["request"]
    assert descriptor.sha256 == sha256(raw_bytes).hexdigest()
    assert descriptor.path == "request.bin"
    assert descriptor.phase is RunPhase.PREFLIGHT
    assert descriptor.revision_binding == REVISION_BINDING
    assert updated.updated_at == "2026-07-10T00:01:00Z"
    assert store.validate_artifact(descriptor) is True


def test_session_store_rejects_mismatched_non_empty_revision_binding(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="revision_binding"):
        register_artifact(store, revision_binding=f"{'c' * 40}..{HEAD_SHA}")

    assert store.load().artifacts == {}


@pytest.mark.parametrize(
    "relative_path",
    [
        "../outside.bin",
        "/absolute.bin",
        "C:/absolute.bin",
        "C:\\absolute.bin",
    ],
)
def test_session_store_rejects_non_relative_or_traversing_artifact_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    store = create_store(tmp_path)

    with pytest.raises(ValueError, match="path|outside|relative|canonical"):
        register_artifact(store, relative_path=relative_path)


def test_session_store_rejects_missing_artifact(tmp_path: Path) -> None:
    store = create_store(tmp_path)

    with pytest.raises(ValueError, match="regular file|does not exist"):
        register_artifact(store, relative_path="missing.json")


def test_session_store_rejects_directory_as_artifact(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    (tmp_path / "artifact-dir").mkdir()

    with pytest.raises(ValueError, match="regular file"):
        register_artifact(store, relative_path="artifact-dir")


def test_session_store_rejects_symlink_escape(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = create_store(run_dir)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = run_dir / "escaped.json"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="outside"):
        register_artifact(store, relative_path="escaped.json")


def test_session_store_validation_detects_changed_artifact_bytes(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    artifact_path = tmp_path / "request.json"
    artifact_path.write_bytes(b'{"head":"HEAD"}')
    updated = register_artifact(store)
    descriptor = updated.artifacts["request"]

    artifact_path.write_bytes(b'{"head":"changed"}')

    assert store.validate_artifact(descriptor) is False


def test_mark_phase_completed_updates_immutable_checkpoint_in_order(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")
    register_artifact(store)

    updated = store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        ["request", "request"],
        "2026-07-10T00:02:00Z",
    )

    checkpoint = updated.phases[RunPhase.PREFLIGHT.value]
    assert checkpoint.status is PhaseStatus.COMPLETED
    assert checkpoint.attempts == 1
    assert checkpoint.started_at == "2026-07-10T00:02:00Z"
    assert checkpoint.completed_at == "2026-07-10T00:02:00Z"
    assert checkpoint.artifacts == ("request",)
    assert checkpoint.error is None
    assert updated.status is RunStatus.RUNNING
    assert updated.current_phase is RunPhase.PREFLIGHT
    assert updated.last_successful_phase is RunPhase.PREFLIGHT
    assert updated.updated_at == "2026-07-10T00:02:00Z"
    assert isinstance(updated.errors, tuple)


@pytest.mark.parametrize(
    "phase",
    [RunPhase.CREATED, RunPhase.QUALITY_GATES, RunPhase.COMPLETED, RunPhase.FAILED],
)
def test_mark_phase_completed_accepts_only_persisted_session_phases(
    tmp_path: Path,
    phase: RunPhase,
) -> None:
    store = create_store(tmp_path)

    with pytest.raises(ValueError, match="SESSION_PHASES|persisted"):
        store.mark_phase_completed(phase, [], "2026-07-10T00:02:00Z")


def test_mark_phase_completed_rejects_unregistered_artifact(tmp_path: Path) -> None:
    store = create_store(tmp_path)

    with pytest.raises(ValueError, match="not registered"):
        store.mark_phase_completed(
            RunPhase.PREFLIGHT,
            ["request"],
            "2026-07-10T00:02:00Z",
        )


def test_mark_phase_completed_rejects_artifact_from_another_phase(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "repository.json").write_text("{}", encoding="utf-8")
    register_artifact(
        store,
        name="repository",
        relative_path="repository.json",
        phase=RunPhase.REPOSITORY_INTELLIGENCE,
    )

    with pytest.raises(ValueError, match="belongs to phase"):
        store.mark_phase_completed(
            RunPhase.PREFLIGHT,
            ["repository"],
            "2026-07-10T00:02:00Z",
        )


def test_mark_phase_completed_rejects_invalid_registered_artifact(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    artifact_path = tmp_path / "request.json"
    artifact_path.write_text("{}", encoding="utf-8")
    register_artifact(store)
    artifact_path.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        store.mark_phase_completed(
            RunPhase.PREFLIGHT,
            ["request"],
            "2026-07-10T00:02:00Z",
        )


def test_mark_phase_completed_cannot_skip_unfinished_predecessor(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)

    with pytest.raises(ValueError, match="preflight"):
        store.mark_phase_completed(
            RunPhase.REPOSITORY_INTELLIGENCE,
            [],
            "2026-07-10T00:02:00Z",
        )


def test_mark_session_completed_requires_every_phase(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        [],
        "2026-07-10T00:01:00Z",
    )

    with pytest.raises(ValueError, match="not completed"):
        store.mark_session_completed("2026-07-10T00:02:00Z")

    assert store.load().status is RunStatus.RUNNING


def test_mark_session_completed_validates_referenced_artifacts(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    artifact_path = tmp_path / "request.json"
    artifact_path.write_text("{}", encoding="utf-8")
    register_artifact(store)
    mark_all_phases(store, preflight_artifacts=["request"])
    artifact_path.write_text('{"tampered":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        store.mark_session_completed("2026-07-10T00:10:00Z")

    assert store.load().status is RunStatus.RUNNING


def test_mark_session_completed_after_all_phases_and_valid_artifacts(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")
    register_artifact(store)
    mark_all_phases(store, preflight_artifacts=["request"])

    completed = store.mark_session_completed("2026-07-10T00:10:00Z")

    assert completed.status is RunStatus.COMPLETED
    assert completed.current_phase is RunPhase.COMPLETED
    assert completed.last_successful_phase is RunPhase.REPORTING
    assert completed.updated_at == "2026-07-10T00:10:00Z"
    assert all(
        checkpoint.status is PhaseStatus.COMPLETED
        for checkpoint in completed.phases.values()
    )


def test_mark_session_failed_preserves_last_successful_phase_and_appends_errors(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        [],
        "2026-07-10T00:01:00Z",
    )

    failed = store.mark_session_failed(
        RunPhase.REVIEWERS,
        "provider unavailable",
        "2026-07-10T00:02:00Z",
    )
    failed_again = store.mark_session_failed(
        RunPhase.REVIEWERS,
        "provider still unavailable",
        "2026-07-10T00:03:00Z",
    )

    checkpoint = failed_again.phases[RunPhase.REVIEWERS.value]
    assert failed.status is RunStatus.FAILED
    assert failed_again.status is RunStatus.FAILED
    assert failed_again.current_phase is RunPhase.FAILED
    assert failed_again.last_successful_phase is RunPhase.PREFLIGHT
    assert checkpoint.status is PhaseStatus.FAILED
    assert checkpoint.attempts == 2
    assert checkpoint.started_at == "2026-07-10T00:02:00Z"
    assert checkpoint.completed_at is None
    assert checkpoint.error == "provider still unavailable"
    assert failed_again.errors == (
        "provider unavailable",
        "provider still unavailable",
    )
    assert failed_again.updated_at == "2026-07-10T00:03:00Z"


def test_mark_session_failed_accepts_only_persisted_session_phases(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)

    with pytest.raises(ValueError, match="SESSION_PHASES|persisted"):
        store.mark_session_failed(
            RunPhase.QUALITY_GATES,
            "failed",
            "2026-07-10T00:02:00Z",
        )

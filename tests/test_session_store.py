from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import errno
import json
import os
from pathlib import Path

import pytest

import review_agent.session_store as session_store_module
from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    SESSION_PHASES,
    ArtifactDescriptor,
    PhaseCheckpoint,
    PhaseStatus,
    ReviewExecutionConfig,
    RevisionChangeKind,
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


def child_manifest() -> SessionManifest:
    return replace(
        manifest("review-child"),
        parent_review_id="review-parent",
        root_review_id="review-root",
        incremental_from_sha="c" * 40,
        revision_change_kind=RevisionChangeKind.HEAD_MOVED,
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
    schema = "review_request_v1" if name == "request" else f"{name}_v1"
    return store.register_existing_artifact(
        name=name,
        relative_path=relative_path,
        schema=schema,
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


@pytest.mark.parametrize(
    ("field_name", "modified_manifest"),
    [
        (
            "repository",
            replace(
                manifest(),
                repository=RepositoryIdentity("C:/other", "C:/other/.git", None),
            ),
        ),
        (
            "revisions",
            replace(
                manifest(),
                revisions=ResolvedRevisions("main", "feature", BASE_SHA, HEAD_SHA),
            ),
        ),
        (
            "execution",
            replace(
                manifest(),
                execution=replace(
                    manifest().execution,
                    reviewer_provider="other-provider",
                ),
            ),
        ),
        ("created_at", replace(manifest(), created_at="2026-07-09T23:59:00Z")),
    ],
)
def test_session_store_write_rejects_immutable_session_metadata(
    tmp_path: Path,
    field_name: str,
    modified_manifest: SessionManifest,
) -> None:
    store = create_store(tmp_path)

    with pytest.raises(ValueError, match=field_name):
        store.write(modified_manifest)


@pytest.mark.parametrize(
    ("field_name", "changes"),
    [
        ("parent_review_id", {"parent_review_id": "other-parent"}),
        ("root_review_id", {"root_review_id": "other-root"}),
        ("original_base_sha", {"original_base_sha": "d" * 40}),
        ("incremental_from_sha", {"incremental_from_sha": "d" * 40}),
        (
            "revision_change_kind",
            {
                "revision_change_kind": RevisionChangeKind.BASE_MOVED,
                "incremental_from_sha": None,
            },
        ),
    ],
)
def test_session_store_write_rejects_immutable_lineage(
    tmp_path: Path,
    field_name: str,
    changes: dict[str, object],
) -> None:
    original = child_manifest()
    store = SessionStore(tmp_path)
    store.create(original)

    with pytest.raises(ValueError, match=field_name):
        store.write(replace(original, **changes))


def test_session_store_write_rejects_schema_version_mutation(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    modified = manifest()
    object.__setattr__(modified, "schema_version", 2)

    with pytest.raises(ValueError, match="schema_version"):
        store.write(modified)


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
    ("name", "schema", "phase"),
    [
        ("intent", "intent_packet_v1", RunPhase.PREFLIGHT),
        ("request", "other_request_v1", RunPhase.PREFLIGHT),
        ("request", "review_request_v1", RunPhase.REPOSITORY_INTELLIGENCE),
    ],
)
def test_session_store_allows_unbound_artifact_only_for_preflight_request(
    tmp_path: Path,
    name: str,
    schema: str,
    phase: RunPhase,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "artifact.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="revision_binding"):
        store.register_existing_artifact(
            name=name,
            relative_path="artifact.json",
            schema=schema,
            phase=phase,
            revision_binding=None,
            now="2026-07-10T00:01:00Z",
        )


def test_session_store_rejects_empty_revision_binding_even_for_request(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="revision_binding"):
        register_artifact(store, revision_binding="")


def test_session_store_validation_enforces_revision_binding_policy(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    artifact_path = tmp_path / "intent.json"
    artifact_path.write_text("{}", encoding="utf-8")
    digest = sha256(artifact_path.read_bytes()).hexdigest()

    unbound = ArtifactDescriptor(
        name="intent",
        path="intent.json",
        sha256=digest,
        schema="intent_packet_v1",
        phase=RunPhase.PREFLIGHT,
        revision_binding=None,
    )
    empty = replace(
        unbound,
        name="request",
        schema="review_request_v1",
        revision_binding="",
    )

    assert store.validate_artifact(unbound) is False
    assert store.validate_artifact(empty) is False


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


def test_session_store_hashes_with_one_open_and_the_same_fstat_read_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = create_store(tmp_path)
    artifact_path = tmp_path / "request.bin"
    artifact_path.write_bytes(b"artifact bytes")
    resolved_artifact = artifact_path.resolve()
    artifact_open_calls: list[tuple[int, int]] = []
    fstat_descriptors: list[int] = []
    read_descriptors: list[int] = []
    real_open = session_store_module.os.open
    real_fstat = session_store_module.os.fstat
    real_read = session_store_module.os.read

    def recording_open(path: object, flags: int, *args: object) -> int:
        file_descriptor = real_open(path, flags, *args)
        if Path(path) == resolved_artifact:
            artifact_open_calls.append((file_descriptor, flags))
        return file_descriptor

    def recording_fstat(file_descriptor: int):
        fstat_descriptors.append(file_descriptor)
        return real_fstat(file_descriptor)

    def recording_read(file_descriptor: int, size: int) -> bytes:
        read_descriptors.append(file_descriptor)
        return real_read(file_descriptor, size)

    monkeypatch.setattr(session_store_module.os, "open", recording_open)
    monkeypatch.setattr(session_store_module.os, "fstat", recording_fstat)
    monkeypatch.setattr(session_store_module.os, "read", recording_read)

    register_artifact(store, relative_path="request.bin")

    assert len(artifact_open_calls) == 1
    artifact_descriptor, flags = artifact_open_calls[0]
    assert fstat_descriptors == [artifact_descriptor]
    assert read_descriptors
    assert set(read_descriptors) == {artifact_descriptor}
    if getattr(os, "O_NOFOLLOW", 0):
        assert flags & os.O_NOFOLLOW


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


def test_mark_phase_completed_is_idempotent_for_the_same_artifact_set(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")
    (tmp_path / "intent.json").write_text("{}", encoding="utf-8")
    register_artifact(store)
    register_artifact(
        store,
        name="intent",
        relative_path="intent.json",
        phase=RunPhase.PREFLIGHT,
        revision_binding=REVISION_BINDING,
    )
    completed = store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        ["request", "intent"],
        "2026-07-10T00:02:00Z",
    )
    session_bytes = (tmp_path / "session.json").read_bytes()

    repeated = store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        ["intent", "request", "request"],
        "2026-07-10T00:03:00Z",
    )

    assert repeated == completed
    assert repeated.phases["preflight"].attempts == 1
    assert repeated.updated_at == "2026-07-10T00:02:00Z"
    assert (tmp_path / "session.json").read_bytes() == session_bytes


@pytest.mark.parametrize("mutation", ["tamper", "delete"])
def test_mark_phase_completed_idempotence_revalidates_artifacts(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = create_store(tmp_path)
    artifact_path = tmp_path / "request.json"
    artifact_path.write_text("{}", encoding="utf-8")
    register_artifact(store)
    completed = store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        ["request"],
        "2026-07-10T00:02:00Z",
    )
    if mutation == "tamper":
        artifact_path.write_text('{"tampered":true}', encoding="utf-8")
    else:
        artifact_path.unlink()

    with pytest.raises(ValueError, match="validation"):
        store.mark_phase_completed(
            RunPhase.PREFLIGHT,
            ["request"],
            "2026-07-10T00:03:00Z",
        )

    assert store.load() == completed


def test_mark_phase_completed_rejects_different_artifacts_after_completion(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")
    register_artifact(store)
    store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        ["request"],
        "2026-07-10T00:02:00Z",
    )

    with pytest.raises(ValueError, match="already completed"):
        store.mark_phase_completed(
            RunPhase.PREFLIGHT,
            [],
            "2026-07-10T00:03:00Z",
        )

    assert store.load().phases["preflight"].attempts == 1


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
        revision_binding=REVISION_BINDING,
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


def test_mark_session_completed_rejects_orphan_registry_artifact(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")
    register_artifact(store)
    mark_all_phases(store)

    with pytest.raises(ValueError, match="registry|orphan"):
        store.mark_session_completed("2026-07-10T00:10:00Z")


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


def test_mark_session_completed_is_idempotent_without_rewriting_manifest(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    mark_all_phases(store)
    completed = store.mark_session_completed("2026-07-10T00:10:00Z")
    session_bytes = (tmp_path / "session.json").read_bytes()

    repeated = store.mark_session_completed("2026-07-10T00:11:00Z")

    assert repeated == completed
    assert repeated.updated_at == "2026-07-10T00:10:00Z"
    assert (tmp_path / "session.json").read_bytes() == session_bytes


@pytest.mark.parametrize("mutation", ["tamper", "delete"])
def test_mark_session_completed_idempotence_revalidates_artifacts(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = create_store(tmp_path)
    artifact_path = tmp_path / "request.json"
    artifact_path.write_text("{}", encoding="utf-8")
    register_artifact(store)
    mark_all_phases(store, preflight_artifacts=["request"])
    completed = store.mark_session_completed("2026-07-10T00:10:00Z")
    if mutation == "tamper":
        artifact_path.write_text('{"tampered":true}', encoding="utf-8")
    else:
        artifact_path.unlink()

    with pytest.raises(ValueError, match="validation"):
        store.mark_session_completed("2026-07-10T00:11:00Z")

    assert store.load() == completed


def test_mark_session_completed_idempotence_revalidates_registry_consistency(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    mark_all_phases(store)
    store.mark_session_completed("2026-07-10T00:10:00Z")
    session_path = tmp_path / "session.json"
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    payload["artifacts"]["orphan"] = {
        "name": "orphan",
        "path": "orphan.json",
        "sha256": "c" * 64,
        "schema": "orphan_v1",
        "phase": "reporting",
        "revision_binding": REVISION_BINDING,
    }
    session_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="registry|orphan"):
        store.mark_session_completed("2026-07-10T00:11:00Z")


def test_completed_session_cannot_be_failed_reopened_or_given_new_artifact(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    mark_all_phases(store)
    completed = store.mark_session_completed("2026-07-10T00:10:00Z")
    (tmp_path / "late.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="completed Session"):
        store.mark_session_failed(
            RunPhase.REVIEWERS,
            "late failure",
            "2026-07-10T00:11:00Z",
        )
    with pytest.raises(ValueError, match="completed Session"):
        register_artifact(
            store,
            name="late",
            relative_path="late.json",
            phase=RunPhase.REPORTING,
            revision_binding=REVISION_BINDING,
            now="2026-07-10T00:11:00Z",
        )
    with pytest.raises(ValueError, match="completed"):
        store.write(
            replace(
                completed,
                status=RunStatus.RUNNING,
                current_phase=RunPhase.REPORTING,
                updated_at="2026-07-10T00:11:00Z",
            )
        )


def test_completed_phase_cannot_be_marked_failed(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        [],
        "2026-07-10T00:01:00Z",
    )

    with pytest.raises(ValueError, match="completed phase"):
        store.mark_session_failed(
            RunPhase.PREFLIGHT,
            "late failure",
            "2026-07-10T00:02:00Z",
        )


def test_completed_phase_rejects_new_artifact_but_allows_identical_registration(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")
    (tmp_path / "late.json").write_text("{}", encoding="utf-8")
    registered = register_artifact(store)
    completed = store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        ["request"],
        "2026-07-10T00:02:00Z",
    )

    repeated = register_artifact(store, now="2026-07-10T00:03:00Z")

    assert repeated == completed
    assert repeated.artifacts["request"] == registered.artifacts["request"]
    with pytest.raises(ValueError, match="completed phase"):
        register_artifact(
            store,
            name="late",
            relative_path="late.json",
            phase=RunPhase.PREFLIGHT,
            revision_binding=REVISION_BINDING,
            now="2026-07-10T00:03:00Z",
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
        RunPhase.REPOSITORY_INTELLIGENCE,
        "provider unavailable",
        "2026-07-10T00:02:00Z",
    )
    failed_again = store.mark_session_failed(
        RunPhase.REPOSITORY_INTELLIGENCE,
        "provider still unavailable",
        "2026-07-10T00:03:00Z",
    )

    checkpoint = failed_again.phases[RunPhase.REPOSITORY_INTELLIGENCE.value]
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


@pytest.mark.parametrize(
    "predecessor_status",
    [PhaseStatus.PENDING, PhaseStatus.FAILED, PhaseStatus.INVALIDATED],
)
def test_mark_session_failed_rejects_non_completed_predecessor(
    tmp_path: Path,
    predecessor_status: PhaseStatus,
) -> None:
    store = create_store(tmp_path)
    if predecessor_status is not PhaseStatus.PENDING:
        current = store.load()
        phases = dict(current.phases)
        phases[RunPhase.PREFLIGHT.value] = PhaseCheckpoint(
            status=predecessor_status,
            attempts=1,
            started_at="2026-07-10T00:01:00Z",
            completed_at=None,
            artifacts=(),
            error="preflight unavailable",
        )
        store.write(
            replace(
                current,
                status=RunStatus.FAILED,
                current_phase=RunPhase.FAILED,
                phases=phases,
                updated_at="2026-07-10T00:01:00Z",
            )
        )

    with pytest.raises(ValueError, match="preflight|predecessor"):
        store.mark_session_failed(
            RunPhase.REPOSITORY_INTELLIGENCE,
            "repository intelligence failed",
            "2026-07-10T00:02:00Z",
        )


def test_mark_session_failed_rejects_completed_successor(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    current = store.load()
    phases = dict(current.phases)
    phases[RunPhase.REVIEWERS.value] = PhaseCheckpoint(
        status=PhaseStatus.COMPLETED,
        attempts=1,
        started_at="2026-07-10T00:01:00Z",
        completed_at="2026-07-10T00:02:00Z",
        artifacts=(),
        error=None,
    )
    store.write(
        replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=RunPhase.REVIEWERS,
            phases=phases,
            updated_at="2026-07-10T00:02:00Z",
        )
    )

    with pytest.raises(ValueError, match="completed successor|reviewers"):
        store.mark_session_failed(
            RunPhase.PREFLIGHT,
            "preflight failed",
            "2026-07-10T00:03:00Z",
        )


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


@pytest.mark.skipif(os.name != "nt", reason="Windows rename has no-replace semantics")
def test_session_create_falls_back_to_windows_rename_when_link_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path)
    rename_calls: list[tuple[Path, Path]] = []
    real_rename = session_store_module.os.rename

    def unsupported_link(source: object, destination: object) -> None:
        raise OSError(errno.ENOTSUP, "hard links unsupported")

    def recording_rename(source: object, destination: object) -> None:
        rename_calls.append((Path(source), Path(destination)))
        real_rename(source, destination)

    monkeypatch.setattr(session_store_module.os, "link", unsupported_link)
    monkeypatch.setattr(session_store_module.os, "rename", recording_rename)

    store.create(manifest())

    assert len(rename_calls) == 1
    assert rename_calls[0][1] == tmp_path / "session.json"
    assert store.load() == manifest()


@pytest.mark.skipif(os.name != "nt", reason="Windows rename has no-replace semantics")
def test_session_create_fallback_never_overwrites_existing_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SessionStore(tmp_path)
    original = manifest()
    store.create(original)

    def unsupported_link(source: object, destination: object) -> None:
        raise OSError(errno.ENOTSUP, "hard links unsupported")

    def existing_destination(source: object, destination: object) -> None:
        assert Path(destination).exists()
        raise FileExistsError("destination exists")

    monkeypatch.setattr(session_store_module.os, "link", unsupported_link)
    monkeypatch.setattr(session_store_module.os, "rename", existing_destination)

    with pytest.raises(FileExistsError, match="session.json"):
        store.create(replace(original, updated_at="2026-07-10T00:01:00Z"))

    assert store.load() == original

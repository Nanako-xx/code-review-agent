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
    LEGACY_SESSION_PHASES,
    LEGACY_SESSION_SCHEMA_VERSION,
    MODEL_STAGE_SESSION_SCHEMA_VERSION,
    PREVIOUS_SESSION_SCHEMA_VERSION,
    PREVIOUS_SESSION_PHASES,
    SESSION_PHASES,
    SESSION_SCHEMA_VERSION,
    ArtifactDescriptor,
    ModelStageConfig,
    PhaseCheckpoint,
    PhaseStatus,
    ReviewWaveCheckpoint,
    ReviewExecutionConfig,
    RevisionChangeKind,
    SessionManifest,
    SupplementalBudget,
    SupplementalPolicy,
    SupplementalTaskStatus,
    initial_session_manifest,
    session_manifest_from_dict,
    session_manifest_to_dict,
)
from review_agent.session_store import PhaseValidation, SessionStore


NOW = "2026-07-10T00:00:00Z"
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
REVISION_BINDING = f"{BASE_SHA}..{HEAD_SHA}"
WAVE_1 = f"W-{'1' * 64}"
WAVE_2 = f"W-{'2' * 64}"
TASK_1 = f"STASK-{'3' * 64}"
TASK_2 = f"STASK-{'4' * 64}"
TASK_3 = f"STASK-{'9' * 64}"
ASSIGNMENT_1 = "5" * 64
ASSIGNMENT_2 = "6" * 64
ASSIGNMENT_3 = "a" * 64
TRIGGER_1 = "7" * 64
TRIGGER_2 = "8" * 64


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


def _downgrade_v4_payload(payload: dict[str, object], schema_version: int) -> None:
    payload["schema_version"] = schema_version
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution.pop("semantic_reconciler")
    execution.pop("supplemental_policy")
    payload.pop("supplemental_waves")
    if schema_version < MODEL_STAGE_SESSION_SCHEMA_VERSION:
        execution.pop("risk_assessor")
        execution.pop("portfolio_planner")
    if schema_version != LEGACY_SESSION_SCHEMA_VERSION:
        phases = payload["phases"]
        assert isinstance(phases, dict)
        payload["phases"] = {
            phase.value: phases[phase.value]
            for phase in PREVIOUS_SESSION_PHASES
        }


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
    assert payload["schema_version"] == SESSION_SCHEMA_VERSION
    assert payload["review_id"] == "review-1"
    assert payload["phases"]["preflight"]["status"] == "pending"
    assert store.load() == original
    assert [path for path in run_dir.iterdir() if path.name.endswith(".tmp")] == []


def test_session_store_loads_v1_for_audit_without_synthesizing_results(
    tmp_path: Path,
) -> None:
    payload = session_manifest_to_dict(manifest())
    _downgrade_v4_payload(payload, LEGACY_SESSION_SCHEMA_VERSION)
    payload["phases"] = {
        phase.value: payload["phases"][phase.value]
        for phase in LEGACY_SESSION_PHASES
    }
    for checkpoint in payload["phases"].values():
        checkpoint.pop("user_decisions")
    (tmp_path / "session.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    store = SessionStore(tmp_path)

    loaded = store.load()

    assert loaded == session_manifest_from_dict(payload)
    assert loaded.schema_version == 1
    assert list(loaded.phases) == [phase.value for phase in LEGACY_SESSION_PHASES]
    assert "intent_resolution" not in loaded.phases
    assert loaded.execution.risk_assessor == ModelStageConfig()
    assert loaded.execution.portfolio_planner == ModelStageConfig()
    assert loaded.execution.semantic_reconciler == ModelStageConfig()
    assert loaded.execution.supplemental_policy == SupplementalPolicy()
    assert loaded.supplemental_waves == {}
    with pytest.raises(ValueError, match="read-only audit|schema v1"):
        store.mark_phase_running(RunPhase.PREFLIGHT, "2026-07-10T00:01:00Z")


def test_session_store_refuses_to_create_a_new_v1_session(tmp_path: Path) -> None:
    payload = session_manifest_to_dict(manifest())
    _downgrade_v4_payload(payload, LEGACY_SESSION_SCHEMA_VERSION)
    payload["phases"] = {
        phase.value: payload["phases"][phase.value]
        for phase in LEGACY_SESSION_PHASES
    }
    for checkpoint in payload["phases"].values():
        checkpoint.pop("user_decisions")
    legacy = session_manifest_from_dict(payload)

    with pytest.raises(ValueError, match="new Sessions.*current"):
        SessionStore(tmp_path).create(legacy)


def test_session_store_loads_v2_with_local_model_stages(tmp_path: Path) -> None:
    payload = session_manifest_to_dict(manifest())
    _downgrade_v4_payload(payload, PREVIOUS_SESSION_SCHEMA_VERSION)
    (tmp_path / "session.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = SessionStore(tmp_path).load()

    assert loaded.schema_version == PREVIOUS_SESSION_SCHEMA_VERSION
    assert loaded.execution.reviewer_provider == "fake"
    assert loaded.execution.risk_assessor == ModelStageConfig()
    assert loaded.execution.portfolio_planner == ModelStageConfig()
    assert loaded.execution.semantic_reconciler == ModelStageConfig()
    assert list(loaded.phases) == [phase.value for phase in PREVIOUS_SESSION_PHASES]


def test_session_store_resumes_v2_without_enabling_model_stages(tmp_path: Path) -> None:
    payload = session_manifest_to_dict(manifest())
    _downgrade_v4_payload(payload, PREVIOUS_SESSION_SCHEMA_VERSION)
    (tmp_path / "session.json").write_text(json.dumps(payload), encoding="utf-8")
    store = SessionStore(tmp_path)

    updated = store.mark_phase_running(RunPhase.PREFLIGHT, NOW)

    assert updated.schema_version == PREVIOUS_SESSION_SCHEMA_VERSION
    assert updated.phases[RunPhase.PREFLIGHT.value].status is PhaseStatus.RUNNING
    assert updated.execution.risk_assessor == ModelStageConfig()
    assert updated.execution.portfolio_planner == ModelStageConfig()


def test_session_store_resumes_v3_with_original_layout_and_model_semantics(
    tmp_path: Path,
) -> None:
    payload = session_manifest_to_dict(manifest())
    _downgrade_v4_payload(payload, MODEL_STAGE_SESSION_SCHEMA_VERSION)
    (tmp_path / "session.json").write_text(json.dumps(payload), encoding="utf-8")
    store = SessionStore(tmp_path)

    updated = store.mark_phase_running(RunPhase.PREFLIGHT, NOW)

    assert updated.schema_version == MODEL_STAGE_SESSION_SCHEMA_VERSION
    assert list(updated.phases) == [phase.value for phase in PREVIOUS_SESSION_PHASES]
    assert updated.execution.risk_assessor == manifest().execution.risk_assessor
    assert updated.execution.portfolio_planner == manifest().execution.portfolio_planner
    assert updated.execution.semantic_reconciler == ModelStageConfig()
    with pytest.raises(ValueError, match="schema v4|Session schema layout"):
        store.mark_phase_running(
            RunPhase.RECONCILIATION_ANALYSIS,
            "2026-07-10T00:01:00Z",
        )


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
    object.__setattr__(modified, "schema_version", LEGACY_SESSION_SCHEMA_VERSION)

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


def test_session_store_load_reads_one_no_follow_fstat_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = create_store(tmp_path)
    session_path = store.session_path.resolve()
    open_calls: list[tuple[int, int]] = []
    fstat_descriptors: list[int] = []
    read_descriptors: list[int] = []
    descriptor_paths: dict[int, Path] = {}
    real_open = session_store_module.os.open
    real_fstat = session_store_module.os.fstat
    real_read = session_store_module.os.read

    def recording_open(path: object, flags: int, *args: object) -> int:
        descriptor = real_open(path, flags, *args)
        descriptor_paths[descriptor] = Path(path).resolve()
        if descriptor_paths[descriptor] == session_path:
            open_calls.append((descriptor, flags))
        return descriptor

    def recording_fstat(descriptor: int):
        if descriptor_paths.get(descriptor) == session_path:
            fstat_descriptors.append(descriptor)
        return real_fstat(descriptor)

    def recording_read(descriptor: int, size: int) -> bytes:
        if descriptor_paths.get(descriptor) == session_path:
            read_descriptors.append(descriptor)
        return real_read(descriptor, size)

    def forbid_path_read(*args: object, **kwargs: object) -> str:
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(session_store_module.os, "open", recording_open)
    monkeypatch.setattr(session_store_module.os, "fstat", recording_fstat)
    monkeypatch.setattr(session_store_module.os, "read", recording_read)
    monkeypatch.setattr(Path, "read_text", forbid_path_read)

    assert store.load() == manifest()

    assert len(open_calls) == 1
    descriptor, flags = open_calls[0]
    assert fstat_descriptors == [descriptor, descriptor]
    assert read_descriptors
    assert set(read_descriptors) == {descriptor}
    if getattr(os, "O_NOFOLLOW", 0):
        assert flags & os.O_NOFOLLOW


def test_session_store_load_rejects_symlink_manifest_authority(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path / "run")
    target = tmp_path / "outside-session.json"
    target.write_bytes(store.session_path.read_bytes())
    store.session_path.unlink()
    try:
        store.session_path.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="regular file"):
        store.load()


def test_session_store_load_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    original = store.session_path.read_text(encoding="utf-8")
    duplicate = original.replace(
        '"schema_version": 4,',
        '"schema_version": 4,\n  "schema_version": 4,',
        1,
    )
    store.session_path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
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
        "artifact.json:stream",
        "CON",
        "aux.txt",
        "trailing.",
        "directory/trailing. ",
        "PROGRA~1/artifact.json",
        "cafe\u0301.json",
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
    descriptor_paths: dict[int, Path] = {}
    real_open = session_store_module.os.open
    real_fstat = session_store_module.os.fstat
    real_read = session_store_module.os.read

    def recording_open(path: object, flags: int, *args: object) -> int:
        file_descriptor = real_open(path, flags, *args)
        descriptor_paths[file_descriptor] = Path(path)
        if Path(path) == resolved_artifact:
            artifact_open_calls.append((file_descriptor, flags))
        return file_descriptor

    def recording_fstat(file_descriptor: int):
        if descriptor_paths.get(file_descriptor) == resolved_artifact:
            fstat_descriptors.append(file_descriptor)
        return real_fstat(file_descriptor)

    def recording_read(file_descriptor: int, size: int) -> bytes:
        if descriptor_paths.get(file_descriptor) == resolved_artifact:
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


def test_mark_phase_completed_requires_every_registered_artifact_for_that_phase(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")
    (tmp_path / "intent.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repository.json").write_text("{}", encoding="utf-8")
    register_artifact(store)
    register_artifact(
        store,
        name="intent",
        relative_path="intent.json",
        phase=RunPhase.PREFLIGHT,
        revision_binding=REVISION_BINDING,
    )
    register_artifact(
        store,
        name="repository",
        relative_path="repository.json",
        phase=RunPhase.REPOSITORY_INTELLIGENCE,
        revision_binding=REVISION_BINDING,
    )

    with pytest.raises(ValueError, match="registry|artifact set"):
        store.mark_phase_completed(
            RunPhase.PREFLIGHT,
            ["request"],
            "2026-07-10T00:02:00Z",
        )

    unchanged = store.load()
    assert unchanged.phases["preflight"].status is PhaseStatus.PENDING
    completed = store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        ["request", "intent"],
        "2026-07-10T00:03:00Z",
    )
    assert completed.phases["preflight"].artifacts == ("request", "intent")


def test_mark_phase_completed_idempotence_rejects_same_phase_registry_orphan(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    (tmp_path / "request.json").write_text("{}", encoding="utf-8")
    orphan_path = tmp_path / "orphan.json"
    orphan_path.write_text("{}", encoding="utf-8")
    register_artifact(store)
    completed = store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        ["request"],
        "2026-07-10T00:02:00Z",
    )
    orphan = ArtifactDescriptor(
        name="orphan",
        path="orphan.json",
        sha256=sha256(orphan_path.read_bytes()).hexdigest(),
        schema="orphan_v1",
        phase=RunPhase.PREFLIGHT,
        revision_binding=REVISION_BINDING,
    )
    store.write(
        replace(
            completed,
            artifacts={**completed.artifacts, "orphan": orphan},
            updated_at="2026-07-10T00:03:00Z",
        )
    )

    with pytest.raises(ValueError, match="registry|artifact set"):
        store.mark_phase_completed(
            RunPhase.PREFLIGHT,
            ["request"],
            "2026-07-10T00:04:00Z",
        )


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
    [RunPhase.CREATED, RunPhase.COMPLETED, RunPhase.FAILED],
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
    mark_all_phases(store)
    orphan_path = tmp_path / "orphan.json"
    orphan_path.write_text("{}", encoding="utf-8")
    current = store.load()
    orphan = ArtifactDescriptor(
        name="orphan",
        path="orphan.json",
        sha256=sha256(orphan_path.read_bytes()).hexdigest(),
        schema="orphan_v1",
        phase=RunPhase.PREFLIGHT,
        revision_binding=REVISION_BINDING,
    )
    store.write(
        replace(
            current,
            artifacts={"orphan": orphan},
            updated_at="2026-07-10T00:09:00Z",
        )
    )

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


def _complete_empty_predecessors(
    store: SessionStore,
    through: RunPhase,
) -> None:
    target_index = SESSION_PHASES.index(through)
    for index, phase in enumerate(SESSION_PHASES[:target_index], start=1):
        store.mark_phase_completed(
            phase,
            [],
            f"2026-07-10T01:{index:02d}:00Z",
        )


def _prepare_awaiting_user(store: SessionStore) -> SessionManifest:
    _complete_empty_predecessors(store, RunPhase.INTENT_RESOLUTION)
    store.mark_phase_running(
        RunPhase.INTENT_RESOLUTION,
        "2026-07-10T02:00:00Z",
    )
    for index, name in enumerate(("intent_candidates", "intent_questions"), start=1):
        relative_path = f"{name}.json"
        (store.run_dir / relative_path).write_text("{}", encoding="utf-8")
        register_artifact(
            store,
            name=name,
            relative_path=relative_path,
            phase=RunPhase.INTENT_RESOLUTION,
            revision_binding=REVISION_BINDING,
            now=f"2026-07-10T02:00:{index:02d}Z",
        )
    return store.mark_phase_awaiting_user(
        RunPhase.INTENT_RESOLUTION,
        ["intent_candidates", "intent_questions"],
        "2026-07-10T02:01:00Z",
    )


def test_awaiting_user_transition_is_idempotent_and_not_a_failure(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)

    awaiting = _prepare_awaiting_user(store)
    session_bytes = (tmp_path / "session.json").read_bytes()
    repeated = store.mark_phase_awaiting_user(
        RunPhase.INTENT_RESOLUTION,
        ["intent_candidates", "intent_questions"],
        "2026-07-10T02:02:00Z",
    )

    checkpoint = awaiting.phases[RunPhase.INTENT_RESOLUTION.value]
    assert repeated == awaiting
    assert (tmp_path / "session.json").read_bytes() == session_bytes
    assert awaiting.status is RunStatus.AWAITING_USER
    assert awaiting.current_phase is RunPhase.INTENT_RESOLUTION
    assert awaiting.errors == ()
    assert checkpoint.status is PhaseStatus.AWAITING_USER
    assert checkpoint.attempts == 1
    assert checkpoint.started_at == "2026-07-10T02:00:00Z"
    assert checkpoint.completed_at is None
    assert checkpoint.artifacts == ("intent_candidates", "intent_questions")


def test_submit_user_decision_and_resume_are_idempotent_and_preserve_artifacts(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    awaiting = _prepare_awaiting_user(store)
    (tmp_path / "intent_event_1.json").write_text(
        json.dumps({"event_id": "decision-1", "action": "confirmed"}),
        encoding="utf-8",
    )
    register_artifact(
        store,
        name="intent_event_1",
        relative_path="intent_event_1.json",
        phase=RunPhase.INTENT_RESOLUTION,
        revision_binding=REVISION_BINDING,
        now="2026-07-10T02:02:00Z",
    )

    submitted = store.submit_user_decision(
        "decision-1",
        "intent_event_1",
        "2026-07-10T02:03:00Z",
    )
    session_bytes = (tmp_path / "session.json").read_bytes()
    repeated_submission = store.submit_user_decision(
        "decision-1",
        "intent_event_1",
        "2026-07-10T02:04:00Z",
    )
    repeated_session_bytes = (tmp_path / "session.json").read_bytes()
    resumed = store.resume_awaiting_user("2026-07-10T02:05:00Z")
    repeated_resume = store.resume_awaiting_user("2026-07-10T02:06:00Z")

    submitted_checkpoint = submitted.phases[RunPhase.INTENT_RESOLUTION.value]
    resumed_checkpoint = resumed.phases[RunPhase.INTENT_RESOLUTION.value]
    assert awaiting.status is RunStatus.AWAITING_USER
    assert repeated_submission == submitted
    assert repeated_session_bytes == session_bytes
    assert submitted_checkpoint.user_decisions == {
        "decision-1": "intent_event_1"
    }
    assert submitted_checkpoint.artifacts == (
        "intent_candidates",
        "intent_questions",
        "intent_event_1",
    )
    assert resumed.status is RunStatus.RUNNING
    assert resumed_checkpoint.status is PhaseStatus.RUNNING
    assert resumed_checkpoint.started_at == "2026-07-10T02:00:00Z"
    assert resumed_checkpoint.completed_at is None
    assert resumed_checkpoint.artifacts == submitted_checkpoint.artifacts
    assert resumed_checkpoint.user_decisions == submitted_checkpoint.user_decisions
    assert repeated_resume == resumed


def test_awaiting_user_rejects_illegal_phase_missing_decision_and_tampering(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    with pytest.raises(ValueError, match="only for intent_resolution"):
        store.mark_phase_awaiting_user(
            RunPhase.PLANNING,
            ["intent_questions"],
            "2026-07-10T02:00:00Z",
        )

    _prepare_awaiting_user(store)
    with pytest.raises(ValueError, match="submitted user decision"):
        store.resume_awaiting_user("2026-07-10T02:02:00Z")
    with pytest.raises(ValueError, match="cannot discard.*awaiting_user"):
        store.discard_uncommitted_phase_artifacts(
            RunPhase.INTENT_RESOLUTION,
            ["intent_candidates"],
            "2026-07-10T02:02:30Z",
        )

    (tmp_path / "intent_questions.json").write_text(
        '{"tampered":true}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation"):
        store.mark_phase_awaiting_user(
            RunPhase.INTENT_RESOLUTION,
            ["intent_candidates", "intent_questions"],
            "2026-07-10T02:03:00Z",
        )


def test_submit_user_decision_rejects_new_or_conflicting_event_outside_protocol(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _complete_empty_predecessors(store, RunPhase.INTENT_RESOLUTION)
    store.mark_phase_running(
        RunPhase.INTENT_RESOLUTION,
        "2026-07-10T02:00:00Z",
    )
    (tmp_path / "intent_event_1.json").write_text("{}", encoding="utf-8")
    register_artifact(
        store,
        name="intent_event_1",
        relative_path="intent_event_1.json",
        phase=RunPhase.INTENT_RESOLUTION,
        revision_binding=REVISION_BINDING,
    )
    with pytest.raises(ValueError, match="only while.*awaiting_user"):
        store.submit_user_decision(
            "decision-1",
            "intent_event_1",
            "2026-07-10T02:01:00Z",
        )

    store.discard_uncommitted_phase_artifacts(
        RunPhase.INTENT_RESOLUTION,
        [],
        "2026-07-10T02:01:10Z",
    )
    for name in ("intent_candidates", "intent_questions"):
        (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")
        register_artifact(
            store,
            name=name,
            relative_path=f"{name}.json",
            phase=RunPhase.INTENT_RESOLUTION,
            revision_binding=REVISION_BINDING,
        )
    store.mark_phase_awaiting_user(
        RunPhase.INTENT_RESOLUTION,
        ["intent_candidates", "intent_questions"],
        "2026-07-10T02:02:00Z",
    )
    for name in ("intent_event_1", "intent_event_2"):
        (tmp_path / f"{name}.json").write_text("{}", encoding="utf-8")
        register_artifact(
            store,
            name=name,
            relative_path=f"{name}.json",
            phase=RunPhase.INTENT_RESOLUTION,
            revision_binding=REVISION_BINDING,
        )
    store.submit_user_decision(
        "decision-1",
        "intent_event_1",
        "2026-07-10T02:03:00Z",
    )

    with pytest.raises(ValueError, match="different artifact"):
        store.submit_user_decision(
            "decision-1",
            "intent_event_2",
            "2026-07-10T02:04:00Z",
        )


def _register_reviewer_result(
    store: SessionStore,
    index: int,
    *,
    now: str,
) -> str:
    name = f"reviewer_{index}_result"
    relative_path = f"{name}.json"
    (store.run_dir / relative_path).write_text(
        json.dumps({"status": "completed", "reviewer": index}),
        encoding="utf-8",
    )
    register_artifact(
        store,
        name=name,
        relative_path=relative_path,
        phase=RunPhase.REVIEWERS,
        revision_binding=REVISION_BINDING,
        now=now,
    )
    return name


def test_mark_phase_running_is_idempotent_and_counts_new_attempts(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)

    first = store.mark_phase_running(RunPhase.PREFLIGHT, "2026-07-10T01:00:00Z")
    repeated = store.mark_phase_running(RunPhase.PREFLIGHT, "2026-07-10T01:01:00Z")
    store.mark_session_failed(
        RunPhase.PREFLIGHT,
        "transient failure",
        "2026-07-10T01:02:00Z",
    )
    retried = store.mark_phase_running(
        RunPhase.PREFLIGHT,
        "2026-07-10T01:03:00Z",
    )

    assert first.phases["preflight"].attempts == 1
    assert repeated == first
    assert retried.phases["preflight"].status is PhaseStatus.RUNNING
    assert retried.phases["preflight"].attempts == 2
    assert retried.phases["preflight"].started_at == "2026-07-10T01:03:00Z"


def test_validate_phase_checks_status_schema_hash_and_revision_binding(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    artifact_path = tmp_path / "request.json"
    artifact_path.write_text("{}", encoding="utf-8")
    register_artifact(store)
    store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        ["request"],
        "2026-07-10T01:00:00Z",
    )

    valid = store.validate_phase(
        RunPhase.PREFLIGHT,
        {"request": "review_request_v1"},
    )
    wrong_schema = store.validate_phase(
        RunPhase.PREFLIGHT,
        {"request": "review_request_v2"},
    )
    artifact_path.write_text('{"tampered":true}', encoding="utf-8")
    bad_hash = store.validate_phase(RunPhase.PREFLIGHT)

    assert valid == PhaseValidation(RunPhase.PREFLIGHT, True)
    assert wrong_schema.valid is False
    assert "schema" in str(wrong_schema.reason)
    assert bad_hash.valid is False
    assert "validation" in str(bad_hash.reason)


def test_invalidate_from_preserves_upstream_and_removes_downstream_registry(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    phase_artifacts: dict[RunPhase, str] = {}
    through_reviewers = SESSION_PHASES[: SESSION_PHASES.index(RunPhase.REVIEWERS) + 1]
    for index, phase in enumerate(through_reviewers):
        name = "request" if phase is RunPhase.PREFLIGHT else f"artifact_{index}"
        relative_path = f"{name}.json"
        (tmp_path / relative_path).write_text("{}", encoding="utf-8")
        register_artifact(
            store,
            name=name,
            relative_path=relative_path,
            phase=phase,
            revision_binding=(
                None if phase is RunPhase.PREFLIGHT else REVISION_BINDING
            ),
            now=f"2026-07-10T01:{index:02d}:00Z",
        )
        store.mark_phase_completed(
            phase,
            [name],
            f"2026-07-10T01:{index:02d}:30Z",
        )
        phase_artifacts[phase] = name

    invalidated = store.invalidate_from(
        RunPhase.REPOSITORY_INTELLIGENCE,
        "artifact hash mismatch",
        "2026-07-10T01:10:00Z",
    )

    assert invalidated.phases["preflight"].status is PhaseStatus.COMPLETED
    assert invalidated.phases["preflight"].artifacts == ("request",)
    assert all(
        invalidated.phases[phase.value].status is PhaseStatus.INVALIDATED
        for phase in SESSION_PHASES[
            SESSION_PHASES.index(RunPhase.REPOSITORY_INTELLIGENCE) :
        ]
    )
    assert set(invalidated.artifacts) == {
        phase_artifacts[RunPhase.PREFLIGHT],
        phase_artifacts[RunPhase.QUALITY_GATES],
    }
    assert invalidated.last_successful_phase is RunPhase.QUALITY_GATES
    assert invalidated.current_phase is RunPhase.REPOSITORY_INTELLIGENCE
    assert (tmp_path / f"{phase_artifacts[RunPhase.REVIEWERS]}.json").exists()


def test_invalidate_from_can_reopen_completed_session_only_through_recovery_api(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    mark_all_phases(store)
    completed = store.mark_session_completed("2026-07-10T01:10:00Z")

    recovered = store.invalidate_from(
        RunPhase.REVIEWERS,
        "reviewer artifact missing",
        "2026-07-10T01:11:00Z",
    )

    assert completed.status is RunStatus.COMPLETED
    assert recovered.status is RunStatus.RUNNING
    assert recovered.phases["preflight"].status is PhaseStatus.COMPLETED
    assert recovered.phases["planning"].status is PhaseStatus.COMPLETED
    assert recovered.phases["reviewers"].status is PhaseStatus.INVALIDATED


def test_invalidate_from_is_idempotent_for_same_phase_and_reason(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    store.mark_phase_completed(
        RunPhase.PREFLIGHT,
        [],
        "2026-07-10T01:00:00Z",
    )
    first = store.invalidate_from(
        RunPhase.REPOSITORY_INTELLIGENCE,
        "missing repository artifact",
        "2026-07-10T01:01:00Z",
    )
    session_bytes = (tmp_path / "session.json").read_bytes()

    repeated = store.invalidate_from(
        RunPhase.REPOSITORY_INTELLIGENCE,
        "missing repository artifact",
        "2026-07-10T01:02:00Z",
    )

    assert repeated == first
    assert repeated.errors.count(
        "invalidated repository_intelligence: missing repository artifact"
    ) == 1
    assert (tmp_path / "session.json").read_bytes() == session_bytes


def test_reviewer_tasks_resume_only_failed_task_and_preserve_completed_task(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _complete_empty_predecessors(store, RunPhase.REVIEWERS)
    store.mark_phase_running(RunPhase.REVIEWERS, "2026-07-10T02:00:00Z")
    store.initialize_reviewer_tasks(
        ["reviewer-0", "reviewer-1"],
        "2026-07-10T02:00:10Z",
    )

    store.mark_reviewer_task_running("reviewer-0", "2026-07-10T02:01:00Z")
    artifact_0 = _register_reviewer_result(
        store,
        0,
        now="2026-07-10T02:01:10Z",
    )
    completed_zero = store.mark_reviewer_task_completed(
        "reviewer-0",
        [artifact_0],
        "2026-07-10T02:01:20Z",
    )
    store.mark_reviewer_task_running("reviewer-1", "2026-07-10T02:02:00Z")
    store.mark_reviewer_task_failed(
        "reviewer-1",
        "provider timeout",
        "2026-07-10T02:02:10Z",
    )

    store.mark_phase_running(RunPhase.REVIEWERS, "2026-07-10T02:03:00Z")
    reused_zero = store.mark_reviewer_task_running(
        "reviewer-0",
        "2026-07-10T02:03:10Z",
    )
    store.mark_reviewer_task_running("reviewer-1", "2026-07-10T02:03:20Z")
    artifact_1 = _register_reviewer_result(
        store,
        1,
        now="2026-07-10T02:03:30Z",
    )
    store.mark_reviewer_task_completed(
        "reviewer-1",
        [artifact_1],
        "2026-07-10T02:03:40Z",
    )
    finished = store.mark_phase_completed(
        RunPhase.REVIEWERS,
        [artifact_0, artifact_1],
        "2026-07-10T02:04:00Z",
    )

    assert completed_zero.phases["reviewers"].tasks["reviewer-0"].attempts == 1
    assert reused_zero.phases["reviewers"].tasks["reviewer-0"].attempts == 1
    assert finished.phases["reviewers"].tasks["reviewer-0"].attempts == 1
    assert finished.phases["reviewers"].tasks["reviewer-1"].attempts == 2
    assert finished.phases["reviewers"].status is PhaseStatus.COMPLETED
    assert store.validate_phase(RunPhase.REVIEWERS).valid is True


def test_reviewer_phase_cannot_complete_with_pending_task(tmp_path: Path) -> None:
    store = create_store(tmp_path)
    _complete_empty_predecessors(store, RunPhase.REVIEWERS)
    store.mark_phase_running(RunPhase.REVIEWERS, "2026-07-10T03:00:00Z")
    store.initialize_reviewer_tasks(
        ["reviewer-0"],
        "2026-07-10T03:00:10Z",
    )

    with pytest.raises(ValueError, match="reviewer tasks complete"):
        store.mark_phase_completed(
            RunPhase.REVIEWERS,
            [],
            "2026-07-10T03:01:00Z",
        )


def test_reviewer_task_must_enter_running_before_terminal_transition(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _complete_empty_predecessors(store, RunPhase.REVIEWERS)
    store.mark_phase_running(RunPhase.REVIEWERS, "2026-07-10T03:10:00Z")
    store.initialize_reviewer_tasks(
        ["reviewer-0"],
        "2026-07-10T03:10:10Z",
    )
    artifact = _register_reviewer_result(
        store,
        0,
        now="2026-07-10T03:10:20Z",
    )

    with pytest.raises(ValueError, match="must be running before completion"):
        store.mark_reviewer_task_completed(
            "reviewer-0",
            [artifact],
            "2026-07-10T03:10:30Z",
        )
    with pytest.raises(ValueError, match="must be running before failure"):
        store.mark_reviewer_task_failed(
            "reviewer-0",
            "provider unavailable",
            "2026-07-10T03:10:40Z",
        )


def test_mark_session_failed_preserves_last_successful_phase_and_appends_errors(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _complete_empty_predecessors(store, RunPhase.REPOSITORY_INTELLIGENCE)

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
    assert failed_again.last_successful_phase is RunPhase.QUALITY_GATES
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
            RunPhase.CREATED,
            "failed",
            "2026-07-10T00:02:00Z",
        )


def _prepare_supplemental_phase(store: SessionStore) -> None:
    _complete_empty_predecessors(store, RunPhase.SUPPLEMENTAL_INVESTIGATION)
    store.mark_phase_running(
        RunPhase.SUPPLEMENTAL_INVESTIGATION,
        "2026-07-10T03:00:00Z",
    )


def _register_supplemental_artifact(
    store: SessionStore,
    name: str,
    now: str,
) -> None:
    (store.run_dir / f"{name}.json").write_text("{}", encoding="utf-8")
    register_artifact(
        store,
        name=name,
        relative_path=f"{name}.json",
        phase=RunPhase.SUPPLEMENTAL_INVESTIGATION,
        revision_binding=REVISION_BINDING,
        now=now,
    )


def test_session_store_atomically_tracks_wave_task_and_budget_lifecycle(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _prepare_supplemental_phase(store)
    reservation = SupplementalBudget(
        tasks=1,
        tool_calls=8,
        tokens=8192,
        elapsed_seconds=30,
    )
    charged = SupplementalBudget(
        tasks=1,
        tool_calls=5,
        tokens=4096,
        elapsed_seconds=12.5,
    )

    initialized = store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1, TASK_2: ASSIGNMENT_2},
        "2026-07-10T03:01:00Z",
        trigger_digest=TRIGGER_1,
    )
    repeated = store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1, TASK_2: ASSIGNMENT_2},
        "2026-07-10T03:02:00Z",
        trigger_digest=TRIGGER_1,
    )
    assert repeated == initialized

    reserved = store.reserve_task_budget(
        TASK_1,
        reservation,
        "2026-07-10T03:03:00Z",
    )
    assert (
        reserved.supplemental_waves[WAVE_1].tasks[TASK_1].status
        is SupplementalTaskStatus.RESERVED
    )
    store.mark_task_running(TASK_1, "2026-07-10T03:04:00Z")
    _register_supplemental_artifact(
        store,
        "supplemental_task_1_result",
        "2026-07-10T03:05:00Z",
    )
    completed = store.mark_task_completed(
        TASK_1,
        ["supplemental_task_1_result"],
        charged,
        "2026-07-10T03:06:00Z",
    )
    completed_task = completed.supplemental_waves[WAVE_1].tasks[TASK_1]
    assert completed_task.status is SupplementalTaskStatus.COMPLETED
    assert completed_task.reservation == SupplementalBudget()
    assert completed_task.charged == charged

    store.reserve_task_budget(TASK_2, reservation, "2026-07-10T03:07:00Z")
    store.mark_task_running(TASK_2, "2026-07-10T03:08:00Z")
    failed = store.mark_task_failed(
        TASK_2,
        "provider unavailable",
        charged,
        "2026-07-10T03:09:00Z",
    )
    assert (
        failed.supplemental_waves[WAVE_1].tasks[TASK_2].status
        is SupplementalTaskStatus.FAILED
    )

    _register_supplemental_artifact(
        store,
        "supplemental_wave_1_summary",
        "2026-07-10T03:10:00Z",
    )
    finished = store.mark_wave_completed(
        WAVE_1,
        ["supplemental_task_1_result", "supplemental_wave_1_summary"],
        "task_failure",
        "2026-07-10T03:11:00Z",
    )
    wave = finished.supplemental_waves[WAVE_1]
    assert wave.status is PhaseStatus.COMPLETED
    assert wave.stop_reason == "task_failure"
    assert wave.tasks[TASK_1].charged == charged


def test_session_store_marks_running_reservation_as_unknown_before_retry(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _prepare_supplemental_phase(store)
    reservation = SupplementalBudget(
        tasks=1,
        tool_calls=4,
        tokens=2048,
        elapsed_seconds=15,
    )
    invocation_id = f"INV-{'b' * 64}"
    store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1},
        "2026-07-10T03:01:00Z",
        trigger_digest=TRIGGER_1,
    )
    store.reserve_task_budget(TASK_1, reservation, "2026-07-10T03:02:00Z")
    store.mark_task_running(TASK_1, "2026-07-10T03:03:00Z")

    unknown = store.mark_task_unknown(
        TASK_1,
        invocation_id,
        "result returned before checkpoint commit",
        "2026-07-10T03:04:00Z",
    )
    task = unknown.supplemental_waves[WAVE_1].tasks[TASK_1]
    assert task.status is SupplementalTaskStatus.FAILED
    assert task.reservation == SupplementalBudget()
    assert task.unknown_consumed == reservation
    assert task.unknown_invocation_ids == (invocation_id,)

    retried = store.reserve_task_budget(
        TASK_1,
        reservation,
        "2026-07-10T03:05:00Z",
    )
    retried_task = retried.supplemental_waves[WAVE_1].tasks[TASK_1]
    assert retried_task.status is SupplementalTaskStatus.RESERVED
    assert retried_task.unknown_consumed == reservation


def test_session_store_rejects_budget_oversubscription_without_mutation(
    tmp_path: Path,
) -> None:
    original = manifest()
    low = replace(
        original,
        execution=replace(
            original.execution,
            supplemental_policy=SupplementalPolicy.for_risk("low"),
        ),
    )
    store = SessionStore(tmp_path)
    store.create(low)
    _prepare_supplemental_phase(store)
    store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1},
        "2026-07-10T03:01:00Z",
        trigger_digest=TRIGGER_1,
    )
    before = store.load()

    with pytest.raises(ValueError, match="max_tokens_per_task|budget"):
        store.reserve_task_budget(
            TASK_1,
            SupplementalBudget(tasks=1, tokens=16385, elapsed_seconds=1),
            "2026-07-10T03:02:00Z",
        )

    assert store.load() == before


def test_invalidate_wave_from_preserves_earlier_wave_and_unaffected_task(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _prepare_supplemental_phase(store)
    reservation = SupplementalBudget(tasks=1, tokens=1024, elapsed_seconds=5)

    store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1},
        "2026-07-10T03:01:00Z",
        trigger_digest=TRIGGER_1,
    )
    store.reserve_task_budget(TASK_1, reservation, "2026-07-10T03:02:00Z")
    store.mark_task_running(TASK_1, "2026-07-10T03:03:00Z")
    _register_supplemental_artifact(store, "wave_1_task", "2026-07-10T03:04:00Z")
    store.mark_task_completed(
        TASK_1,
        ["wave_1_task"],
        reservation,
        "2026-07-10T03:05:00Z",
    )
    _register_supplemental_artifact(store, "wave_1_summary", "2026-07-10T03:06:00Z")
    store.mark_wave_completed(
        WAVE_1,
        ["wave_1_task", "wave_1_summary"],
        "no_requests",
        "2026-07-10T03:07:00Z",
    )

    store.initialize_wave(
        WAVE_2,
        {TASK_2: ASSIGNMENT_2, TASK_3: ASSIGNMENT_3},
        "2026-07-10T03:08:00Z",
        trigger_digest=TRIGGER_2,
    )
    for offset, (task_id, artifact_name) in enumerate(
        ((TASK_2, "wave_2_task_2"), (TASK_3, "wave_2_task_3")),
        start=9,
    ):
        store.reserve_task_budget(
            task_id,
            reservation,
            f"2026-07-10T03:{offset:02d}:00Z",
        )
        store.mark_task_running(
            task_id,
            f"2026-07-10T03:{offset + 1:02d}:00Z",
        )
        _register_supplemental_artifact(
            store,
            artifact_name,
            f"2026-07-10T03:{offset + 2:02d}:00Z",
        )
        store.mark_task_completed(
            task_id,
            [artifact_name],
            reservation,
            f"2026-07-10T03:{offset + 3:02d}:00Z",
        )
    _register_supplemental_artifact(store, "wave_2_summary", "2026-07-10T03:20:00Z")
    store.mark_wave_completed(
        WAVE_2,
        ["wave_2_task_2", "wave_2_task_3", "wave_2_summary"],
        "no_requests",
        "2026-07-10T03:21:00Z",
    )

    invalidated = store.invalidate_wave_from(
        WAVE_2,
        "task artifact hash mismatch",
        "2026-07-10T03:22:00Z",
        task_id=TASK_2,
    )

    assert invalidated.supplemental_waves[WAVE_1].status is PhaseStatus.COMPLETED
    assert (
        invalidated.supplemental_waves[WAVE_2].tasks[TASK_3].status
        is SupplementalTaskStatus.COMPLETED
    )
    assert (
        invalidated.supplemental_waves[WAVE_2].tasks[TASK_2].status
        is SupplementalTaskStatus.INVALIDATED
    )
    assert "wave_1_task" in invalidated.artifacts
    assert "wave_2_task_3" in invalidated.artifacts
    assert "wave_2_task_2" not in invalidated.artifacts
    assert "wave_2_summary" not in invalidated.artifacts
    assert (
        invalidated.phases[RunPhase.RECONCILIATION_ANALYSIS.value].status
        is PhaseStatus.COMPLETED
    )
    assert (
        invalidated.phases[RunPhase.SUPPLEMENTAL_INVESTIGATION.value].status
        is PhaseStatus.INVALIDATED
    )


def test_wave_level_invalidation_preserves_committed_tasks_and_wave_plan(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _prepare_supplemental_phase(store)
    reservation = SupplementalBudget(tasks=1, tokens=1024, elapsed_seconds=5)
    plan_name = f"supplemental_wave_{WAVE_1}_plan"
    task_name = "wave_level_task"
    summary_name = "wave_level_summary"

    store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1},
        "2026-07-10T03:01:00Z",
        trigger_digest=TRIGGER_1,
    )
    _register_supplemental_artifact(store, plan_name, "2026-07-10T03:02:00Z")
    store.reserve_task_budget(TASK_1, reservation, "2026-07-10T03:03:00Z")
    store.mark_task_running(TASK_1, "2026-07-10T03:04:00Z")
    _register_supplemental_artifact(store, task_name, "2026-07-10T03:05:00Z")
    store.mark_task_completed(
        TASK_1,
        [task_name],
        reservation,
        "2026-07-10T03:06:00Z",
    )
    _register_supplemental_artifact(store, summary_name, "2026-07-10T03:07:00Z")
    store.mark_wave_completed(
        WAVE_1,
        [plan_name, task_name, summary_name],
        "resolved",
        "2026-07-10T03:08:00Z",
    )

    invalidated = store.invalidate_wave_from(
        WAVE_1,
        "wave summary hash mismatch",
        "2026-07-10T03:09:00Z",
    )

    wave = invalidated.supplemental_waves[WAVE_1]
    assert wave.status is PhaseStatus.INVALIDATED
    assert wave.tasks[TASK_1].status is SupplementalTaskStatus.COMPLETED
    assert task_name in invalidated.artifacts
    assert plan_name in invalidated.artifacts
    assert summary_name not in invalidated.artifacts


def test_effective_risk_policy_bounds_retry_and_marks_task_unrunnable(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _prepare_supplemental_phase(store)
    effective_policy = SupplementalPolicy.for_risk("low")
    reservation = SupplementalBudget(
        tasks=1,
        tool_calls=8,
        tokens=16_384,
        elapsed_seconds=120,
    )

    store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1},
        "2026-07-10T03:01:00Z",
        trigger_digest=TRIGGER_1,
        effective_policy=effective_policy,
    )
    store.reserve_task_budget(TASK_1, reservation, "2026-07-10T03:02:00Z")
    store.mark_task_running(TASK_1, "2026-07-10T03:03:00Z")
    _register_supplemental_artifact(store, "low_risk_result", "2026-07-10T03:04:00Z")
    store.mark_task_completed(
        TASK_1,
        ["low_risk_result"],
        reservation,
        "2026-07-10T03:05:00Z",
    )
    store.invalidate_wave_from(
        WAVE_1,
        "task artifact hash mismatch",
        "2026-07-10T03:06:00Z",
        task_id=TASK_1,
    )
    store.mark_phase_running(
        RunPhase.SUPPLEMENTAL_INVESTIGATION,
        "2026-07-10T03:07:00Z",
    )
    store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1},
        "2026-07-10T03:08:00Z",
        trigger_digest=TRIGGER_1,
        effective_policy=effective_policy,
    )

    with pytest.raises(ValueError, match="remaining global budget"):
        store.reserve_task_budget(
            TASK_1,
            reservation,
            "2026-07-10T03:09:00Z",
        )

    exhausted = store.mark_task_unrunnable(
        TASK_1,
        "supplemental budget exhausted before another task attempt",
        "2026-07-10T03:10:00Z",
    )
    task = exhausted.supplemental_waves[WAVE_1].tasks[TASK_1]
    assert task.status is SupplementalTaskStatus.FAILED
    assert task.charged == reservation
    assert task.attempts == 1
    assert task.error == "supplemental budget exhausted before another task attempt"


def test_retried_supplemental_task_accumulates_durable_charges(
    tmp_path: Path,
) -> None:
    store = create_store(tmp_path)
    _prepare_supplemental_phase(store)
    first_charge = SupplementalBudget(
        tasks=1,
        tool_calls=2,
        tokens=100,
        elapsed_seconds=3,
    )
    second_charge = SupplementalBudget(
        tasks=1,
        tool_calls=1,
        tokens=50,
        elapsed_seconds=2,
    )
    reservation = SupplementalBudget(
        tasks=1,
        tool_calls=4,
        tokens=200,
        elapsed_seconds=5,
    )

    store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1},
        "2026-07-10T03:01:00Z",
        trigger_digest=TRIGGER_1,
    )
    store.reserve_task_budget(TASK_1, reservation, "2026-07-10T03:02:00Z")
    store.mark_task_running(TASK_1, "2026-07-10T03:03:00Z")
    _register_supplemental_artifact(store, "first_result", "2026-07-10T03:04:00Z")
    store.mark_task_completed(
        TASK_1,
        ["first_result"],
        first_charge,
        "2026-07-10T03:05:00Z",
    )
    store.invalidate_wave_from(
        WAVE_1,
        "task artifact hash mismatch",
        "2026-07-10T03:06:00Z",
        task_id=TASK_1,
    )
    store.mark_phase_running(
        RunPhase.SUPPLEMENTAL_INVESTIGATION,
        "2026-07-10T03:07:00Z",
    )
    store.initialize_wave(
        WAVE_1,
        {TASK_1: ASSIGNMENT_1},
        "2026-07-10T03:08:00Z",
        trigger_digest=TRIGGER_1,
    )
    store.reserve_task_budget(TASK_1, reservation, "2026-07-10T03:09:00Z")
    store.mark_task_running(TASK_1, "2026-07-10T03:10:00Z")
    _register_supplemental_artifact(store, "second_result", "2026-07-10T03:11:00Z")
    retried = store.mark_task_completed(
        TASK_1,
        ["second_result"],
        second_charge,
        "2026-07-10T03:12:00Z",
    )

    task = retried.supplemental_waves[WAVE_1].tasks[TASK_1]
    assert task.attempts == 2
    assert task.charged == first_charge + second_charge


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

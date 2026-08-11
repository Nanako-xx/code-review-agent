from __future__ import annotations

from pathlib import Path

import pytest

from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.resume import resume_session_v6
from review_agent.revision import RepositoryIdentity
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    SESSION_V6_PHASES,
    SESSION_V6_SCHEMA_VERSION,
    PhaseStatus,
    SessionV6ArtifactRef,
    SessionV6Manifest,
    new_session_v6_manifest,
    session_v6_manifest_from_dict,
)
from review_agent.session_store import SessionV6Store


def _runtime(tmp_path: Path):
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    git_common.mkdir(parents=True)
    identity = RepositoryIdentity(
        canonical_path=str(repository.resolve()),
        git_common_dir=str(git_common.resolve()),
        origin_url=None,
    )
    workspace_store = PRWorkspaceStore(tmp_path / "ra")
    workspace = workspace_store.create_or_load_workspace(
        workspace_store.resolve_pr(identity, "local", "session-v6"),
        PRMetadata(title="Session v6"),
    )
    snapshot = workspace_store.create_or_load_snapshot(
        workspace, "a" * 40, "b" * 40
    )
    session = workspace_store.create_session(workspace, snapshot)
    return workspace_store, session, SessionV6Store(workspace_store, session)


def _artifact(
    store: SessionV6Store,
    logical_name: str,
    relative_path: str,
) -> SessionV6ArtifactRef:
    descriptor = store.workspace_store.publish_create_only(
        store.session.snapshot,
        relative_path,
        logical_name.encode("utf-8"),
    )
    return SessionV6ArtifactRef(
        logical_name=logical_name,
        artifact_id=descriptor.artifact_id,
        relative_path=descriptor.relative_path,
        sha256=descriptor.sha256,
    )


def _complete_preflight(store: SessionV6Store) -> None:
    store.start_phase(RunPhase.PREFLIGHT)
    store.complete_phase(
        RunPhase.PREFLIGHT,
        (
            _artifact(store, "preflight.diff_patch", "DiffArtifact/diff.patch"),
            _artifact(store, "preflight.diff_index", "DiffArtifact/index.json"),
            _artifact(
                store,
                "preflight.quality_gate",
                "QualityGate/quality-gate.json",
            ),
            _artifact(
                store,
                "preflight.changed_symbols",
                "ChangedSymbols/changed-symbols.json",
            ),
        ),
    )


def test_new_session_v6_has_exactly_five_phases_and_canonical_round_trip(
    tmp_path: Path,
) -> None:
    _workspace_store, session, store = _runtime(tmp_path)

    manifest = store.create()
    round_trip = session_v6_manifest_from_dict(manifest.to_dict())

    assert manifest.schema_version == SESSION_V6_SCHEMA_VERSION == 6
    assert tuple(manifest.phases) == tuple(phase.value for phase in SESSION_V6_PHASES)
    assert SESSION_V6_PHASES == (
        RunPhase.PREFLIGHT,
        RunPhase.INTENT,
        RunPhase.PLANNING,
        RunPhase.REVIEWERS,
        RunPhase.AGGREGATION,
    )
    assert manifest.status is RunStatus.CREATED
    assert manifest.current_phase is RunPhase.PREFLIGHT
    assert round_trip == manifest
    assert (session.path / "pipeline-state.json").is_file()


def test_phase_predecessor_and_artifact_ownership_are_enforced(tmp_path: Path) -> None:
    _workspace_store, _session, store = _runtime(tmp_path)
    store.create()

    with pytest.raises(ValueError, match="predecessor"):
        store.start_phase(RunPhase.PLANNING)

    store.start_phase(RunPhase.PREFLIGHT)
    with pytest.raises(ValueError, match="ownership"):
        store.complete_phase(
            RunPhase.PREFLIGHT,
            (_artifact(store, "intent.packet", "Intent/current.json"),),
        )


def test_running_phase_restarts_but_completed_phase_is_reused(tmp_path: Path) -> None:
    _workspace_store, _session, store = _runtime(tmp_path)
    store.create()
    first = store.start_phase(RunPhase.PREFLIGHT)
    restarted = store.start_phase(RunPhase.PREFLIGHT)

    assert first.reused is False
    assert restarted.restarted is True
    assert restarted.manifest.phases["preflight"].attempt == 2

    store.complete_phase(
        RunPhase.PREFLIGHT,
        (
            _artifact(
                store,
                "preflight.diff_patch",
                "DiffArtifact/diff.patch",
            ),
        ),
    )
    reused = store.start_phase(RunPhase.PREFLIGHT)

    assert reused.reused is True
    assert reused.manifest.phases["preflight"].attempt == 2
    assert reused.manifest.current_phase is RunPhase.INTENT


def test_invalidation_keeps_predecessor_and_restarts_only_suffix(tmp_path: Path) -> None:
    _workspace_store, _session, store = _runtime(tmp_path)
    store.create()
    _complete_preflight(store)
    store.start_phase(RunPhase.INTENT)
    store.complete_phase(
        RunPhase.INTENT,
        (_artifact(store, "intent.packet", "Intent/current.json"),),
    )
    invalidated = store.invalidate_from(RunPhase.INTENT, "intent_changed")

    assert invalidated.phases["preflight"].status is PhaseStatus.COMPLETED
    for phase in SESSION_V6_PHASES[1:]:
        assert invalidated.phases[phase.value].status is PhaseStatus.INVALIDATED
    restarted = store.start_phase(RunPhase.INTENT)
    assert restarted.manifest.phases["intent"].attempt == 2
    assert restarted.manifest.phases["planning"].status is PhaseStatus.INVALIDATED


def test_failed_phase_can_restart_without_reusing_failure(tmp_path: Path) -> None:
    _workspace_store, _session, store = _runtime(tmp_path)
    store.create()
    started = store.start_phase(RunPhase.PREFLIGHT)
    assert started.manifest.revision == 1
    failed = store.fail_phase(RunPhase.PREFLIGHT, "preflight_failed")

    assert failed.status is RunStatus.FAILED
    assert store.next_incomplete_phase() is RunPhase.PREFLIGHT
    retried = store.start_phase(RunPhase.PREFLIGHT)
    assert retried.manifest.status is RunStatus.RUNNING
    assert retried.manifest.phases["preflight"].attempt == 2


def test_v6_resume_reuses_completed_prefix_and_selects_first_incomplete(
    tmp_path: Path,
) -> None:
    _workspace_store, _session, store = _runtime(tmp_path)
    store.create()
    _complete_preflight(store)

    resumed = resume_session_v6(store)

    assert resumed.starting_phase is RunPhase.INTENT
    assert resumed.reused_phases == (RunPhase.PREFLIGHT,)


def test_completed_phase_artifact_is_hash_verified_on_load(tmp_path: Path) -> None:
    _workspace_store, session, store = _runtime(tmp_path)
    store.create()
    _complete_preflight(store)
    (session.snapshot.path / "DiffArtifact" / "diff.patch").write_bytes(b"tampered")

    with pytest.raises(ValueError):
        store.load()


def test_manifest_rejects_completed_phase_after_incomplete_prefix() -> None:
    manifest = new_session_v6_manifest(
        session_id="SESSION-" + "a" * 64,
        pr_id="PR-" + "b" * 64,
        snapshot_id="S-" + "c" * 64,
    )
    payload = manifest.to_dict()
    payload["status"] = "running"
    payload["current_phase"] = "preflight"
    payload["phases"]["intent"] = {
        "status": "completed",
        "attempt": 1,
        "artifacts": [],
        "error_code": None,
        "invalidation_reason": None,
    }

    with pytest.raises(ValueError, match="completed Phases"):
        SessionV6Manifest.from_dict(payload)

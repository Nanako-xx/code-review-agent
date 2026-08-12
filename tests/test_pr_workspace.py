from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_agent.pr_workspace import (
    PRMetadata,
    PRWorkspaceError,
    PRWorkspaceSecurityError,
    PRWorkspaceStore,
)
from review_agent.revision import (
    CanonicalRepositoryIdentity,
    RepositoryIdentity,
    canonical_repository_identity,
)
from review_agent.safe_io import canonical_json_bytes


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
NEW_HEAD_SHA = "c" * 40


def _repository_identity(tmp_path: Path) -> RepositoryIdentity:
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    git_common.mkdir(parents=True)
    return RepositoryIdentity(
        canonical_path=str(repository.resolve()),
        git_common_dir=str(git_common.resolve()),
        origin_url="https://github.com/example/project.git",
    )


def _metadata() -> PRMetadata:
    return PRMetadata(
        title="Preserve retry idempotency",
        description="Avoid duplicate jobs when a provider retries.",
        base_ref="main",
        head_ref="feature/retry-fix",
        author="amy",
        status="open",
    )


def test_repository_and_pr_identity_are_canonical_and_stable(tmp_path: Path) -> None:
    identity = _repository_identity(tmp_path)
    canonical = canonical_repository_identity(identity)
    store = PRWorkspaceStore(tmp_path / "ra")

    first = store.resolve_pr(identity, "GitHub", "42")
    second = store.resolve_pr(identity, "github", "42")
    other = store.resolve_pr(identity, "github", "43")

    assert first == second
    assert first.pr_id.startswith("PR-") and len(first.pr_id) == 67
    assert first.repository == canonical
    assert first.provider == "github"
    assert other.pr_id != first.pr_id

    with pytest.raises(ValueError, match="hash does not match"):
        CanonicalRepositoryIdentity(
            repository_key="f" * 64,
            git_common_dir=canonical.git_common_dir,
            origin_url=canonical.origin_url,
        )


def test_workspace_uses_canonical_layout_and_short_physical_ids(
    tmp_path: Path,
) -> None:
    identity = _repository_identity(tmp_path)
    store = PRWorkspaceStore(tmp_path / "short-root")
    resolved = store.resolve_pr(identity, "local", "task-123")
    workspace = store.create_or_load_workspace(resolved, _metadata())

    assert workspace.path.parent == store.root / "pr"
    assert workspace.path.name == "p-" + resolved.pr_id[3:35]
    assert resolved.pr_id not in str(workspace.path)
    assert not str(store.root).startswith("\\\\?\\")
    assert not workspace.path.is_symlink()

    expected = {
        "manifest.json",
        "PR/pr.json",
        "Intent/history",
        "Snapshots",
        "Sessions",
    }
    for relative in expected:
        assert (workspace.path / relative).exists(), relative

    manifest = json.loads((workspace.path / "manifest.json").read_text("utf-8"))
    assert manifest == {
        "current_intent_version": None,
        "current_snapshot_id": None,
        "pr_id": resolved.pr_id,
        "workspace_schema_version": "pr_workspace_manifest_v1",
    }
    pr_payload = json.loads((workspace.path / "PR" / "pr.json").read_text("utf-8"))
    assert pr_payload["repository_identity"] == resolved.repository.to_dict()
    assert pr_payload["pr_number_or_external_review_id"] == "task-123"


def test_same_commits_reuse_snapshot_and_new_head_preserves_old_snapshot(
    tmp_path: Path,
) -> None:
    identity = _repository_identity(tmp_path)
    store = PRWorkspaceStore(tmp_path / "ra")
    workspace = store.create_or_load_workspace(
        store.resolve_pr(identity, "local", "task-1"),
        _metadata(),
    )

    first = store.create_or_load_snapshot(workspace, BASE_SHA, HEAD_SHA)
    repeated = store.create_or_load_snapshot(workspace, BASE_SHA, HEAD_SHA)
    old_manifest = (first.path / "snapshot.json").read_bytes()
    newer = store.create_or_load_snapshot(workspace, BASE_SHA, NEW_HEAD_SHA)

    assert repeated == first
    assert newer.snapshot_id != first.snapshot_id
    assert first.path.name == "s-" + first.snapshot_id[2:34]
    assert (first.path / "snapshot.json").read_bytes() == old_manifest
    assert json.loads((workspace.path / "manifest.json").read_text("utf-8"))[
        "current_snapshot_id"
    ] == newer.snapshot_id

    for relative in (
        "DiffArtifact",
        "QualityGate",
        "ChangedSymbols",
        "Risk",
        "ReviewPlan/Assignments",
        "ToolResults/artifacts",
        "Results",
    ):
        assert (first.path / relative).is_dir(), relative


def test_workspace_and_snapshot_manifests_fail_closed_on_physical_id_collision(
    tmp_path: Path,
) -> None:
    identity = _repository_identity(tmp_path)
    store = PRWorkspaceStore(tmp_path / "ra")
    resolved = store.resolve_pr(identity, "local", "task-1")
    workspace = store.create_or_load_workspace(resolved, _metadata())
    manifest_path = workspace.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["pr_id"] = "PR-" + "f" * 64
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(PRWorkspaceSecurityError, match="binding|collision"):
        store.create_or_load_workspace(resolved, _metadata())


def test_create_session_binds_pr_and_snapshot_and_creates_runtime_files(
    tmp_path: Path,
) -> None:
    identity = _repository_identity(tmp_path)
    store = PRWorkspaceStore(tmp_path / "ra")
    workspace = store.create_or_load_workspace(
        store.resolve_pr(identity, "local", "task-1"),
        _metadata(),
    )
    snapshot = store.create_or_load_snapshot(workspace, BASE_SHA, HEAD_SHA)
    session_id = "SESSION-" + "d" * 64

    session = store.create_session(
        workspace,
        snapshot,
        session_id=session_id,
    )

    assert session.path.name == "u-" + "d" * 32
    assert session.session_id == session_id
    assert (session.path / "state.json").is_file()
    assert (session.path / "execution-log.jsonl").read_bytes() == b""
    context = json.loads((session.path / "context-manifest.json").read_text("utf-8"))
    assert context["snapshot_id"] == snapshot.snapshot_id
    assert context["compaction_generation"] == 0

    with pytest.raises(PRWorkspaceError, match="already exists"):
        store.create_session(workspace, snapshot, session_id=session_id)


def test_artifacts_are_create_only_hash_verified_and_isolated_by_pr(
    tmp_path: Path,
) -> None:
    identity = _repository_identity(tmp_path)
    store = PRWorkspaceStore(tmp_path / "ra")
    workspace_a = store.create_or_load_workspace(
        store.resolve_pr(identity, "local", "task-a"),
        _metadata(),
    )
    workspace_b = store.create_or_load_workspace(
        store.resolve_pr(identity, "local", "task-b"),
        _metadata(),
    )
    snapshot_a = store.create_or_load_snapshot(workspace_a, BASE_SHA, HEAD_SHA)
    snapshot_b = store.create_or_load_snapshot(workspace_b, BASE_SHA, HEAD_SHA)
    content = canonical_json_bytes({"result": "ok"})

    artifact = store.publish_create_only(
        snapshot_a,
        "ToolResults/artifacts/result.json",
        content,
    )

    assert store.resolve_snapshot_artifact(snapshot_a, artifact.artifact_id) == (
        snapshot_a.path / "ToolResults" / "artifacts" / "result.json"
    )
    assert store.read_verified_json(snapshot_a, artifact.artifact_id) == {
        "result": "ok"
    }

    with pytest.raises(PRWorkspaceSecurityError, match="not authorized"):
        store.resolve_snapshot_artifact(snapshot_b, artifact.artifact_id)

    with pytest.raises(PRWorkspaceError, match="already exists"):
        store.publish_create_only(
            snapshot_a,
            "ToolResults/artifacts/result.json",
            canonical_json_bytes({"result": "different"}),
        )

    artifact.path.write_bytes(b"tampered")
    with pytest.raises(PRWorkspaceSecurityError, match="hash"):
        store.read_verified_json(snapshot_a, artifact.artifact_id)


def test_snapshot_handle_from_another_store_is_not_authorized(tmp_path: Path) -> None:
    identity = _repository_identity(tmp_path)
    store_a = PRWorkspaceStore(tmp_path / "a")
    store_b = PRWorkspaceStore(tmp_path / "b")
    workspace = store_a.create_or_load_workspace(
        store_a.resolve_pr(identity, "local", "task-a"),
        _metadata(),
    )
    snapshot = store_a.create_or_load_snapshot(workspace, BASE_SHA, HEAD_SHA)

    with pytest.raises(PRWorkspaceSecurityError, match="store"):
        store_b.publish_create_only(
            snapshot,
            "ToolResults/artifacts/result.json",
            b"{}",
        )

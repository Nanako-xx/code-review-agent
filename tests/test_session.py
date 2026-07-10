from __future__ import annotations

from dataclasses import replace

import pytest

from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    SESSION_PHASES,
    SESSION_SCHEMA_VERSION,
    ArtifactDescriptor,
    PhaseCheckpoint,
    PhaseStatus,
    ReviewExecutionConfig,
    RevisionChangeKind,
    SessionManifest,
    initial_session_manifest,
    session_manifest_from_dict,
    session_manifest_to_dict,
)


NOW = "2026-07-10T00:00:00Z"


def execution_config() -> ReviewExecutionConfig:
    return ReviewExecutionConfig(
        reviewer_provider="openai-compatible",
        reviewer_model="review-model",
        reviewer_base_url="https://provider.example/v1",
        reviewer_api_key_env="REVIEW_AGENT_API_KEY",
        reviewer_mode="multi",
        reviewer_loop="agent-loop",
        non_interactive=True,
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
    }
    assert "api_key" not in execution
    assert "authorization" not in str(payload).casefold()


@pytest.mark.parametrize("secret_field", ["api_key", "Authorization"])
def test_session_manifest_rejects_secret_execution_fields(secret_field: str) -> None:
    payload = session_manifest_to_dict(manifest())
    payload["execution"][secret_field] = "secret-value"

    with pytest.raises(ValueError, match=secret_field.casefold()):
        session_manifest_from_dict(payload)


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
    "model_type",
    [
        ReviewExecutionConfig,
        ArtifactDescriptor,
        PhaseCheckpoint,
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

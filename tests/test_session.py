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

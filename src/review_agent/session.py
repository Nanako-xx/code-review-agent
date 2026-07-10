from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
import re
from types import MappingProxyType
from typing import Any, Mapping, TypeVar
from urllib.parse import urlsplit

from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.run_state import RunPhase, RunStatus


SESSION_SCHEMA_VERSION = 1
SESSION_PHASES = (
    RunPhase.PREFLIGHT,
    RunPhase.REPOSITORY_INTELLIGENCE,
    RunPhase.REVIEWERS,
    RunPhase.RECONCILIATION,
    RunPhase.COMPLETION,
    RunPhase.FINAL_RISK,
    RunPhase.REPORTING,
)

_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GIT_OBJECT_ID_PATTERN = re.compile(r"^(?:[0-9A-Fa-f]{40}|[0-9A-Fa-f]{64})$")
_SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class RevisionChangeKind(str, Enum):
    INITIAL = "initial"
    HEAD_MOVED = "head_moved"
    BASE_MOVED = "base_moved"
    BASE_AND_HEAD_MOVED = "base_and_head_moved"


@dataclass(frozen=True)
class ReviewExecutionConfig:
    reviewer_provider: str
    reviewer_model: str | None
    reviewer_base_url: str | None
    reviewer_api_key_env: str
    reviewer_mode: str
    reviewer_loop: str
    non_interactive: bool

    def __post_init__(self) -> None:
        if not _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(self.reviewer_api_key_env):
            raise ValueError(
                "reviewer_api_key_env must be an environment variable name, "
                "not an API key value"
            )
        if self.reviewer_base_url is not None:
            parsed = urlsplit(self.reviewer_base_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "reviewer_base_url must be an HTTP(S) base URL without "
                    "credentials, query parameters, or fragments"
                )
        if type(self.non_interactive) is not bool:
            raise ValueError("non_interactive must be a boolean")


@dataclass(frozen=True)
class ArtifactDescriptor:
    name: str
    path: str
    sha256: str
    schema: str
    phase: RunPhase
    revision_binding: str | None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, "name")
        _require_non_empty_string(self.schema, "schema")
        _validate_artifact_path(self.path)
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.sha256
        ):
            raise ValueError("sha256 must be a full 64-character hexadecimal digest")
        if not isinstance(self.phase, RunPhase) or self.phase not in SESSION_PHASES:
            raise ValueError("phase must be one of the persisted SESSION_PHASES")


@dataclass(frozen=True)
class PhaseCheckpoint:
    status: PhaseStatus = PhaseStatus.PENDING
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PhaseStatus):
            raise ValueError("status must be a PhaseStatus")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        artifacts = _immutable_string_tuple(self.artifacts, "artifacts")
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True)
class SessionManifest:
    schema_version: int
    review_id: str
    parent_review_id: str | None
    root_review_id: str
    repository: RepositoryIdentity
    revisions: ResolvedRevisions
    original_base_sha: str
    incremental_from_sha: str | None
    revision_change_kind: RevisionChangeKind
    execution: ReviewExecutionConfig
    status: RunStatus
    current_phase: RunPhase
    last_successful_phase: RunPhase | None
    phases: Mapping[str, PhaseCheckpoint]
    artifacts: Mapping[str, ArtifactDescriptor]
    errors: tuple[str, ...]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        if self.schema_version != SESSION_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must equal the supported Session schema version "
                f"{SESSION_SCHEMA_VERSION}"
            )
        _require_non_empty_string(self.review_id, "review_id")
        _require_non_empty_string(self.root_review_id, "root_review_id")
        if self.parent_review_id is not None:
            _require_non_empty_string(self.parent_review_id, "parent_review_id")
        if not isinstance(self.revision_change_kind, RevisionChangeKind):
            raise ValueError("revision_change_kind must be a RevisionChangeKind")
        if not isinstance(self.status, RunStatus):
            raise ValueError("status must be a RunStatus")
        if not isinstance(self.current_phase, RunPhase):
            raise ValueError("current_phase must be a RunPhase")
        if self.last_successful_phase is not None and not isinstance(
            self.last_successful_phase,
            RunPhase,
        ):
            raise ValueError("last_successful_phase must be a RunPhase or null")

        _validate_manifest_object_ids(self)
        _validate_manifest_lineage(self)

        if not isinstance(self.phases, Mapping):
            raise ValueError("phases must be a mapping")
        phases = dict(self.phases)
        expected_phase_names = {phase.value for phase in SESSION_PHASES}
        if set(phases) != expected_phase_names:
            raise ValueError("phases must contain exactly the persisted SESSION_PHASES")
        if any(
            not isinstance(checkpoint, PhaseCheckpoint)
            for checkpoint in phases.values()
        ):
            raise ValueError("phases values must be PhaseCheckpoint instances")

        if not isinstance(self.artifacts, Mapping):
            raise ValueError("artifacts must be a mapping")
        artifacts = dict(self.artifacts)
        for registry_name, descriptor in artifacts.items():
            if not isinstance(registry_name, str):
                raise ValueError("artifact registry keys must be strings")
            if not isinstance(descriptor, ArtifactDescriptor):
                raise ValueError(
                    "artifact registry values must be ArtifactDescriptor instances"
                )
            if registry_name != descriptor.name:
                raise ValueError(
                    f"artifact registry key {registry_name!r} must match "
                    f"descriptor.name {descriptor.name!r}"
                )

        errors = _immutable_string_tuple(self.errors, "errors")
        object.__setattr__(self, "phases", MappingProxyType(phases))
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))
        object.__setattr__(self, "errors", errors)


def _validate_manifest_object_ids(manifest: SessionManifest) -> None:
    object_ids = {
        "resolved_base_sha": manifest.revisions.resolved_base_sha,
        "resolved_head_sha": manifest.revisions.resolved_head_sha,
        "original_base_sha": manifest.original_base_sha,
    }
    if manifest.incremental_from_sha is not None:
        object_ids["incremental_from_sha"] = manifest.incremental_from_sha

    for field_name, object_id in object_ids.items():
        if not isinstance(object_id, str) or not _GIT_OBJECT_ID_PATTERN.fullmatch(
            object_id
        ):
            raise ValueError(
                f"{field_name} must be a full 40- or 64-character hexadecimal "
                "Git object ID"
            )
    if len({len(object_id) for object_id in object_ids.values()}) != 1:
        raise ValueError(
            "resolved_base_sha, resolved_head_sha, original_base_sha, and "
            "incremental_from_sha must use the same object ID format"
        )


def _validate_manifest_lineage(manifest: SessionManifest) -> None:
    if manifest.revision_change_kind is RevisionChangeKind.INITIAL:
        if manifest.parent_review_id is not None:
            raise ValueError("initial Session parent_review_id must be null")
        if manifest.root_review_id != manifest.review_id:
            raise ValueError("initial Session root_review_id must equal review_id")
        if (
            manifest.original_base_sha.casefold()
            != manifest.revisions.resolved_base_sha.casefold()
        ):
            raise ValueError(
                "initial Session original_base_sha must equal resolved_base_sha"
            )
        if manifest.incremental_from_sha is not None:
            raise ValueError("initial Session incremental_from_sha must be null")
        return

    if manifest.parent_review_id is None:
        raise ValueError("child Session parent_review_id must be present")
    if manifest.parent_review_id == manifest.review_id:
        raise ValueError("child Session parent_review_id must not self-reference")
    if manifest.root_review_id == manifest.review_id:
        raise ValueError("child Session root_review_id must not self-reference")

    if manifest.revision_change_kind is RevisionChangeKind.HEAD_MOVED:
        if manifest.incremental_from_sha is None:
            raise ValueError("HEAD_MOVED Session incremental_from_sha must be present")
        return

    if manifest.incremental_from_sha is not None:
        raise ValueError(
            "Base drift Session incremental_from_sha must be null because it "
            "requires a full re-review"
        )


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_artifact_path(path: Any) -> None:
    if not isinstance(path, str) or not path or path != path.strip():
        raise ValueError("path must be a non-empty canonical relative path")
    if "\\" in path:
        raise ValueError("path must use canonical forward-slash separators")

    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    path_parts = path.split("/")
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in path_parts)
        or posix_path.as_posix() != path
    ):
        raise ValueError(
            "path must be a canonical relative path without parent traversal"
        )


def _immutable_string_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a collection of strings")
    try:
        frozen_values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a collection of strings") from error
    if any(not isinstance(value, str) for value in frozen_values):
        raise ValueError(f"{field_name} must contain only strings")
    return frozen_values


def initial_session_manifest(
    *,
    review_id: str,
    repository: RepositoryIdentity,
    revisions: ResolvedRevisions,
    execution: ReviewExecutionConfig,
    now: str,
) -> SessionManifest:
    return SessionManifest(
        schema_version=SESSION_SCHEMA_VERSION,
        review_id=review_id,
        parent_review_id=None,
        root_review_id=review_id,
        repository=repository,
        revisions=revisions,
        original_base_sha=revisions.resolved_base_sha,
        incremental_from_sha=None,
        revision_change_kind=RevisionChangeKind.INITIAL,
        execution=execution,
        status=RunStatus.CREATED,
        current_phase=RunPhase.CREATED,
        last_successful_phase=None,
        phases={phase.value: PhaseCheckpoint() for phase in SESSION_PHASES},
        artifacts={},
        errors=(),
        created_at=now,
        updated_at=now,
    )


def session_manifest_to_dict(manifest: SessionManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "review_id": manifest.review_id,
        "parent_review_id": manifest.parent_review_id,
        "root_review_id": manifest.root_review_id,
        "repository": {
            "canonical_path": manifest.repository.canonical_path,
            "git_common_dir": manifest.repository.git_common_dir,
            "origin_url": manifest.repository.origin_url,
        },
        "revisions": {
            "requested_base": manifest.revisions.requested_base,
            "requested_head": manifest.revisions.requested_head,
            "resolved_base_sha": manifest.revisions.resolved_base_sha,
            "resolved_head_sha": manifest.revisions.resolved_head_sha,
            "original_base_sha": manifest.original_base_sha,
            "incremental_from_sha": manifest.incremental_from_sha,
            "change_kind": manifest.revision_change_kind.value,
        },
        "execution": {
            "reviewer_provider": manifest.execution.reviewer_provider,
            "reviewer_model": manifest.execution.reviewer_model,
            "reviewer_base_url": manifest.execution.reviewer_base_url,
            "reviewer_api_key_env": manifest.execution.reviewer_api_key_env,
            "reviewer_mode": manifest.execution.reviewer_mode,
            "reviewer_loop": manifest.execution.reviewer_loop,
            "non_interactive": manifest.execution.non_interactive,
        },
        "status": manifest.status.value,
        "current_phase": manifest.current_phase.value,
        "last_successful_phase": (
            manifest.last_successful_phase.value
            if manifest.last_successful_phase is not None
            else None
        ),
        "phases": {
            name: {
                "status": checkpoint.status.value,
                "attempts": checkpoint.attempts,
                "started_at": checkpoint.started_at,
                "completed_at": checkpoint.completed_at,
                "artifacts": list(checkpoint.artifacts),
                "error": checkpoint.error,
            }
            for name, checkpoint in manifest.phases.items()
        },
        "artifacts": {
            name: {
                "name": descriptor.name,
                "path": descriptor.path,
                "sha256": descriptor.sha256,
                "schema": descriptor.schema,
                "phase": descriptor.phase.value,
                "revision_binding": descriptor.revision_binding,
            }
            for name, descriptor in manifest.artifacts.items()
        },
        "errors": list(manifest.errors),
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
    }


def session_manifest_from_dict(payload: Mapping[str, Any]) -> SessionManifest:
    root = _object(payload, "session")
    _exact_fields(
        root,
        {
            "schema_version",
            "review_id",
            "parent_review_id",
            "root_review_id",
            "repository",
            "revisions",
            "execution",
            "status",
            "current_phase",
            "last_successful_phase",
            "phases",
            "artifacts",
            "errors",
            "created_at",
            "updated_at",
        },
        "session",
    )

    schema_version = _integer(root, "schema_version", "session")
    if schema_version != SESSION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported session schema_version: "
            f"{schema_version}; expected {SESSION_SCHEMA_VERSION}"
        )

    repository_payload = _object_field(root, "repository", "session")
    _exact_fields(
        repository_payload,
        {"canonical_path", "git_common_dir", "origin_url"},
        "session.repository",
    )
    repository = RepositoryIdentity(
        canonical_path=_string(repository_payload, "canonical_path", "session.repository"),
        git_common_dir=_string(repository_payload, "git_common_dir", "session.repository"),
        origin_url=_optional_string(repository_payload, "origin_url", "session.repository"),
    )

    revisions_payload = _object_field(root, "revisions", "session")
    _exact_fields(
        revisions_payload,
        {
            "requested_base",
            "requested_head",
            "resolved_base_sha",
            "resolved_head_sha",
            "original_base_sha",
            "incremental_from_sha",
            "change_kind",
        },
        "session.revisions",
    )
    revisions = ResolvedRevisions(
        requested_base=_string(revisions_payload, "requested_base", "session.revisions"),
        requested_head=_string(revisions_payload, "requested_head", "session.revisions"),
        resolved_base_sha=_string(
            revisions_payload,
            "resolved_base_sha",
            "session.revisions",
        ),
        resolved_head_sha=_string(
            revisions_payload,
            "resolved_head_sha",
            "session.revisions",
        ),
    )

    execution_payload = _object_field(root, "execution", "session")
    _exact_fields(
        execution_payload,
        {
            "reviewer_provider",
            "reviewer_model",
            "reviewer_base_url",
            "reviewer_api_key_env",
            "reviewer_mode",
            "reviewer_loop",
            "non_interactive",
        },
        "session.execution",
    )
    execution = ReviewExecutionConfig(
        reviewer_provider=_string(
            execution_payload,
            "reviewer_provider",
            "session.execution",
        ),
        reviewer_model=_optional_string(
            execution_payload,
            "reviewer_model",
            "session.execution",
        ),
        reviewer_base_url=_optional_string(
            execution_payload,
            "reviewer_base_url",
            "session.execution",
        ),
        reviewer_api_key_env=_string(
            execution_payload,
            "reviewer_api_key_env",
            "session.execution",
        ),
        reviewer_mode=_string(
            execution_payload,
            "reviewer_mode",
            "session.execution",
        ),
        reviewer_loop=_string(
            execution_payload,
            "reviewer_loop",
            "session.execution",
        ),
        non_interactive=_boolean(
            execution_payload,
            "non_interactive",
            "session.execution",
        ),
    )

    phases_payload = _object_field(root, "phases", "session")
    expected_phase_names = {phase.value for phase in SESSION_PHASES}
    _exact_fields(phases_payload, expected_phase_names, "session.phases")
    phases = {
        phase.value: _phase_checkpoint_from_dict(
            _object_field(phases_payload, phase.value, "session.phases"),
            f"session.phases.{phase.value}",
        )
        for phase in SESSION_PHASES
    }

    artifacts_payload = _object_field(root, "artifacts", "session")
    artifacts: dict[str, ArtifactDescriptor] = {}
    for artifact_name, artifact_payload in artifacts_payload.items():
        if not isinstance(artifact_name, str):
            raise ValueError("session.artifacts keys must be strings")
        descriptor = _artifact_descriptor_from_dict(
            _object(artifact_payload, f"session.artifacts.{artifact_name}"),
            f"session.artifacts.{artifact_name}",
        )
        if descriptor.name != artifact_name:
            raise ValueError(
                f"session.artifacts.{artifact_name}.name must match its registry key"
            )
        artifacts[artifact_name] = descriptor

    last_successful_phase_value = root["last_successful_phase"]
    last_successful_phase = (
        None
        if last_successful_phase_value is None
        else _enum_value(
            RunPhase,
            last_successful_phase_value,
            "session.last_successful_phase",
        )
    )

    return SessionManifest(
        schema_version=schema_version,
        review_id=_string(root, "review_id", "session"),
        parent_review_id=_optional_string(root, "parent_review_id", "session"),
        root_review_id=_string(root, "root_review_id", "session"),
        repository=repository,
        revisions=revisions,
        original_base_sha=_string(
            revisions_payload,
            "original_base_sha",
            "session.revisions",
        ),
        incremental_from_sha=_optional_string(
            revisions_payload,
            "incremental_from_sha",
            "session.revisions",
        ),
        revision_change_kind=_enum_field(
            RevisionChangeKind,
            revisions_payload,
            "change_kind",
            "session.revisions",
        ),
        execution=execution,
        status=_enum_field(RunStatus, root, "status", "session"),
        current_phase=_enum_field(RunPhase, root, "current_phase", "session"),
        last_successful_phase=last_successful_phase,
        phases=phases,
        artifacts=artifacts,
        errors=_string_list(root, "errors", "session"),
        created_at=_string(root, "created_at", "session"),
        updated_at=_string(root, "updated_at", "session"),
    )


def _phase_checkpoint_from_dict(
    payload: Mapping[str, Any],
    context: str,
) -> PhaseCheckpoint:
    _exact_fields(
        payload,
        {"status", "attempts", "started_at", "completed_at", "artifacts", "error"},
        context,
    )
    attempts = _integer(payload, "attempts", context)
    if attempts < 0:
        raise ValueError(f"{context}.attempts must be non-negative")
    return PhaseCheckpoint(
        status=_enum_field(PhaseStatus, payload, "status", context),
        attempts=attempts,
        started_at=_optional_string(payload, "started_at", context),
        completed_at=_optional_string(payload, "completed_at", context),
        artifacts=_string_list(payload, "artifacts", context),
        error=_optional_string(payload, "error", context),
    )


def _artifact_descriptor_from_dict(
    payload: Mapping[str, Any],
    context: str,
) -> ArtifactDescriptor:
    _exact_fields(
        payload,
        {"name", "path", "sha256", "schema", "phase", "revision_binding"},
        context,
    )
    return ArtifactDescriptor(
        name=_string(payload, "name", context),
        path=_string(payload, "path", context),
        sha256=_string(payload, "sha256", context),
        schema=_string(payload, "schema", context),
        phase=_enum_field(RunPhase, payload, "phase", context),
        revision_binding=_optional_string(payload, "revision_binding", context),
    )


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _object_field(
    payload: Mapping[str, Any],
    field_name: str,
    context: str,
) -> Mapping[str, Any]:
    return _object(payload[field_name], f"{context}.{field_name}")


def _exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    keys = set(payload)
    missing = expected - keys
    if missing:
        raise ValueError(
            f"{context} is missing required field(s): {', '.join(sorted(missing))}"
        )
    unexpected = keys - expected
    if unexpected:
        names = ", ".join(sorted(str(name).casefold() for name in unexpected))
        raise ValueError(f"{context} contains unsupported field(s): {names}")


def _string(payload: Mapping[str, Any], field_name: str, context: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str):
        raise ValueError(f"{context}.{field_name} must be a string")
    return value


def _optional_string(
    payload: Mapping[str, Any],
    field_name: str,
    context: str,
) -> str | None:
    value = payload[field_name]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context}.{field_name} must be a string or null")
    return value


def _integer(payload: Mapping[str, Any], field_name: str, context: str) -> int:
    value = payload[field_name]
    if type(value) is not int:
        raise ValueError(f"{context}.{field_name} must be an integer")
    return value


def _boolean(payload: Mapping[str, Any], field_name: str, context: str) -> bool:
    value = payload[field_name]
    if type(value) is not bool:
        raise ValueError(f"{context}.{field_name} must be a boolean")
    return value


def _string_list(
    payload: Mapping[str, Any],
    field_name: str,
    context: str,
) -> list[str]:
    value = payload[field_name]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context}.{field_name} must be a list of strings")
    return list(value)


EnumType = TypeVar("EnumType", bound=Enum)


def _enum_field(
    enum_type: type[EnumType],
    payload: Mapping[str, Any],
    field_name: str,
    context: str,
) -> EnumType:
    return _enum_value(enum_type, payload[field_name], f"{context}.{field_name}")


def _enum_value(
    enum_type: type[EnumType],
    value: Any,
    context: str,
) -> EnumType:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{context} has unsupported value: {value}") from error

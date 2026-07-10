from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import uuid
from typing import Iterable

from review_agent.checkpoint import _atomic_write_text
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    SESSION_PHASES,
    ArtifactDescriptor,
    PhaseCheckpoint,
    PhaseStatus,
    SessionManifest,
    session_manifest_from_dict,
    session_manifest_to_dict,
)


class SessionStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.session_path = self.run_dir / "session.json"

    def create(self, manifest: SessionManifest) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        content = self._serialize(manifest)
        staging = self.session_path.with_name(
            f".{self.session_path.name}.{uuid.uuid4().hex}.create"
        )
        try:
            _atomic_write_text(staging, content)
            try:
                os.link(staging, self.session_path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Session manifest already exists: {self.session_path}"
                ) from error
        finally:
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass
        return self.session_path

    def load(self) -> SessionManifest:
        payload = json.loads(self.session_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("session.json must contain a JSON object")
        return session_manifest_from_dict(payload)

    def write(self, manifest: SessionManifest) -> Path:
        current = self.load()
        if current.review_id != manifest.review_id:
            raise ValueError(
                "cannot replace Session with a different review_id: "
                f"{current.review_id!r} != {manifest.review_id!r}"
            )
        _atomic_write_text(self.session_path, self._serialize(manifest))
        return self.session_path

    def register_existing_artifact(
        self,
        *,
        name: str,
        relative_path: str,
        schema: str,
        phase: RunPhase,
        revision_binding: str | None,
        now: str,
    ) -> SessionManifest:
        phase = _require_session_phase(phase)
        current = self.load()
        _require_revision_binding(current, revision_binding)
        artifact_path = self._resolve_regular_artifact(relative_path)
        try:
            digest = sha256(artifact_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"unable to read artifact file: {relative_path}") from error
        descriptor = ArtifactDescriptor(
            name=name,
            path=relative_path,
            sha256=digest,
            schema=schema,
            phase=phase,
            revision_binding=revision_binding,
        )

        existing = current.artifacts.get(name)
        if existing is not None:
            if existing == descriptor:
                return current
            raise ValueError(f"artifact name is already registered: {name}")

        artifacts = dict(current.artifacts)
        artifacts[name] = descriptor
        updated = replace(current, artifacts=artifacts, updated_at=now)
        self.write(updated)
        return updated

    def validate_artifact(self, descriptor: ArtifactDescriptor) -> bool:
        try:
            current = self.load()
            _require_revision_binding(current, descriptor.revision_binding)
            artifact_path = self._resolve_regular_artifact(descriptor.path)
            actual_hash = sha256(artifact_path.read_bytes()).hexdigest()
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return hmac.compare_digest(actual_hash, descriptor.sha256)

    def mark_phase_completed(
        self,
        phase: RunPhase,
        artifact_names: Iterable[str],
        now: str,
    ) -> SessionManifest:
        phase = _require_session_phase(phase)
        current = self.load()
        phase_index = SESSION_PHASES.index(phase)
        for predecessor in SESSION_PHASES[:phase_index]:
            if (
                current.phases[predecessor.value].status
                is not PhaseStatus.COMPLETED
            ):
                raise ValueError(
                    f"cannot complete {phase.value} before {predecessor.value}"
                )

        checkpoint = current.phases[phase.value]
        requested_artifacts = _unique_artifact_names(artifact_names)
        checkpoint_artifacts = _unique_artifact_names(
            (*checkpoint.artifacts, *requested_artifacts)
        )
        for artifact_name in checkpoint_artifacts:
            self._require_valid_phase_artifact(current, phase, artifact_name)

        phases = dict(current.phases)
        phases[phase.value] = PhaseCheckpoint(
            status=PhaseStatus.COMPLETED,
            attempts=checkpoint.attempts + 1,
            started_at=checkpoint.started_at or now,
            completed_at=now,
            artifacts=checkpoint_artifacts,
            error=None,
        )
        updated = replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=phase,
            last_successful_phase=phase,
            phases=phases,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def mark_session_completed(self, now: str) -> SessionManifest:
        current = self.load()
        incomplete = [
            phase.value
            for phase in SESSION_PHASES
            if current.phases[phase.value].status is not PhaseStatus.COMPLETED
        ]
        if incomplete:
            raise ValueError(
                "cannot complete Session because phases are not completed: "
                + ", ".join(incomplete)
            )

        for phase in SESSION_PHASES:
            checkpoint = current.phases[phase.value]
            for artifact_name in checkpoint.artifacts:
                self._require_valid_phase_artifact(current, phase, artifact_name)

        updated = replace(
            current,
            status=RunStatus.COMPLETED,
            current_phase=RunPhase.COMPLETED,
            last_successful_phase=RunPhase.REPORTING,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def mark_session_failed(
        self,
        phase: RunPhase,
        error: str,
        now: str,
    ) -> SessionManifest:
        phase = _require_session_phase(phase)
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be a non-empty string")
        current = self.load()
        checkpoint = current.phases[phase.value]
        phases = dict(current.phases)
        phases[phase.value] = PhaseCheckpoint(
            status=PhaseStatus.FAILED,
            attempts=checkpoint.attempts + 1,
            started_at=checkpoint.started_at or now,
            completed_at=None,
            artifacts=checkpoint.artifacts,
            error=error,
        )
        updated = replace(
            current,
            status=RunStatus.FAILED,
            current_phase=RunPhase.FAILED,
            phases=phases,
            errors=(*current.errors, error),
            updated_at=now,
        )
        self.write(updated)
        return updated

    @staticmethod
    def _serialize(manifest: SessionManifest) -> str:
        return json.dumps(
            session_manifest_to_dict(manifest),
            indent=2,
            ensure_ascii=False,
        )

    def _resolve_regular_artifact(self, relative_path: str) -> Path:
        canonical_path = _canonical_relative_path(relative_path)
        try:
            root = self.run_dir.resolve(strict=True)
            candidate = self.run_dir.joinpath(*PurePosixPath(canonical_path).parts)
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                f"artifact does not exist or is not a regular file: {relative_path}"
            ) from error
        except OSError as error:
            raise ValueError(f"unable to resolve artifact path: {relative_path}") from error

        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"artifact path resolves outside the Session run directory: {relative_path}"
            ) from error

        try:
            mode = resolved.stat().st_mode
        except OSError as error:
            raise ValueError(f"unable to inspect artifact file: {relative_path}") from error
        if not stat.S_ISREG(mode):
            raise ValueError(f"artifact must be a regular file: {relative_path}")
        return resolved

    def _require_valid_phase_artifact(
        self,
        manifest: SessionManifest,
        phase: RunPhase,
        artifact_name: str,
    ) -> ArtifactDescriptor:
        descriptor = manifest.artifacts.get(artifact_name)
        if descriptor is None:
            raise ValueError(f"artifact is not registered: {artifact_name}")
        if descriptor.phase is not phase:
            raise ValueError(
                f"artifact {artifact_name!r} belongs to phase "
                f"{descriptor.phase.value}, not {phase.value}"
            )
        if not self.validate_artifact(descriptor):
            raise ValueError(f"artifact validation failed: {artifact_name}")
        return descriptor


def _require_session_phase(phase: RunPhase) -> RunPhase:
    if not isinstance(phase, RunPhase) or phase not in SESSION_PHASES:
        raise ValueError("phase must be one of the persisted SESSION_PHASES")
    return phase


def _canonical_relative_path(relative_path: str) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("artifact path must be a non-empty relative path")
    if relative_path != relative_path.strip() or "\\" in relative_path:
        raise ValueError("artifact path must be a canonical relative path")
    posix_path = PurePosixPath(relative_path)
    windows_path = PureWindowsPath(relative_path)
    parts = relative_path.split("/")
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in parts)
        or posix_path.as_posix() != relative_path
    ):
        raise ValueError("artifact path must be a canonical relative path inside run_dir")
    return relative_path


def _require_revision_binding(
    manifest: SessionManifest,
    revision_binding: str | None,
) -> None:
    if revision_binding:
        expected = (
            f"{manifest.revisions.resolved_base_sha}.."
            f"{manifest.revisions.resolved_head_sha}"
        )
        if revision_binding != expected:
            raise ValueError(
                "revision_binding must exactly match the Session resolved revisions"
            )


def _unique_artifact_names(artifact_names: Iterable[str]) -> tuple[str, ...]:
    if isinstance(artifact_names, (str, bytes)):
        raise ValueError("artifact_names must be an iterable of names")
    unique: list[str] = []
    seen: set[str] = set()
    for name in artifact_names:
        if not isinstance(name, str) or not name:
            raise ValueError("artifact names must be non-empty strings")
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return tuple(unique)

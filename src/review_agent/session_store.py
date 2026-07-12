from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import errno
import hmac
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import uuid
from typing import Iterable, Mapping

from review_agent.checkpoint import _atomic_write_text, _fsync_parent_directory
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    SESSION_PHASES,
    ArtifactDescriptor,
    PhaseCheckpoint,
    PhaseStatus,
    ReviewerTaskCheckpoint,
    SessionManifest,
    session_manifest_from_dict,
    session_manifest_to_dict,
)


@dataclass(frozen=True)
class PhaseValidation:
    phase: RunPhase
    valid: bool
    reason: str | None = None


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
            except OSError as error:
                self._fallback_create_without_overwrite(staging, error)
            _fsync_parent_directory(self.run_dir)
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
        self._require_immutable_metadata(current, manifest)
        if current.status is RunStatus.COMPLETED:
            if manifest == current:
                return self.session_path
            raise ValueError("completed Session is immutable and cannot be reopened")
        _atomic_write_text(self.session_path, self._serialize(manifest))
        return self.session_path

    @staticmethod
    def _require_immutable_metadata(
        current: SessionManifest,
        manifest: SessionManifest,
    ) -> None:
        immutable_fields = (
            "schema_version",
            "review_id",
            "repository",
            "revisions",
            "parent_review_id",
            "root_review_id",
            "original_base_sha",
            "revision_change_kind",
            "incremental_from_sha",
            "execution",
            "created_at",
        )
        for field_name in immutable_fields:
            if getattr(current, field_name) != getattr(manifest, field_name):
                raise ValueError(
                    f"cannot modify immutable Session field: {field_name}"
                )

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
        existing = current.artifacts.get(name)
        if current.status is RunStatus.COMPLETED and existing is None:
            raise ValueError("cannot register a new artifact on a completed Session")
        if (
            current.phases[phase.value].status is PhaseStatus.COMPLETED
            and existing is None
        ):
            raise ValueError(
                f"cannot register a new artifact on completed phase: {phase.value}"
            )
        _require_revision_binding(
            current,
            name=name,
            schema=schema,
            phase=phase,
            revision_binding=revision_binding,
        )
        digest = self._hash_regular_artifact(relative_path)
        descriptor = ArtifactDescriptor(
            name=name,
            path=relative_path,
            sha256=digest,
            schema=schema,
            phase=phase,
            revision_binding=revision_binding,
        )

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
            _require_revision_binding(
                current,
                name=descriptor.name,
                schema=descriptor.schema,
                phase=descriptor.phase,
                revision_binding=descriptor.revision_binding,
            )
            actual_hash = self._hash_regular_artifact(descriptor.path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return hmac.compare_digest(actual_hash, descriptor.sha256)

    def validate_phase(
        self,
        phase: RunPhase,
        expected_schemas: Mapping[str, str] | None = None,
    ) -> PhaseValidation:
        phase = _require_session_phase(phase)
        try:
            current = self.load()
            checkpoint = current.phases[phase.value]
            if checkpoint.status is not PhaseStatus.COMPLETED:
                raise ValueError(
                    f"phase status is {checkpoint.status.value}, not completed"
                )
            self._require_complete_phase_artifact_set(
                current,
                phase,
                checkpoint.artifacts,
            )
            if expected_schemas is not None:
                for artifact_name in checkpoint.artifacts:
                    expected = expected_schemas.get(artifact_name)
                    if expected is None:
                        raise ValueError(
                            f"no expected schema is defined for artifact: {artifact_name}"
                        )
                    actual = current.artifacts[artifact_name].schema
                    if actual != expected:
                        raise ValueError(
                            f"artifact {artifact_name!r} schema is {actual!r}, "
                            f"expected {expected!r}"
                        )
            if phase is RunPhase.REVIEWERS and checkpoint.tasks:
                for task_name, task in checkpoint.tasks.items():
                    if task.status is not PhaseStatus.COMPLETED:
                        raise ValueError(
                            f"reviewer task {task_name} is {task.status.value}, "
                            "not completed"
                        )
                    if not set(task.artifacts).issubset(checkpoint.artifacts):
                        raise ValueError(
                            f"reviewer task {task_name} references artifacts outside "
                            "the reviewers phase checkpoint"
                        )
                    for artifact_name in task.artifacts:
                        self._require_valid_phase_artifact(
                            current,
                            RunPhase.REVIEWERS,
                            artifact_name,
                        )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return PhaseValidation(phase=phase, valid=False, reason=str(error))
        return PhaseValidation(phase=phase, valid=True)

    def mark_phase_running(self, phase: RunPhase, now: str) -> SessionManifest:
        phase = _require_session_phase(phase)
        current = self.load()
        if current.status is RunStatus.COMPLETED:
            raise ValueError("cannot run a phase on a completed Session")
        self._require_predecessors_completed(current, phase)
        checkpoint = current.phases[phase.value]
        if checkpoint.status is PhaseStatus.COMPLETED:
            validation = self.validate_phase(phase)
            if validation.valid:
                return current
            raise ValueError(
                f"completed phase {phase.value} is invalid: {validation.reason}"
            )
        if checkpoint.status is PhaseStatus.RUNNING:
            return current

        phases = dict(current.phases)
        phases[phase.value] = replace(
            checkpoint,
            status=PhaseStatus.RUNNING,
            attempts=checkpoint.attempts + 1,
            started_at=now,
            completed_at=None,
            artifacts=(),
            error=None,
        )
        updated = replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=phase,
            phases=phases,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def restart_running_phase(self, phase: RunPhase, now: str) -> SessionManifest:
        phase = _require_session_phase(phase)
        current = self.load()
        checkpoint = current.phases[phase.value]
        if checkpoint.status is not PhaseStatus.RUNNING:
            raise ValueError(f"phase {phase.value} is not running")
        self._require_predecessors_completed(current, phase)
        phases = dict(current.phases)
        phases[phase.value] = replace(
            checkpoint,
            attempts=checkpoint.attempts + 1,
            started_at=now,
            completed_at=None,
            artifacts=(),
            error=None,
        )
        updated = replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=phase,
            phases=phases,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def discard_uncommitted_phase_artifacts(
        self,
        phase: RunPhase,
        preserve_names: Iterable[str],
        now: str,
    ) -> SessionManifest:
        phase = _require_session_phase(phase)
        preserve = set(_unique_artifact_names(preserve_names))
        current = self.load()
        checkpoint = current.phases[phase.value]
        if checkpoint.status is PhaseStatus.COMPLETED:
            raise ValueError("cannot discard artifacts from a completed phase")
        phase_registry = {
            name
            for name, descriptor in current.artifacts.items()
            if descriptor.phase is phase
        }
        if not preserve.issubset(phase_registry):
            raise ValueError("preserved artifacts must already belong to the phase")
        task_artifacts = {
            artifact_name
            for task in checkpoint.tasks.values()
            if task.status is PhaseStatus.COMPLETED
            for artifact_name in task.artifacts
        }
        if not task_artifacts.issubset(preserve):
            raise ValueError("completed reviewer task artifacts must be preserved")
        artifacts = {
            name: descriptor
            for name, descriptor in current.artifacts.items()
            if descriptor.phase is not phase or name in preserve
        }
        filtered_checkpoint_artifacts = tuple(
            name for name in checkpoint.artifacts if name in preserve
        )
        if (
            artifacts == dict(current.artifacts)
            and filtered_checkpoint_artifacts == checkpoint.artifacts
        ):
            return current
        phases = dict(current.phases)
        phases[phase.value] = replace(
            checkpoint,
            artifacts=filtered_checkpoint_artifacts,
        )
        updated = replace(
            current,
            phases=phases,
            artifacts=artifacts,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def initialize_reviewer_tasks(
        self,
        task_names: Iterable[str],
        now: str,
    ) -> SessionManifest:
        names = _unique_reviewer_task_names(task_names)
        current = self.load()
        checkpoint = current.phases[RunPhase.REVIEWERS.value]
        if checkpoint.status is not PhaseStatus.RUNNING:
            raise ValueError("reviewers phase must be running before tasks are initialized")
        if checkpoint.tasks:
            if tuple(checkpoint.tasks) == names:
                return current
            raise ValueError("reviewer task set is already initialized differently")
        phases = dict(current.phases)
        phases[RunPhase.REVIEWERS.value] = replace(
            checkpoint,
            tasks={name: ReviewerTaskCheckpoint() for name in names},
        )
        updated = replace(current, phases=phases, updated_at=now)
        self.write(updated)
        return updated

    def mark_reviewer_task_running(
        self,
        task_name: str,
        now: str,
    ) -> SessionManifest:
        current = self.load()
        checkpoint = current.phases[RunPhase.REVIEWERS.value]
        if checkpoint.status is not PhaseStatus.RUNNING:
            raise ValueError("reviewers phase must be running")
        task = _reviewer_task(checkpoint, task_name)
        if task.status is PhaseStatus.COMPLETED:
            self._require_reviewer_task_artifacts(current, task_name, task.artifacts)
            return current
        if task.status is PhaseStatus.RUNNING:
            return current
        tasks = dict(checkpoint.tasks)
        tasks[task_name] = replace(
            task,
            status=PhaseStatus.RUNNING,
            attempts=task.attempts + 1,
            started_at=now,
            completed_at=None,
            artifacts=(),
            error=None,
        )
        phases = dict(current.phases)
        phases[RunPhase.REVIEWERS.value] = replace(checkpoint, tasks=tasks)
        updated = replace(current, phases=phases, updated_at=now)
        self.write(updated)
        return updated

    def restart_running_reviewer_task(
        self,
        task_name: str,
        now: str,
    ) -> SessionManifest:
        current = self.load()
        checkpoint = current.phases[RunPhase.REVIEWERS.value]
        if checkpoint.status is not PhaseStatus.RUNNING:
            raise ValueError("reviewers phase must be running")
        task = _reviewer_task(checkpoint, task_name)
        if task.status is not PhaseStatus.RUNNING:
            raise ValueError(f"reviewer task {task_name} is not running")
        tasks = dict(checkpoint.tasks)
        tasks[task_name] = replace(
            task,
            attempts=task.attempts + 1,
            started_at=now,
            completed_at=None,
            artifacts=(),
            error=None,
        )
        phases = dict(current.phases)
        phases[RunPhase.REVIEWERS.value] = replace(checkpoint, tasks=tasks)
        updated = replace(current, phases=phases, updated_at=now)
        self.write(updated)
        return updated

    def mark_reviewer_task_completed(
        self,
        task_name: str,
        artifact_names: Iterable[str],
        now: str,
    ) -> SessionManifest:
        current = self.load()
        checkpoint = current.phases[RunPhase.REVIEWERS.value]
        if checkpoint.status is not PhaseStatus.RUNNING:
            raise ValueError("reviewers phase must be running")
        task = _reviewer_task(checkpoint, task_name)
        names = _unique_artifact_names(artifact_names)
        if task.status is PhaseStatus.COMPLETED:
            if set(names) != set(task.artifacts):
                raise ValueError(
                    f"reviewer task {task_name} is already completed with a "
                    "different artifact set"
                )
            self._require_reviewer_task_artifacts(current, task_name, names)
            return current
        if task.status is not PhaseStatus.RUNNING:
            raise ValueError(
                f"reviewer task {task_name} must be running before completion"
            )
        self._require_reviewer_task_artifacts(current, task_name, names)
        tasks = dict(checkpoint.tasks)
        tasks[task_name] = ReviewerTaskCheckpoint(
            status=PhaseStatus.COMPLETED,
            attempts=task.attempts,
            started_at=task.started_at,
            completed_at=now,
            artifacts=names,
            error=None,
        )
        phases = dict(current.phases)
        phases[RunPhase.REVIEWERS.value] = replace(checkpoint, tasks=tasks)
        updated = replace(current, phases=phases, updated_at=now)
        self.write(updated)
        return updated

    def mark_reviewer_task_failed(
        self,
        task_name: str,
        error: str,
        now: str,
    ) -> SessionManifest:
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be a non-empty string")
        current = self.load()
        checkpoint = current.phases[RunPhase.REVIEWERS.value]
        if checkpoint.status is not PhaseStatus.RUNNING:
            raise ValueError("reviewers phase must be running")
        task = _reviewer_task(checkpoint, task_name)
        if task.status is PhaseStatus.COMPLETED:
            raise ValueError(f"cannot fail completed reviewer task: {task_name}")
        if task.status is not PhaseStatus.RUNNING:
            raise ValueError(
                f"reviewer task {task_name} must be running before failure"
            )
        tasks = dict(checkpoint.tasks)
        tasks[task_name] = ReviewerTaskCheckpoint(
            status=PhaseStatus.FAILED,
            attempts=task.attempts,
            started_at=task.started_at,
            completed_at=None,
            artifacts=(),
            error=error,
        )
        phases = dict(current.phases)
        phases[RunPhase.REVIEWERS.value] = replace(
            checkpoint,
            status=PhaseStatus.FAILED,
            tasks=tasks,
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

    def invalidate_from(
        self,
        phase: RunPhase,
        reason: str,
        now: str,
    ) -> SessionManifest:
        phase = _require_session_phase(phase)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        current = self.load()
        phase_index = SESSION_PHASES.index(phase)
        invalidated_phases = set(SESSION_PHASES[phase_index:])
        checkpoint = current.phases[phase.value]
        if (
            current.status is RunStatus.RUNNING
            and current.current_phase is phase
            and checkpoint.status is PhaseStatus.INVALIDATED
            and checkpoint.error == reason
            and all(
                current.phases[candidate.value].status
                is PhaseStatus.INVALIDATED
                and not current.phases[candidate.value].artifacts
                for candidate in SESSION_PHASES[phase_index:]
            )
            and not any(
                descriptor.phase in invalidated_phases
                for descriptor in current.artifacts.values()
            )
        ):
            return current
        phases = dict(current.phases)
        for candidate in SESSION_PHASES[phase_index:]:
            checkpoint = phases[candidate.value]
            candidate_reason = (
                reason
                if candidate is phase
                else f"invalidated because {phase.value} is invalid"
            )
            phases[candidate.value] = PhaseCheckpoint(
                status=PhaseStatus.INVALIDATED,
                attempts=checkpoint.attempts,
                started_at=None,
                completed_at=None,
                artifacts=(),
                error=candidate_reason,
                tasks={},
            )
        artifacts = {
            name: descriptor
            for name, descriptor in current.artifacts.items()
            if descriptor.phase not in invalidated_phases
        }
        last_successful = next(
            (
                candidate
                for candidate in reversed(SESSION_PHASES[:phase_index])
                if phases[candidate.value].status is PhaseStatus.COMPLETED
            ),
            None,
        )
        error_message = f"invalidated {phase.value}: {reason}"
        updated = replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=phase,
            last_successful_phase=last_successful,
            phases=phases,
            artifacts=artifacts,
            errors=(*current.errors, error_message),
            updated_at=now,
        )
        self._require_immutable_metadata(current, updated)
        _atomic_write_text(self.session_path, self._serialize(updated))
        return updated

    def invalidate_reviewer_task(
        self,
        task_name: str,
        reason: str,
        now: str,
    ) -> SessionManifest:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        current = self.load()
        reviewer_checkpoint = current.phases[RunPhase.REVIEWERS.value]
        task = _reviewer_task(reviewer_checkpoint, task_name)
        if task.status is PhaseStatus.PENDING:
            raise ValueError("pending reviewer task has no committed state to invalidate")
        if (
            task.status is PhaseStatus.INVALIDATED
            and task.error == reason
            and current.current_phase is RunPhase.REVIEWERS
        ):
            return current

        invalidated_task = ReviewerTaskCheckpoint(
            status=PhaseStatus.INVALIDATED,
            attempts=max(1, task.attempts),
            started_at=task.started_at or now,
            completed_at=None,
            artifacts=(),
            error=reason,
        )
        tasks = dict(reviewer_checkpoint.tasks)
        tasks[task_name] = invalidated_task
        preserved_task_artifacts = {
            artifact_name
            for name, candidate in tasks.items()
            if name != task_name and candidate.status is PhaseStatus.COMPLETED
            for artifact_name in candidate.artifacts
        }
        phases = dict(current.phases)
        phases[RunPhase.REVIEWERS.value] = PhaseCheckpoint(
            status=PhaseStatus.INVALIDATED,
            attempts=reviewer_checkpoint.attempts,
            started_at=None,
            completed_at=None,
            artifacts=(),
            error=f"reviewer task {task_name} invalid: {reason}",
            tasks=tasks,
        )
        reviewer_index = SESSION_PHASES.index(RunPhase.REVIEWERS)
        downstream = set(SESSION_PHASES[reviewer_index + 1 :])
        for phase in SESSION_PHASES[reviewer_index + 1 :]:
            checkpoint = phases[phase.value]
            phases[phase.value] = PhaseCheckpoint(
                status=PhaseStatus.INVALIDATED,
                attempts=checkpoint.attempts,
                started_at=None,
                completed_at=None,
                artifacts=(),
                error=f"invalidated because reviewer task {task_name} is invalid",
            )
        artifacts = {
            name: descriptor
            for name, descriptor in current.artifacts.items()
            if (
                descriptor.phase is not RunPhase.REVIEWERS
                and descriptor.phase not in downstream
            )
            or name in preserved_task_artifacts
        }
        message = f"invalidated reviewer task {task_name}: {reason}"
        updated = replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=RunPhase.REVIEWERS,
            last_successful_phase=RunPhase.REPOSITORY_INTELLIGENCE,
            phases=phases,
            artifacts=artifacts,
            errors=(*current.errors, message),
            updated_at=now,
        )
        self._require_immutable_metadata(current, updated)
        _atomic_write_text(self.session_path, self._serialize(updated))
        return updated

    def invalidate_reviewers_preserving_tasks(
        self,
        reason: str,
        now: str,
    ) -> SessionManifest:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        current = self.load()
        checkpoint = current.phases[RunPhase.REVIEWERS.value]
        preserved = {
            artifact_name
            for task in checkpoint.tasks.values()
            if task.status is PhaseStatus.COMPLETED
            for artifact_name in task.artifacts
        }
        phases = dict(current.phases)
        phases[RunPhase.REVIEWERS.value] = PhaseCheckpoint(
            status=PhaseStatus.INVALIDATED,
            attempts=checkpoint.attempts,
            started_at=None,
            completed_at=None,
            artifacts=(),
            error=reason,
            tasks=checkpoint.tasks,
        )
        reviewer_index = SESSION_PHASES.index(RunPhase.REVIEWERS)
        downstream = set(SESSION_PHASES[reviewer_index + 1 :])
        for phase in SESSION_PHASES[reviewer_index + 1 :]:
            candidate = phases[phase.value]
            phases[phase.value] = PhaseCheckpoint(
                status=PhaseStatus.INVALIDATED,
                attempts=candidate.attempts,
                started_at=None,
                completed_at=None,
                artifacts=(),
                error="invalidated because reviewers aggregate is invalid",
            )
        artifacts = {
            name: descriptor
            for name, descriptor in current.artifacts.items()
            if (
                descriptor.phase is not RunPhase.REVIEWERS
                and descriptor.phase not in downstream
            )
            or name in preserved
        }
        message = f"invalidated reviewers: {reason}"
        updated = replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=RunPhase.REVIEWERS,
            last_successful_phase=RunPhase.REPOSITORY_INTELLIGENCE,
            phases=phases,
            artifacts=artifacts,
            errors=(*current.errors, message),
            updated_at=now,
        )
        self._require_immutable_metadata(current, updated)
        _atomic_write_text(self.session_path, self._serialize(updated))
        return updated

    def mark_phase_completed(
        self,
        phase: RunPhase,
        artifact_names: Iterable[str],
        now: str,
    ) -> SessionManifest:
        phase = _require_session_phase(phase)
        current = self.load()
        checkpoint = current.phases[phase.value]
        requested_artifacts = _unique_artifact_names(artifact_names)
        if checkpoint.status is PhaseStatus.COMPLETED:
            if set(requested_artifacts) == set(checkpoint.artifacts):
                self._require_complete_phase_artifact_set(
                    current,
                    phase,
                    checkpoint.artifacts,
                )
                return current
            raise ValueError(
                f"phase {phase.value} is already completed with a different "
                "artifact set"
            )
        if current.status is RunStatus.COMPLETED:
            raise ValueError("cannot reopen a completed Session")
        self._require_predecessors_completed(current, phase)

        if phase is RunPhase.REVIEWERS and checkpoint.tasks:
            self._require_reviewer_tasks_completed(checkpoint)
            task_artifacts = {
                artifact_name
                for task in checkpoint.tasks.values()
                for artifact_name in task.artifacts
            }
            if not task_artifacts.issubset(requested_artifacts):
                raise ValueError(
                    "reviewer phase artifact set omits completed reviewer task artifacts"
                )

        self._require_complete_phase_artifact_set(
            current,
            phase,
            requested_artifacts,
        )

        phases = dict(current.phases)
        phases[phase.value] = PhaseCheckpoint(
            status=PhaseStatus.COMPLETED,
            attempts=(
                checkpoint.attempts
                if checkpoint.status is PhaseStatus.RUNNING
                else checkpoint.attempts + 1
            ),
            started_at=checkpoint.started_at or now,
            completed_at=now,
            artifacts=requested_artifacts,
            error=None,
            tasks=checkpoint.tasks,
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
        self._validate_completion_candidate(current)
        if current.status is RunStatus.COMPLETED:
            return current

        updated = replace(
            current,
            status=RunStatus.COMPLETED,
            current_phase=RunPhase.COMPLETED,
            last_successful_phase=RunPhase.REPORTING,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def _validate_completion_candidate(self, current: SessionManifest) -> None:
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

        referenced_artifacts = {
            artifact_name
            for checkpoint in current.phases.values()
            for artifact_name in checkpoint.artifacts
        }
        registry_artifacts = set(current.artifacts)
        if registry_artifacts != referenced_artifacts:
            orphaned = sorted(registry_artifacts - referenced_artifacts)
            missing = sorted(referenced_artifacts - registry_artifacts)
            details: list[str] = []
            if orphaned:
                details.append("orphan registry artifacts: " + ", ".join(orphaned))
            if missing:
                details.append("unregistered checkpoint artifacts: " + ", ".join(missing))
            raise ValueError(
                "artifact registry must exactly match checkpoint artifacts; "
                + "; ".join(details)
            )

        for phase in SESSION_PHASES:
            checkpoint = current.phases[phase.value]
            if phase is RunPhase.REVIEWERS and checkpoint.tasks:
                self._require_reviewer_tasks_completed(checkpoint)
            for artifact_name in checkpoint.artifacts:
                self._require_valid_phase_artifact(current, phase, artifact_name)

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
        if current.status is RunStatus.COMPLETED:
            raise ValueError("cannot fail a completed Session")
        if checkpoint.status is PhaseStatus.COMPLETED:
            raise ValueError(f"cannot fail completed phase: {phase.value}")
        phase_index = SESSION_PHASES.index(phase)
        for predecessor in SESSION_PHASES[:phase_index]:
            predecessor_status = current.phases[predecessor.value].status
            if predecessor_status is not PhaseStatus.COMPLETED:
                raise ValueError(
                    f"cannot fail {phase.value} because predecessor "
                    f"{predecessor.value} is {predecessor_status.value}"
                )
        for successor in SESSION_PHASES[phase_index + 1 :]:
            if current.phases[successor.value].status is PhaseStatus.COMPLETED:
                raise ValueError(
                    f"cannot fail {phase.value} because completed successor "
                    f"{successor.value} exists"
                )
        phases = dict(current.phases)
        phases[phase.value] = PhaseCheckpoint(
            status=PhaseStatus.FAILED,
            attempts=(
                checkpoint.attempts
                if checkpoint.status is PhaseStatus.RUNNING
                else checkpoint.attempts + 1
            ),
            started_at=checkpoint.started_at or now,
            completed_at=None,
            artifacts=checkpoint.artifacts,
            error=error,
            tasks=checkpoint.tasks,
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
    def _require_predecessors_completed(
        manifest: SessionManifest,
        phase: RunPhase,
    ) -> None:
        phase_index = SESSION_PHASES.index(phase)
        for predecessor in SESSION_PHASES[:phase_index]:
            status = manifest.phases[predecessor.value].status
            if status is not PhaseStatus.COMPLETED:
                raise ValueError(
                    f"cannot run {phase.value} before {predecessor.value}; "
                    f"predecessor is {status.value}"
                )

    def _require_reviewer_task_artifacts(
        self,
        manifest: SessionManifest,
        task_name: str,
        artifact_names: Iterable[str],
    ) -> None:
        names = tuple(artifact_names)
        if not names:
            raise ValueError(f"reviewer task {task_name} must commit artifacts")
        for artifact_name in names:
            self._require_valid_phase_artifact(
                manifest,
                RunPhase.REVIEWERS,
                artifact_name,
            )

    @staticmethod
    def _require_reviewer_tasks_completed(checkpoint: PhaseCheckpoint) -> None:
        incomplete_tasks = [
            name
            for name, task in checkpoint.tasks.items()
            if task.status is not PhaseStatus.COMPLETED
        ]
        if incomplete_tasks:
            raise ValueError(
                "cannot complete reviewers before reviewer tasks complete: "
                + ", ".join(incomplete_tasks)
            )

    @staticmethod
    def _serialize(manifest: SessionManifest) -> str:
        return json.dumps(
            session_manifest_to_dict(manifest),
            indent=2,
            ensure_ascii=False,
        )

    def _hash_regular_artifact(self, relative_path: str) -> str:
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

        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            file_descriptor = os.open(resolved, flags)
        except OSError as error:
            raise ValueError(
                f"artifact does not exist or is not a regular file: {relative_path}"
            ) from error
        try:
            try:
                file_status = os.fstat(file_descriptor)
            except OSError as error:
                raise ValueError(
                    f"unable to inspect artifact file: {relative_path}"
                ) from error
            if not stat.S_ISREG(file_status.st_mode):
                raise ValueError(f"artifact must be a regular file: {relative_path}")

            digest = sha256()
            while True:
                try:
                    chunk = os.read(file_descriptor, 1024 * 1024)
                except OSError as error:
                    raise ValueError(
                        f"unable to read artifact file: {relative_path}"
                    ) from error
                if not chunk:
                    break
                digest.update(chunk)
            return digest.hexdigest()
        finally:
            os.close(file_descriptor)

    def _fallback_create_without_overwrite(
        self,
        staging: Path,
        link_error: OSError,
    ) -> None:
        unsupported_link_errors = {
            errno.EACCES,
            errno.ENOSYS,
            errno.EPERM,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if link_error.errno not in unsupported_link_errors or os.name != "nt":
            raise link_error
        try:
            os.rename(staging, self.session_path)
        except OSError as error:
            if self.session_path.exists():
                raise FileExistsError(
                    f"Session manifest already exists: {self.session_path}"
                ) from error
            raise

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

    def _require_complete_phase_artifact_set(
        self,
        manifest: SessionManifest,
        phase: RunPhase,
        artifact_names: Iterable[str],
    ) -> None:
        names = tuple(artifact_names)
        for artifact_name in names:
            self._require_valid_phase_artifact(manifest, phase, artifact_name)

        registry_names = {
            name
            for name, descriptor in manifest.artifacts.items()
            if descriptor.phase is phase
        }
        checkpoint_names = set(names)
        if registry_names != checkpoint_names:
            omitted = sorted(registry_names - checkpoint_names)
            unexpected = sorted(checkpoint_names - registry_names)
            details: list[str] = []
            if omitted:
                details.append("omitted registry artifacts: " + ", ".join(omitted))
            if unexpected:
                details.append("non-registry artifacts: " + ", ".join(unexpected))
            raise ValueError(
                f"phase {phase.value} artifact set must exactly match its registry; "
                + "; ".join(details)
            )


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
    *,
    name: str,
    schema: str,
    phase: RunPhase,
    revision_binding: str | None,
) -> None:
    if revision_binding == "":
        raise ValueError("revision_binding must not be empty")
    unbound_request = (
        name == "request"
        and schema == "review_request_v1"
        and phase is RunPhase.PREFLIGHT
    )
    if revision_binding is None:
        if unbound_request:
            return
        raise ValueError(
            "revision_binding may be null only for the preflight "
            "review_request_v1 request artifact"
        )
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


def _unique_reviewer_task_names(task_names: Iterable[str]) -> tuple[str, ...]:
    if isinstance(task_names, (str, bytes)):
        raise ValueError("task_names must be an iterable of reviewer task names")
    requested = tuple(task_names)
    names = _unique_artifact_names(requested)
    if len(names) != len(requested):
        raise ValueError("reviewer task_names must not contain duplicates")
    if not names:
        raise ValueError("reviewer task_names must not be empty")
    expected = tuple(f"reviewer-{index}" for index in range(len(names)))
    if names != expected:
        raise ValueError(
            "reviewer task_names must be contiguous and ordered from reviewer-0"
        )
    return names


def _reviewer_task(
    checkpoint: PhaseCheckpoint,
    task_name: str,
) -> ReviewerTaskCheckpoint:
    if not isinstance(task_name, str):
        raise ValueError("task_name must be a string")
    task = checkpoint.tasks.get(task_name)
    if task is None:
        raise ValueError(f"reviewer task is not initialized: {task_name}")
    return task

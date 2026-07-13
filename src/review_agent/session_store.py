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
    RESUMABLE_SESSION_SCHEMA_VERSIONS,
    SESSION_PHASES,
    SESSION_SCHEMA_VERSION,
    ArtifactDescriptor,
    PhaseCheckpoint,
    PhaseStatus,
    ReviewWaveCheckpoint,
    ReviewerTaskCheckpoint,
    SessionManifest,
    SupplementalBudget,
    SupplementalPolicy,
    SupplementalTaskCheckpoint,
    SupplementalTaskStatus,
    session_phases_for_schema,
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
        if manifest.schema_version != SESSION_SCHEMA_VERSION:
            raise ValueError(
                "new Sessions must use the current Session schema version"
            )
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
        self._require_current_layout(current)
        self._require_immutable_metadata(current, manifest)
        if current.status is RunStatus.COMPLETED:
            if manifest == current:
                return self.session_path
            raise ValueError("completed Session is immutable and cannot be reopened")
        _atomic_write_text(self.session_path, self._serialize(manifest))
        return self.session_path

    @staticmethod
    def _require_current_layout(manifest: SessionManifest) -> None:
        if manifest.schema_version not in RESUMABLE_SESSION_SCHEMA_VERSIONS:
            raise ValueError(
                "schema v1 Session is available for read-only audit; start a new "
                "schema v4 Session to use state transitions"
            )

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
        self._require_current_layout(current)
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
        self._require_current_layout(current)
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
        if checkpoint.status is PhaseStatus.AWAITING_USER:
            raise ValueError(
                "use resume_awaiting_user to resume an awaiting_user phase"
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
            artifacts=(
                checkpoint.artifacts if checkpoint.user_decisions else ()
            ),
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

    def mark_phase_awaiting_user(
        self,
        phase: RunPhase,
        artifact_names: Iterable[str],
        now: str,
    ) -> SessionManifest:
        phase = _require_session_phase(phase)
        if phase is not RunPhase.INTENT_RESOLUTION:
            raise ValueError(
                "awaiting_user is allowed only for intent_resolution"
            )
        current = self.load()
        self._require_current_layout(current)
        if current.status is RunStatus.COMPLETED:
            raise ValueError("cannot await user input on a completed Session")
        self._require_predecessors_completed(current, phase)
        checkpoint = current.phases[phase.value]
        requested_artifacts = _unique_artifact_names(artifact_names)
        if not requested_artifacts:
            raise ValueError(
                "awaiting_user intent_resolution must retain committed question "
                "or candidate artifacts"
            )
        if checkpoint.status is PhaseStatus.AWAITING_USER:
            if (
                current.status is RunStatus.AWAITING_USER
                and current.current_phase is RunPhase.INTENT_RESOLUTION
                and set(requested_artifacts) == set(checkpoint.artifacts)
            ):
                self._require_complete_phase_artifact_set(
                    current,
                    phase,
                    checkpoint.artifacts,
                )
                return current
            raise ValueError(
                "intent_resolution is already awaiting_user with a different "
                "artifact set"
            )
        if checkpoint.status is not PhaseStatus.RUNNING:
            raise ValueError(
                "intent_resolution must be running before awaiting_user"
            )
        self._require_no_completed_successors(current, phase)
        self._require_complete_phase_artifact_set(
            current,
            phase,
            requested_artifacts,
        )

        phases = dict(current.phases)
        phases[phase.value] = replace(
            checkpoint,
            status=PhaseStatus.AWAITING_USER,
            completed_at=None,
            artifacts=requested_artifacts,
            error=None,
        )
        updated = replace(
            current,
            status=RunStatus.AWAITING_USER,
            current_phase=RunPhase.INTENT_RESOLUTION,
            phases=phases,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def submit_user_decision(
        self,
        event_id: str,
        artifact_name: str,
        now: str,
    ) -> SessionManifest:
        _require_non_empty_string(event_id, "event_id")
        _require_non_empty_string(artifact_name, "artifact_name")
        current = self.load()
        self._require_current_layout(current)
        checkpoint = current.phases[RunPhase.INTENT_RESOLUTION.value]
        existing_artifact = checkpoint.user_decisions.get(event_id)
        if existing_artifact is not None:
            if existing_artifact != artifact_name:
                raise ValueError(
                    f"user decision {event_id!r} was already submitted with a "
                    "different artifact"
                )
            self._require_valid_phase_artifact(
                current,
                RunPhase.INTENT_RESOLUTION,
                artifact_name,
            )
            return current
        if (
            current.status is not RunStatus.AWAITING_USER
            or current.current_phase is not RunPhase.INTENT_RESOLUTION
            or checkpoint.status is not PhaseStatus.AWAITING_USER
        ):
            raise ValueError(
                "new user decisions may be submitted only while "
                "intent_resolution is awaiting_user"
            )
        self._require_valid_phase_artifact(
            current,
            RunPhase.INTENT_RESOLUTION,
            artifact_name,
        )

        user_decisions = dict(checkpoint.user_decisions)
        user_decisions[event_id] = artifact_name
        checkpoint_artifacts = tuple(
            dict.fromkeys((*checkpoint.artifacts, artifact_name))
        )
        phases = dict(current.phases)
        phases[RunPhase.INTENT_RESOLUTION.value] = replace(
            checkpoint,
            artifacts=checkpoint_artifacts,
            user_decisions=user_decisions,
        )
        updated = replace(current, phases=phases, updated_at=now)
        self.write(updated)
        return updated

    def resume_awaiting_user(self, now: str) -> SessionManifest:
        current = self.load()
        self._require_current_layout(current)
        checkpoint = current.phases[RunPhase.INTENT_RESOLUTION.value]
        if (
            current.status is RunStatus.RUNNING
            and current.current_phase is RunPhase.INTENT_RESOLUTION
            and checkpoint.status is PhaseStatus.RUNNING
            and checkpoint.user_decisions
        ):
            return current
        if (
            current.status is not RunStatus.AWAITING_USER
            or current.current_phase is not RunPhase.INTENT_RESOLUTION
            or checkpoint.status is not PhaseStatus.AWAITING_USER
        ):
            raise ValueError(
                "Session is not awaiting_user during intent_resolution"
            )
        if not checkpoint.user_decisions:
            raise ValueError(
                "cannot resume awaiting_user without a submitted user decision"
            )
        self._require_complete_phase_artifact_set(
            current,
            RunPhase.INTENT_RESOLUTION,
            checkpoint.artifacts,
        )

        phases = dict(current.phases)
        phases[RunPhase.INTENT_RESOLUTION.value] = replace(
            checkpoint,
            status=PhaseStatus.RUNNING,
            completed_at=None,
            error=None,
        )
        updated = replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=RunPhase.INTENT_RESOLUTION,
            phases=phases,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def restart_running_phase(self, phase: RunPhase, now: str) -> SessionManifest:
        phase = _require_session_phase(phase)
        current = self.load()
        self._require_current_layout(current)
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
            artifacts=(
                checkpoint.artifacts if checkpoint.user_decisions else ()
            ),
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
        self._require_current_layout(current)
        checkpoint = current.phases[phase.value]
        if checkpoint.status is PhaseStatus.COMPLETED:
            raise ValueError("cannot discard artifacts from a completed phase")
        if checkpoint.status is PhaseStatus.AWAITING_USER:
            raise ValueError(
                "cannot discard committed artifacts while awaiting_user"
            )
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
        decision_artifacts = set(checkpoint.user_decisions.values())
        if not decision_artifacts.issubset(preserve):
            raise ValueError("submitted user decision artifacts must be preserved")
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
        self._require_current_layout(current)
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
        self._require_current_layout(current)
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
        self._require_current_layout(current)
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
        self._require_current_layout(current)
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
        self._require_current_layout(current)
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

    def initialize_wave(
        self,
        wave_id: str,
        task_assignments: Mapping[str, str],
        now: str,
        *,
        trigger_digest: str,
        effective_policy: SupplementalPolicy | None = None,
    ) -> SessionManifest:
        if not isinstance(task_assignments, Mapping):
            raise ValueError("task_assignments must be a mapping")
        if effective_policy is not None and not isinstance(
            effective_policy,
            SupplementalPolicy,
        ):
            raise ValueError("effective_policy must be a SupplementalPolicy")
        assignments = dict(task_assignments)
        if not assignments:
            raise ValueError("a supplemental wave must contain at least one task")

        current = self.load()
        self._require_supplemental_phase_running(current)
        effective_policy = (
            effective_policy or current.execution.supplemental_policy
        )
        existing = current.supplemental_waves.get(wave_id)
        if existing is not None:
            expected_assignments = {
                task_id: task.assignment_digest
                for task_id, task in existing.tasks.items()
            }
            if (
                existing.trigger_digest != trigger_digest
                or existing.effective_policy != effective_policy
                or expected_assignments != assignments
            ):
                raise ValueError(
                    f"supplemental wave {wave_id} is already initialized differently"
                )
            if existing.status in {PhaseStatus.RUNNING, PhaseStatus.COMPLETED}:
                return current
            if existing.status not in {PhaseStatus.FAILED, PhaseStatus.INVALIDATED}:
                raise ValueError(
                    f"supplemental wave {wave_id} cannot be resumed from "
                    f"{existing.status.value}"
                )
            if any(
                task.status in {
                    SupplementalTaskStatus.RESERVED,
                    SupplementalTaskStatus.RUNNING,
                }
                for task in existing.tasks.values()
            ):
                raise ValueError(
                    "cannot resume a supplemental wave with active task reservations"
                )
            waves = dict(current.supplemental_waves)
            waves[wave_id] = replace(
                existing,
                status=PhaseStatus.RUNNING,
                attempts=existing.attempts + 1,
                started_at=now,
                completed_at=None,
                artifacts=(),
                stop_reason=None,
                error=None,
            )
            updated = replace(
                current,
                supplemental_waves=waves,
                updated_at=now,
            )
            self.write(updated)
            return updated

        policy = effective_policy
        waves = dict(current.supplemental_waves)
        if any(
            candidate.effective_policy != policy
            for candidate in waves.values()
        ):
            raise ValueError(
                "supplemental waves must share one effective Runtime policy"
            )
        wave_index = len(waves) + 1
        if wave_index > policy.max_waves:
            raise ValueError("supplemental wave exceeds policy max_waves")
        if len(assignments) > policy.max_tasks_per_wave:
            raise ValueError(
                "supplemental wave exceeds policy max_tasks_per_wave"
            )
        if sum(len(wave.tasks) for wave in waves.values()) + len(assignments) > policy.max_tasks:
            raise ValueError("supplemental tasks exceed policy max_tasks")
        incomplete_waves = [
            candidate.wave_id
            for candidate in waves.values()
            if candidate.status is not PhaseStatus.COMPLETED
        ]
        if incomplete_waves:
            raise ValueError(
                "cannot initialize a new supplemental wave before prior waves "
                "complete: " + ", ".join(incomplete_waves)
            )

        tasks = {
            task_id: SupplementalTaskCheckpoint(
                task_id=task_id,
                assignment_digest=assignment_digest,
            )
            for task_id, assignment_digest in assignments.items()
        }
        wave = ReviewWaveCheckpoint(
            wave_id=wave_id,
            wave_index=wave_index,
            trigger_digest=trigger_digest,
            effective_policy=policy,
            status=PhaseStatus.RUNNING,
            attempts=1,
            started_at=now,
            tasks=tasks,
        )
        waves[wave_id] = wave
        updated = replace(
            current,
            supplemental_waves=waves,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def reserve_task_budget(
        self,
        task_id: str,
        reservation: SupplementalBudget,
        now: str,
    ) -> SessionManifest:
        if not isinstance(reservation, SupplementalBudget):
            raise ValueError("reservation must be a SupplementalBudget")
        current = self.load()
        self._require_supplemental_phase_running(current)
        wave_id, wave, task = _supplemental_task(current, task_id)
        if wave.status is not PhaseStatus.RUNNING:
            raise ValueError(f"supplemental wave {wave_id} must be running")
        if task.status in {
            SupplementalTaskStatus.RESERVED,
            SupplementalTaskStatus.RUNNING,
        }:
            if task.reservation == reservation:
                return current
            raise ValueError(
                f"supplemental task {task_id} already has a different reservation"
            )
        if task.status in {
            SupplementalTaskStatus.COMPLETED,
            SupplementalTaskStatus.PARTIAL,
        }:
            raise ValueError(f"cannot reserve completed supplemental task: {task_id}")
        if task.status not in {
            SupplementalTaskStatus.PENDING,
            SupplementalTaskStatus.FAILED,
            SupplementalTaskStatus.INVALIDATED,
        }:
            raise ValueError(
                f"supplemental task {task_id} cannot be reserved from "
                f"{task.status.value}"
            )

        policy = wave.effective_policy
        _require_task_reservation_within_policy(reservation, policy)
        active = sum(
            candidate.status
            in {SupplementalTaskStatus.RESERVED, SupplementalTaskStatus.RUNNING}
            for candidate_wave in current.supplemental_waves.values()
            for candidate in candidate_wave.tasks.values()
        )
        if active >= policy.max_concurrency:
            raise ValueError("supplemental reservation exceeds policy max_concurrency")
        consumed = _supplemental_consumption(current)
        ceiling = _supplemental_budget_ceiling(current)
        if not (consumed + reservation).fits_within(ceiling):
            raise ValueError("supplemental reservation exceeds remaining global budget")

        updated_task = replace(
            task,
            status=SupplementalTaskStatus.RESERVED,
            started_at=None,
            completed_at=None,
            artifacts=(),
            reservation=reservation,
            error=None,
        )
        updated = _replace_supplemental_task(
            current,
            wave_id,
            task_id,
            updated_task,
            now,
        )
        self.write(updated)
        return updated

    def mark_task_running(self, task_id: str, now: str) -> SessionManifest:
        current = self.load()
        self._require_supplemental_phase_running(current)
        wave_id, wave, task = _supplemental_task(current, task_id)
        if wave.status is not PhaseStatus.RUNNING:
            raise ValueError(f"supplemental wave {wave_id} must be running")
        if task.status is SupplementalTaskStatus.RUNNING:
            return current
        if task.status in {
            SupplementalTaskStatus.COMPLETED,
            SupplementalTaskStatus.PARTIAL,
        }:
            self._require_supplemental_task_artifacts(
                current,
                task_id,
                task.artifacts,
            )
            return current
        if task.status is not SupplementalTaskStatus.RESERVED:
            raise ValueError(
                f"supplemental task {task_id} must be reserved before running"
            )
        updated_task = replace(
            task,
            status=SupplementalTaskStatus.RUNNING,
            attempts=task.attempts + 1,
            started_at=now,
            completed_at=None,
            artifacts=(),
            error=None,
        )
        updated = _replace_supplemental_task(
            current,
            wave_id,
            task_id,
            updated_task,
            now,
        )
        self.write(updated)
        return updated

    def mark_task_completed(
        self,
        task_id: str,
        artifact_names: Iterable[str],
        charged: SupplementalBudget,
        now: str,
    ) -> SessionManifest:
        return self._mark_task_with_artifacts(
            task_id,
            artifact_names,
            charged,
            now,
            status=SupplementalTaskStatus.COMPLETED,
            error=None,
        )

    def mark_task_partial(
        self,
        task_id: str,
        artifact_names: Iterable[str],
        error: str,
        charged: SupplementalBudget,
        now: str,
    ) -> SessionManifest:
        _require_non_empty_string(error, "error")
        return self._mark_task_with_artifacts(
            task_id,
            artifact_names,
            charged,
            now,
            status=SupplementalTaskStatus.PARTIAL,
            error=error,
        )

    def _mark_task_with_artifacts(
        self,
        task_id: str,
        artifact_names: Iterable[str],
        charged: SupplementalBudget,
        now: str,
        *,
        status: SupplementalTaskStatus,
        error: str | None,
    ) -> SessionManifest:
        if not isinstance(charged, SupplementalBudget):
            raise ValueError("charged must be a SupplementalBudget")
        names = _unique_artifact_names(artifact_names)
        current = self.load()
        self._require_supplemental_phase_running(current)
        wave_id, wave, task = _supplemental_task(current, task_id)
        if wave.status is not PhaseStatus.RUNNING:
            raise ValueError(f"supplemental wave {wave_id} must be running")
        if task.status is status:
            if (
                set(task.artifacts) == set(names)
                and charged.fits_within(task.charged)
                and task.error == error
            ):
                self._require_supplemental_task_artifacts(current, task_id, names)
                return current
            raise ValueError(
                f"supplemental task {task_id} is already {status.value} "
                "with different committed state"
            )
        if task.status is not SupplementalTaskStatus.RUNNING:
            raise ValueError(
                f"supplemental task {task_id} must be running before "
                f"becoming {status.value}"
            )
        _require_task_charge(charged, task.reservation)
        self._require_supplemental_task_artifacts(current, task_id, names)
        updated_task = replace(
            task,
            status=status,
            completed_at=now,
            artifacts=names,
            reservation=SupplementalBudget(),
            charged=task.charged + charged,
            error=error,
        )
        updated = _replace_supplemental_task(
            current,
            wave_id,
            task_id,
            updated_task,
            now,
        )
        self.write(updated)
        return updated

    def mark_task_failed(
        self,
        task_id: str,
        error: str,
        charged: SupplementalBudget,
        now: str,
        *,
        artifact_names: Iterable[str] = (),
    ) -> SessionManifest:
        _require_non_empty_string(error, "error")
        if not isinstance(charged, SupplementalBudget):
            raise ValueError("charged must be a SupplementalBudget")
        names = _unique_artifact_names(artifact_names)
        current = self.load()
        self._require_supplemental_phase_running(current)
        wave_id, wave, task = _supplemental_task(current, task_id)
        if wave.status is not PhaseStatus.RUNNING:
            raise ValueError(f"supplemental wave {wave_id} must be running")
        if task.status is SupplementalTaskStatus.FAILED:
            if (
                task.error == error
                and charged.fits_within(task.charged)
                and set(task.artifacts) == set(names)
            ):
                if names:
                    self._require_supplemental_task_artifacts(
                        current,
                        task_id,
                        names,
                    )
                return current
            raise ValueError(
                f"supplemental task {task_id} is already failed differently"
            )
        if task.status is not SupplementalTaskStatus.RUNNING:
            raise ValueError(
                f"supplemental task {task_id} must be running before failure"
            )
        _require_task_charge(charged, task.reservation)
        if names:
            self._require_supplemental_task_artifacts(current, task_id, names)
        updated_task = replace(
            task,
            status=SupplementalTaskStatus.FAILED,
            completed_at=None,
            artifacts=names,
            reservation=SupplementalBudget(),
            charged=task.charged + charged,
            error=error,
        )
        updated = _replace_supplemental_task(
            current,
            wave_id,
            task_id,
            updated_task,
            now,
        )
        self.write(updated)
        return updated

    def mark_task_unrunnable(
        self,
        task_id: str,
        error: str,
        now: str,
    ) -> SessionManifest:
        """Close a task that cannot reserve another bounded attempt."""

        _require_non_empty_string(error, "error")
        current = self.load()
        self._require_supplemental_phase_running(current)
        wave_id, wave, task = _supplemental_task(current, task_id)
        if wave.status is not PhaseStatus.RUNNING:
            raise ValueError(f"supplemental wave {wave_id} must be running")
        if task.status is SupplementalTaskStatus.FAILED and task.error == error:
            return current
        if task.status not in {
            SupplementalTaskStatus.PENDING,
            SupplementalTaskStatus.FAILED,
            SupplementalTaskStatus.INVALIDATED,
        }:
            raise ValueError(
                f"supplemental task {task_id} cannot become unrunnable from "
                f"{task.status.value}"
            )
        updated_task = replace(
            task,
            status=SupplementalTaskStatus.FAILED,
            attempts=max(1, task.attempts),
            started_at=task.started_at or now,
            completed_at=None,
            artifacts=(),
            reservation=SupplementalBudget(),
            error=error,
        )
        updated = _replace_supplemental_task(
            current,
            wave_id,
            task_id,
            updated_task,
            now,
        )
        self.write(updated)
        return updated

    def mark_task_unknown(
        self,
        task_id: str,
        invocation_id: str,
        error: str,
        now: str,
    ) -> SessionManifest:
        _require_non_empty_string(error, "error")
        current = self.load()
        self._require_supplemental_phase_running(current)
        wave_id, wave, task = _supplemental_task(current, task_id)
        if wave.status is not PhaseStatus.RUNNING:
            raise ValueError(f"supplemental wave {wave_id} must be running")
        if invocation_id in task.unknown_invocation_ids:
            if task.status is SupplementalTaskStatus.FAILED and task.error == error:
                return current
            raise ValueError(
                f"supplemental invocation {invocation_id} is already recorded"
            )
        if any(
            invocation_id in candidate.unknown_invocation_ids
            for candidate_wave in current.supplemental_waves.values()
            for candidate in candidate_wave.tasks.values()
        ):
            raise ValueError(
                f"supplemental invocation {invocation_id} belongs to another task"
            )
        if task.status is not SupplementalTaskStatus.RUNNING:
            raise ValueError(
                f"supplemental task {task_id} must be running before unknown usage"
            )
        updated_task = replace(
            task,
            status=SupplementalTaskStatus.FAILED,
            completed_at=None,
            artifacts=(),
            reservation=SupplementalBudget(),
            unknown_consumed=task.unknown_consumed + task.reservation,
            unknown_invocation_ids=(*task.unknown_invocation_ids, invocation_id),
            error=error,
        )
        updated = _replace_supplemental_task(
            current,
            wave_id,
            task_id,
            updated_task,
            now,
        )
        self.write(updated)
        return updated

    def mark_wave_completed(
        self,
        wave_id: str,
        artifact_names: Iterable[str],
        stop_reason: str,
        now: str,
    ) -> SessionManifest:
        names = _unique_artifact_names(artifact_names)
        current = self.load()
        supplemental_phase = self._require_supplemental_phase_running(current)
        wave = _supplemental_wave(current, wave_id)
        if wave.status is PhaseStatus.COMPLETED:
            if set(wave.artifacts) == set(names) and wave.stop_reason == stop_reason:
                for artifact_name in names:
                    self._require_valid_phase_artifact(
                        current,
                        RunPhase.SUPPLEMENTAL_INVESTIGATION,
                        artifact_name,
                    )
                return current
            raise ValueError(
                f"supplemental wave {wave_id} is already completed differently"
            )
        if wave.status is not PhaseStatus.RUNNING:
            raise ValueError(f"supplemental wave {wave_id} must be running")
        incomplete = [
            task_id
            for task_id, task in wave.tasks.items()
            if task.status
            not in {
                SupplementalTaskStatus.COMPLETED,
                SupplementalTaskStatus.PARTIAL,
                SupplementalTaskStatus.FAILED,
            }
        ]
        if incomplete:
            raise ValueError(
                "cannot complete supplemental wave before tasks terminate: "
                + ", ".join(incomplete)
            )
        for artifact_name in names:
            self._require_valid_phase_artifact(
                current,
                RunPhase.SUPPLEMENTAL_INVESTIGATION,
                artifact_name,
            )
        task_artifacts = {
            artifact_name
            for task in wave.tasks.values()
            for artifact_name in task.artifacts
        }
        if not task_artifacts.issubset(names):
            raise ValueError(
                "supplemental wave artifact set omits committed task artifacts"
            )

        completed_wave = replace(
            wave,
            status=PhaseStatus.COMPLETED,
            completed_at=now,
            artifacts=names,
            stop_reason=stop_reason,
            error=None,
        )
        waves = dict(current.supplemental_waves)
        waves[wave_id] = completed_wave
        phases = dict(current.phases)
        phases[RunPhase.SUPPLEMENTAL_INVESTIGATION.value] = replace(
            supplemental_phase,
            artifacts=tuple(
                dict.fromkeys((*supplemental_phase.artifacts, *names))
            ),
        )
        updated = replace(
            current,
            phases=phases,
            supplemental_waves=waves,
            updated_at=now,
        )
        self.write(updated)
        return updated

    def invalidate_wave_from(
        self,
        wave_id: str,
        reason: str,
        now: str,
        *,
        task_id: str | None = None,
    ) -> SessionManifest:
        _require_non_empty_string(reason, "reason")
        current = self.load()
        self._require_supplemental_layout(current)
        target = _supplemental_wave(current, wave_id)
        if task_id is not None and task_id not in target.tasks:
            raise ValueError(
                f"supplemental task {task_id} does not belong to wave {wave_id}"
            )
        waves = dict(current.supplemental_waves)
        target_index = target.wave_index

        preserved_supplemental: set[str] = set()
        for candidate in waves.values():
            if candidate.wave_index < target_index:
                preserved_supplemental.update(candidate.artifacts)
                preserved_supplemental.update(
                    artifact_name
                    for task in candidate.tasks.values()
                    for artifact_name in task.artifacts
                )
            elif candidate.wave_index == target_index:
                plan_name = f"supplemental_wave_{candidate.wave_id}_plan"
                if plan_name in current.artifacts:
                    preserved_supplemental.add(plan_name)
                preserved_supplemental.update(
                    artifact_name
                    for candidate_task_id, task in candidate.tasks.items()
                    if task_id is None or candidate_task_id != task_id
                    for artifact_name in task.artifacts
                )

        for candidate_id, candidate in tuple(waves.items()):
            if candidate.wave_index < target_index:
                continue
            tasks = dict(candidate.tasks)
            invalidate_all_tasks = candidate.wave_index > target_index
            for candidate_task_id, task in tuple(tasks.items()):
                should_invalidate = invalidate_all_tasks or (
                    candidate.wave_index == target_index
                    and task_id is not None
                    and candidate_task_id == task_id
                )
                if not should_invalidate:
                    continue
                if task.status in {
                    SupplementalTaskStatus.RESERVED,
                    SupplementalTaskStatus.RUNNING,
                }:
                    raise ValueError(
                        "active supplemental task must be charged or marked unknown "
                        "before invalidation"
                    )
                tasks[candidate_task_id] = replace(
                    task,
                    status=SupplementalTaskStatus.INVALIDATED,
                    completed_at=None,
                    artifacts=(),
                    reservation=SupplementalBudget(),
                    error=(
                        reason
                        if candidate.wave_index == target_index
                        else f"invalidated because wave {wave_id} is invalid"
                    ),
                )
            waves[candidate_id] = replace(
                candidate,
                status=PhaseStatus.INVALIDATED,
                completed_at=None,
                artifacts=(),
                tasks=tasks,
                stop_reason=None,
                error=(
                    reason
                    if candidate.wave_index == target_index
                    else f"invalidated because wave {wave_id} is invalid"
                ),
            )

        layout = session_phases_for_schema(current.schema_version)
        supplemental_index = layout.index(RunPhase.SUPPLEMENTAL_INVESTIGATION)
        downstream = set(layout[supplemental_index + 1 :])
        artifacts = {
            name: descriptor
            for name, descriptor in current.artifacts.items()
            if (
                descriptor.phase is RunPhase.SUPPLEMENTAL_INVESTIGATION
                and name in preserved_supplemental
            )
            or (
                descriptor.phase is not RunPhase.SUPPLEMENTAL_INVESTIGATION
                and descriptor.phase not in downstream
            )
        }
        phases = dict(current.phases)
        supplemental_error = f"supplemental wave {wave_id} invalid: {reason}"
        phases[RunPhase.SUPPLEMENTAL_INVESTIGATION.value] = PhaseCheckpoint(
            status=PhaseStatus.INVALIDATED,
            attempts=phases[RunPhase.SUPPLEMENTAL_INVESTIGATION.value].attempts,
            error=supplemental_error,
        )
        for phase in layout[supplemental_index + 1 :]:
            checkpoint = phases[phase.value]
            phases[phase.value] = PhaseCheckpoint(
                status=PhaseStatus.INVALIDATED,
                attempts=checkpoint.attempts,
                error=f"invalidated because supplemental wave {wave_id} is invalid",
            )

        error_message = f"invalidated supplemental wave {wave_id}: {reason}"
        updated = replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=RunPhase.SUPPLEMENTAL_INVESTIGATION,
            last_successful_phase=next(
                (
                    phase
                    for phase in reversed(layout[:supplemental_index])
                    if phases[phase.value].status is PhaseStatus.COMPLETED
                ),
                None,
            ),
            phases=phases,
            artifacts=artifacts,
            supplemental_waves=waves,
            errors=(*current.errors, error_message),
            updated_at=now,
        )
        if updated == current:
            return current
        self._require_immutable_metadata(current, updated)
        _atomic_write_text(self.session_path, self._serialize(updated))
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
        self._require_current_layout(current)
        layout = session_phases_for_schema(current.schema_version)
        if phase not in layout:
            raise ValueError(
                f"phase {phase.value} is not part of this Session schema layout"
            )
        phase_index = layout.index(phase)
        invalidated_phases = set(layout[phase_index:])
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
                for candidate in layout[phase_index:]
            )
            and not any(
                descriptor.phase in invalidated_phases
                for descriptor in current.artifacts.values()
            )
        ):
            return current
        phases = dict(current.phases)
        for candidate in layout[phase_index:]:
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
                for candidate in reversed(layout[:phase_index])
                if phases[candidate.value].status is PhaseStatus.COMPLETED
            ),
            None,
        )
        supplemental_waves = current.supplemental_waves
        if RunPhase.SUPPLEMENTAL_INVESTIGATION in invalidated_phases:
            supplemental_waves = {}
        error_message = f"invalidated {phase.value}: {reason}"
        updated = replace(
            current,
            status=RunStatus.RUNNING,
            current_phase=phase,
            last_successful_phase=last_successful,
            phases=phases,
            artifacts=artifacts,
            supplemental_waves=supplemental_waves,
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
        self._require_current_layout(current)
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
        layout = session_phases_for_schema(current.schema_version)
        reviewer_index = layout.index(RunPhase.REVIEWERS)
        downstream = set(layout[reviewer_index + 1 :])
        for phase in layout[reviewer_index + 1 :]:
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
            last_successful_phase=RunPhase.PLANNING,
            phases=phases,
            artifacts=artifacts,
            supplemental_waves={},
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
        self._require_current_layout(current)
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
        layout = session_phases_for_schema(current.schema_version)
        reviewer_index = layout.index(RunPhase.REVIEWERS)
        downstream = set(layout[reviewer_index + 1 :])
        for phase in layout[reviewer_index + 1 :]:
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
            last_successful_phase=RunPhase.PLANNING,
            phases=phases,
            artifacts=artifacts,
            supplemental_waves={},
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
        self._require_current_layout(current)
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
        if checkpoint.status is PhaseStatus.AWAITING_USER:
            raise ValueError(
                "resume intent_resolution before marking it completed"
            )
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
            user_decisions=checkpoint.user_decisions,
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
        self._require_current_layout(current)
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
        self._require_current_layout(current)
        incomplete = [
            phase.value
            for phase in session_phases_for_schema(current.schema_version)
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

        for phase in session_phases_for_schema(current.schema_version):
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
        self._require_current_layout(current)
        checkpoint = current.phases[phase.value]
        if current.status is RunStatus.COMPLETED:
            raise ValueError("cannot fail a completed Session")
        if checkpoint.status is PhaseStatus.COMPLETED:
            raise ValueError(f"cannot fail completed phase: {phase.value}")
        layout = session_phases_for_schema(current.schema_version)
        if phase not in layout:
            raise ValueError(
                f"phase {phase.value} is not part of this Session schema layout"
            )
        phase_index = layout.index(phase)
        for predecessor in layout[:phase_index]:
            predecessor_status = current.phases[predecessor.value].status
            if predecessor_status is not PhaseStatus.COMPLETED:
                raise ValueError(
                    f"cannot fail {phase.value} because predecessor "
                    f"{predecessor.value} is {predecessor_status.value}"
                )
        for successor in layout[phase_index + 1 :]:
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
                if checkpoint.status in {
                    PhaseStatus.RUNNING,
                    PhaseStatus.AWAITING_USER,
                }
                else checkpoint.attempts + 1
            ),
            started_at=checkpoint.started_at or now,
            completed_at=None,
            artifacts=checkpoint.artifacts,
            error=error,
            tasks=checkpoint.tasks,
            user_decisions=checkpoint.user_decisions,
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
        layout = session_phases_for_schema(manifest.schema_version)
        if phase not in layout:
            raise ValueError(
                f"phase {phase.value} is not part of this Session schema layout"
            )
        phase_index = layout.index(phase)
        for predecessor in layout[:phase_index]:
            status = manifest.phases[predecessor.value].status
            if status is not PhaseStatus.COMPLETED:
                raise ValueError(
                    f"cannot run {phase.value} before {predecessor.value}; "
                    f"predecessor is {status.value}"
                )

    @staticmethod
    def _require_no_completed_successors(
        manifest: SessionManifest,
        phase: RunPhase,
    ) -> None:
        layout = session_phases_for_schema(manifest.schema_version)
        if phase not in layout:
            raise ValueError(
                f"phase {phase.value} is not part of this Session schema layout"
            )
        phase_index = layout.index(phase)
        completed = [
            successor.value
            for successor in layout[phase_index + 1 :]
            if manifest.phases[successor.value].status is PhaseStatus.COMPLETED
        ]
        if completed:
            raise ValueError(
                f"cannot await user input in {phase.value} because completed "
                f"successor phases exist: {', '.join(completed)}"
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
    def _require_supplemental_layout(manifest: SessionManifest) -> PhaseCheckpoint:
        if manifest.schema_version != SESSION_SCHEMA_VERSION:
            raise ValueError(
                "supplemental investigation is available only for current "
                "Session schema Sessions"
            )
        checkpoint = manifest.phases.get(
            RunPhase.SUPPLEMENTAL_INVESTIGATION.value
        )
        if checkpoint is None:
            raise ValueError(
                "supplemental investigation phase is missing from Session layout"
            )
        return checkpoint

    @classmethod
    def _require_supplemental_phase_running(
        cls,
        manifest: SessionManifest,
    ) -> PhaseCheckpoint:
        checkpoint = cls._require_supplemental_layout(manifest)
        if (
            manifest.status is not RunStatus.RUNNING
            or manifest.current_phase is not RunPhase.SUPPLEMENTAL_INVESTIGATION
            or checkpoint.status is not PhaseStatus.RUNNING
        ):
            raise ValueError("supplemental investigation phase must be running")
        return checkpoint

    def _require_supplemental_task_artifacts(
        self,
        manifest: SessionManifest,
        task_id: str,
        artifact_names: Iterable[str],
    ) -> None:
        names = tuple(artifact_names)
        if not names:
            raise ValueError(
                f"supplemental task {task_id} must commit artifacts"
            )
        for artifact_name in names:
            self._require_valid_phase_artifact(
                manifest,
                RunPhase.SUPPLEMENTAL_INVESTIGATION,
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


def _supplemental_wave(
    manifest: SessionManifest,
    wave_id: str,
) -> ReviewWaveCheckpoint:
    _require_non_empty_string(wave_id, "wave_id")
    wave = manifest.supplemental_waves.get(wave_id)
    if wave is None:
        raise ValueError(f"supplemental wave is not initialized: {wave_id}")
    return wave


def _supplemental_task(
    manifest: SessionManifest,
    task_id: str,
) -> tuple[str, ReviewWaveCheckpoint, SupplementalTaskCheckpoint]:
    _require_non_empty_string(task_id, "task_id")
    matches = [
        (wave_id, wave, wave.tasks[task_id])
        for wave_id, wave in manifest.supplemental_waves.items()
        if task_id in wave.tasks
    ]
    if not matches:
        raise ValueError(f"supplemental task is not initialized: {task_id}")
    if len(matches) != 1:
        raise ValueError(f"supplemental task ID is ambiguous: {task_id}")
    return matches[0]


def _replace_supplemental_task(
    manifest: SessionManifest,
    wave_id: str,
    task_id: str,
    task: SupplementalTaskCheckpoint,
    now: str,
) -> SessionManifest:
    if not isinstance(task, SupplementalTaskCheckpoint):
        raise ValueError("task must be a SupplementalTaskCheckpoint")
    wave = _supplemental_wave(manifest, wave_id)
    if task_id not in wave.tasks or task.task_id != task_id:
        raise ValueError("supplemental task identity does not match wave registry")
    tasks = dict(wave.tasks)
    tasks[task_id] = task
    waves = dict(manifest.supplemental_waves)
    waves[wave_id] = replace(wave, tasks=tasks)
    return replace(
        manifest,
        supplemental_waves=waves,
        updated_at=now,
    )


def _supplemental_budget_ceiling(
    manifest: SessionManifest,
) -> SupplementalBudget:
    policy = next(
        (
            wave.effective_policy
            for wave in sorted(
                manifest.supplemental_waves.values(),
                key=lambda item: item.wave_index,
            )
        ),
        manifest.execution.supplemental_policy,
    )
    return SupplementalBudget(
        tasks=policy.max_tasks,
        tool_calls=policy.max_total_tool_calls,
        tokens=policy.max_total_tokens,
        elapsed_seconds=policy.max_elapsed_seconds,
    )


def _supplemental_consumption(
    manifest: SessionManifest,
) -> SupplementalBudget:
    total = SupplementalBudget()
    for wave in manifest.supplemental_waves.values():
        for task in wave.tasks.values():
            total = (
                total
                + task.charged
                + task.unknown_consumed
                + task.reservation
            )
    return total


def _require_task_reservation_within_policy(
    reservation: SupplementalBudget,
    policy: SupplementalPolicy,
) -> None:
    if reservation.tasks != 1:
        raise ValueError("supplemental task reservation must reserve exactly one task")
    if reservation.tool_calls > policy.max_tool_calls_per_task:
        raise ValueError(
            "supplemental task reservation exceeds max_tool_calls_per_task"
        )
    if reservation.tokens > policy.max_tokens_per_task:
        raise ValueError(
            "supplemental task reservation exceeds max_tokens_per_task"
        )
    if reservation.elapsed_seconds > policy.max_elapsed_seconds:
        raise ValueError(
            "supplemental task reservation exceeds elapsed-time budget"
        )


def _require_task_charge(
    charged: SupplementalBudget,
    reservation: SupplementalBudget,
) -> None:
    if charged.tasks != 1:
        raise ValueError("supplemental task charge must charge exactly one task")
    if not charged.fits_within(reservation):
        raise ValueError("supplemental task charge exceeds its reservation")


def _require_session_phase(phase: RunPhase) -> RunPhase:
    if not isinstance(phase, RunPhase) or phase not in SESSION_PHASES:
        raise ValueError("phase must be one of the persisted SESSION_PHASES")
    return phase


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


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

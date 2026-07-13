from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path, PurePosixPath

from review_agent.artifacts import artifact_schema
from review_agent.checkpoint import CheckpointStore
from review_agent.hydration import review_request_from_dict
from review_agent.intent_clarification import IntentClarifier
from review_agent.incremental import (
    classify_revision_change,
    deterministic_child_review_id,
)
from review_agent.models import ReviewRequest
from review_agent.pipeline import (
    PHASE_MESSAGES,
    PipelineHydrationError,
    PipelineResult,
    ReviewPipeline,
)
from review_agent.revision import (
    RepositoryIdentity,
    ResolvedRevisions,
    RevisionResolver,
)
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    LEGACY_SESSION_SCHEMA_VERSION,
    PhaseStatus,
    RevisionChangeKind,
    SessionManifest,
    child_session_manifest,
)
from review_agent.session_store import SessionStore


class ResumeAction(str, Enum):
    AUDIT_COMPLETED = "audit_completed"
    CONTINUE_SESSION = "continue_session"
    CREATE_INCREMENTAL_SESSION = "create_incremental_session"
    AWAITING_USER = "awaiting_user"
    BLOCKED = "blocked"


class ResumeBlockedError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResumeResult:
    action: ResumeAction
    manifest: SessionManifest
    starting_phase: RunPhase | None
    reused_phases: tuple[RunPhase, ...]
    pipeline_result: PipelineResult | None = None
    parent_review_id: str | None = None
    new_review_id: str | None = None
    change_kind: RevisionChangeKind | None = None
    full_range: str | None = None
    incremental_range: str | None = None
    child_created: bool = False


class ReviewSessionResumer:
    def __init__(
        self,
        *,
        repository: Path,
        checkpoint_store: CheckpointStore,
        session_store: SessionStore,
        resolver: RevisionResolver | None = None,
        intent_clarifier: IntentClarifier | None = None,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.checkpoint_store = checkpoint_store
        self.session_store = session_store
        self.resolver = resolver or RevisionResolver()
        self.intent_clarifier = intent_clarifier

    def resume(self) -> ResumeResult:
        manifest = self.session_store.load()
        identity, current_revisions = self._validate_repository_and_revisions(
            manifest
        )
        request = self._load_request(manifest)
        if manifest.schema_version == LEGACY_SESSION_SCHEMA_VERSION:
            if manifest.status is not RunStatus.COMPLETED:
                raise ResumeBlockedError(
                    "schema v1 Session is read-only; start a new schema v3 review"
                )
            return ResumeResult(
                action=ResumeAction.AUDIT_COMPLETED,
                manifest=manifest,
                starting_phase=None,
                reused_phases=tuple(
                    RunPhase(phase) for phase in manifest.phases
                ),
            )
        change_kind = classify_revision_change(manifest, current_revisions)
        if change_kind is not RevisionChangeKind.INITIAL:
            return self._resume_incremental_child(
                parent=manifest,
                repository=identity,
                revisions=current_revisions,
                request=request,
                change_kind=change_kind,
            )
        pipeline = ReviewPipeline(
            repository=self.repository,
            checkpoint_store=self.checkpoint_store,
            session_store=self.session_store,
            request=request,
            intent_clarifier=self.intent_clarifier,
        )
        reused: list[RunPhase] = []
        starting_phase: RunPhase | None = None

        for phase in PHASE_MESSAGES:
            checkpoint = self.session_store.load().phases[phase.value]
            if checkpoint.status is not PhaseStatus.COMPLETED:
                starting_phase = phase
                break
            if phase is RunPhase.REVIEWERS and checkpoint.tasks:
                failures = pipeline.validate_completed_reviewer_tasks()
                if failures:
                    task_name = sorted(failures)[0]
                    self.session_store.invalidate_reviewer_task(
                        task_name,
                        failures[task_name],
                        _utc_now(),
                    )
                    starting_phase = RunPhase.REVIEWERS
                    break
            try:
                pipeline.load_phase(phase)
            except PipelineHydrationError as error:
                if phase is RunPhase.REVIEWERS and checkpoint.tasks:
                    self.session_store.invalidate_reviewers_preserving_tasks(
                        str(error),
                        _utc_now(),
                    )
                else:
                    self.session_store.invalidate_from(
                        phase,
                        str(error),
                        _utc_now(),
                    )
                starting_phase = phase
                break
            reused.append(phase)

        if starting_phase is RunPhase.INTENT_RESOLUTION:
            checkpoint = self.session_store.load().phases[
                RunPhase.INTENT_RESOLUTION.value
            ]
            if checkpoint.status is PhaseStatus.AWAITING_USER:
                if self.intent_clarifier is None:
                    return ResumeResult(
                        action=ResumeAction.AWAITING_USER,
                        manifest=self.session_store.load(),
                        starting_phase=starting_phase,
                        reused_phases=tuple(reused),
                    )
                pipeline.apply_submitted_intent_decisions()
                open_questions = [
                    question
                    for question in pipeline.context.intent_questions
                    if question.status.value in {"pending", "open"}
                ]
                for question in open_questions:
                    decision = self.intent_clarifier.decide(question)
                    if decision is None:
                        return ResumeResult(
                            action=ResumeAction.AWAITING_USER,
                            manifest=self.session_store.load(),
                            starting_phase=starting_phase,
                            reused_phases=tuple(reused),
                        )
                    artifact_name = f"intent_decision_{decision.decision_id}"
                    filename = f"intent_decision_{decision.decision_id}.json"
                    self.checkpoint_store.write_json(filename, asdict(decision))
                    self.session_store.register_existing_artifact(
                        name=artifact_name,
                        relative_path=filename,
                        schema=artifact_schema(artifact_name),
                        phase=RunPhase.INTENT_RESOLUTION,
                        revision_binding=pipeline.context.revision_binding,
                        now=_utc_now(),
                    )
                    self.session_store.submit_user_decision(
                        decision.decision_id,
                        artifact_name,
                        _utc_now(),
                    )
                self.session_store.resume_awaiting_user(_utc_now())

        if starting_phase is None:
            latest = self.session_store.load()
            if latest.status is not RunStatus.COMPLETED:
                latest = self.session_store.mark_session_completed(_utc_now())
            return ResumeResult(
                action=ResumeAction.AUDIT_COMPLETED,
                manifest=latest,
                starting_phase=None,
                reused_phases=tuple(reused),
            )

        result = pipeline.execute(
            starting_phase=starting_phase,
            resuming=True,
        )
        return ResumeResult(
            action=ResumeAction.CONTINUE_SESSION,
            manifest=self.session_store.load(),
            starting_phase=starting_phase,
            reused_phases=tuple(reused),
            pipeline_result=result,
        )

    def _validate_repository_and_revisions(
        self,
        manifest: SessionManifest,
    ) -> tuple[RepositoryIdentity, ResolvedRevisions]:
        identity = self.resolver.repository_identity(self.repository)
        if _canonical_path(identity.git_common_dir) != _canonical_path(
            manifest.repository.git_common_dir
        ):
            raise ResumeBlockedError("repository identity mismatch")
        for label, sha in (
            ("base", manifest.revisions.resolved_base_sha),
            ("head", manifest.revisions.resolved_head_sha),
        ):
            if not self.resolver.commit_exists(self.repository, sha):
                raise ResumeBlockedError(f"stored resolved {label} commit is missing: {sha}")
        current = self.resolver.resolve_pair(
            self.repository,
            manifest.revisions.requested_base,
            manifest.revisions.requested_head,
        )
        return identity, current

    def _resume_incremental_child(
        self,
        *,
        parent: SessionManifest,
        repository: RepositoryIdentity,
        revisions: ResolvedRevisions,
        request: ReviewRequest,
        change_kind: RevisionChangeKind,
    ) -> ResumeResult:
        child_id = deterministic_child_review_id(
            repository=repository,
            parent_review_id=parent.review_id,
            revisions=revisions,
        )
        child_checkpoint = CheckpointStore(self.repository, child_id)
        child_store = SessionStore(child_checkpoint.run_dir)
        child_created = not child_store.session_path.exists()
        if child_created:
            try:
                child_store.create(
                    child_session_manifest(
                        review_id=child_id,
                        parent=parent,
                        repository=repository,
                        revisions=revisions,
                        change_kind=change_kind,
                        now=_utc_now(),
                    )
                )
            except FileExistsError:
                child_created = False
        try:
            child = child_store.load()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ResumeBlockedError(
                f"deterministic child Session is invalid: {error}"
            ) from error
        self._validate_existing_child(
            child=child,
            parent=parent,
            revisions=revisions,
            change_kind=change_kind,
        )

        nested_result: ResumeResult | None = None
        pipeline_result: PipelineResult | None = None
        if "request" not in child.artifacts:
            preflight = child.phases[RunPhase.PREFLIGHT.value]
            if preflight.status is PhaseStatus.COMPLETED:
                raise ResumeBlockedError(
                    "child request artifact is missing from completed preflight"
                )
            pipeline_result = ReviewPipeline(
                repository=self.repository,
                checkpoint_store=child_checkpoint,
                session_store=child_store,
                request=request,
                intent_clarifier=self.intent_clarifier,
            ).execute(
                starting_phase=RunPhase.PREFLIGHT,
                resuming=preflight.status is not PhaseStatus.PENDING,
            )
            starting_phase = RunPhase.PREFLIGHT
            reused_phases: tuple[RunPhase, ...] = ()
        else:
            nested_result = ReviewSessionResumer(
                repository=self.repository,
                checkpoint_store=child_checkpoint,
                session_store=child_store,
                resolver=self.resolver,
                intent_clarifier=self.intent_clarifier,
            ).resume()
            pipeline_result = nested_result.pipeline_result
            starting_phase = nested_result.starting_phase
            reused_phases = nested_result.reused_phases

        incremental_range = (
            f"{parent.revisions.resolved_head_sha}..{revisions.resolved_head_sha}"
            if change_kind is RevisionChangeKind.HEAD_MOVED
            else None
        )
        return ResumeResult(
            action=ResumeAction.CREATE_INCREMENTAL_SESSION,
            manifest=child_store.load(),
            starting_phase=starting_phase,
            reused_phases=reused_phases,
            pipeline_result=pipeline_result,
            parent_review_id=parent.review_id,
            new_review_id=child_id,
            change_kind=change_kind,
            full_range=f"{revisions.resolved_base_sha}..{revisions.resolved_head_sha}",
            incremental_range=incremental_range,
            child_created=child_created,
        )

    @staticmethod
    def _validate_existing_child(
        *,
        child: SessionManifest,
        parent: SessionManifest,
        revisions: ResolvedRevisions,
        change_kind: RevisionChangeKind,
    ) -> None:
        if (
            child.parent_review_id != parent.review_id
            or child.root_review_id != parent.root_review_id
            or child.revisions != revisions
            or child.revision_change_kind is not change_kind
            or child.original_base_sha != parent.original_base_sha
        ):
            raise ResumeBlockedError(
                "deterministic child Session already exists with mismatched lineage"
            )

    def _load_request(self, manifest: SessionManifest) -> ReviewRequest:
        descriptor = manifest.artifacts.get("request")
        if descriptor is None:
            raise ResumeBlockedError(
                "request artifact is unavailable; the original review intent "
                "cannot be reconstructed safely"
            )
        if descriptor.schema != artifact_schema("request"):
            raise ResumeBlockedError("request artifact schema is unsupported")
        if not self.session_store.validate_artifact(descriptor):
            raise ResumeBlockedError("request artifact failed hash validation")
        path = self.checkpoint_store.run_dir.joinpath(
            *PurePosixPath(descriptor.path).parts
        )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request artifact must contain an object")
            return review_request_from_dict(payload)
        except Exception as error:
            raise ResumeBlockedError(f"request artifact is invalid: {error}") from error


def _canonical_path(value: str) -> str:
    return os.path.normcase(str(Path(value).resolve()))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

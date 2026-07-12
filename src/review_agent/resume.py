from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path, PurePosixPath

from review_agent.artifacts import artifact_schema
from review_agent.checkpoint import CheckpointStore
from review_agent.hydration import review_request_from_dict
from review_agent.models import ReviewRequest
from review_agent.pipeline import (
    PHASE_MESSAGES,
    PipelineHydrationError,
    PipelineResult,
    ReviewPipeline,
)
from review_agent.revision import RevisionResolver
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import PhaseStatus, SessionManifest
from review_agent.session_store import SessionStore


class ResumeAction(str, Enum):
    AUDIT_COMPLETED = "audit_completed"
    CONTINUE_SESSION = "continue_session"
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


class ReviewSessionResumer:
    def __init__(
        self,
        *,
        repository: Path,
        checkpoint_store: CheckpointStore,
        session_store: SessionStore,
        resolver: RevisionResolver | None = None,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.checkpoint_store = checkpoint_store
        self.session_store = session_store
        self.resolver = resolver or RevisionResolver()

    def resume(self) -> ResumeResult:
        manifest = self.session_store.load()
        self._validate_repository_and_revisions(manifest)
        request = self._load_request(manifest)
        pipeline = ReviewPipeline(
            repository=self.repository,
            checkpoint_store=self.checkpoint_store,
            session_store=self.session_store,
            request=request,
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

    def _validate_repository_and_revisions(self, manifest: SessionManifest) -> None:
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
        if (
            current.resolved_base_sha.casefold()
            != manifest.revisions.resolved_base_sha.casefold()
            or current.resolved_head_sha.casefold()
            != manifest.revisions.resolved_head_sha.casefold()
        ):
            raise ResumeBlockedError(
                "requested revisions have drifted; Batch C must create an "
                "incremental child Session"
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

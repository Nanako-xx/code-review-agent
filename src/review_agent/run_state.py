from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    AWAITING_USER = "awaiting_user"
    COMPLETED = "completed"
    FAILED = "failed"


class RunPhase(str, Enum):
    CREATED = "created"
    PREFLIGHT = "preflight"
    QUALITY_GATES = "quality_gates"
    REPOSITORY_INTELLIGENCE = "repository_intelligence"
    MEMORY_SELECTION = "memory_selection"
    INTENT_DISCOVERY = "intent_discovery"
    INTENT_RESOLUTION = "intent_resolution"
    PLANNING = "planning"
    REVIEWERS = "reviewers"
    RECONCILIATION_ANALYSIS = "reconciliation_analysis"
    SUPPLEMENTAL_INVESTIGATION = "supplemental_investigation"
    RECONCILIATION = "reconciliation"
    COMPLETION = "completion"
    FINAL_RISK = "final_risk"
    MEMORY_PROPOSAL = "memory_proposal"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunState:
    review_id: str
    status: RunStatus
    phase: RunPhase
    repository_path: str
    base_revision: str
    head_revision: str
    message: str
    resolved_base_revision: str | None = None
    resolved_head_revision: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status is RunStatus.AWAITING_USER:
            if self.phase is not RunPhase.INTENT_RESOLUTION:
                raise ValueError(
                    "awaiting_user RunState is allowed only during intent_resolution"
                )


def initial_run_state(
    *,
    review_id: str,
    repository_path: str,
    base_revision: str,
    head_revision: str,
    resolved_base_revision: str | None = None,
    resolved_head_revision: str | None = None,
) -> RunState:
    return RunState(
        review_id=review_id,
        status=RunStatus.CREATED,
        phase=RunPhase.CREATED,
        repository_path=repository_path,
        base_revision=base_revision,
        head_revision=head_revision,
        message="Run created",
        resolved_base_revision=resolved_base_revision,
        resolved_head_revision=resolved_head_revision,
    )


def advance_run_state(
    state: RunState,
    *,
    phase: RunPhase,
    message: str,
    artifacts: dict[str, str] | None = None,
) -> RunState:
    next_artifacts = dict(state.artifacts)
    if artifacts:
        next_artifacts.update(artifacts)
    status = RunStatus.COMPLETED if phase is RunPhase.COMPLETED else RunStatus.RUNNING
    return RunState(
        review_id=state.review_id,
        status=status,
        phase=phase,
        repository_path=state.repository_path,
        base_revision=state.base_revision,
        head_revision=state.head_revision,
        message=message,
        resolved_base_revision=state.resolved_base_revision,
        resolved_head_revision=state.resolved_head_revision,
        artifacts=next_artifacts,
        errors=list(state.errors),
    )


def await_user_run_state(
    state: RunState,
    *,
    message: str,
    artifacts: dict[str, str] | None = None,
) -> RunState:
    next_artifacts = dict(state.artifacts)
    if artifacts:
        next_artifacts.update(artifacts)
    return RunState(
        review_id=state.review_id,
        status=RunStatus.AWAITING_USER,
        phase=RunPhase.INTENT_RESOLUTION,
        repository_path=state.repository_path,
        base_revision=state.base_revision,
        head_revision=state.head_revision,
        message=message,
        resolved_base_revision=state.resolved_base_revision,
        resolved_head_revision=state.resolved_head_revision,
        artifacts=next_artifacts,
        errors=list(state.errors),
    )


def fail_run_state(state: RunState, *, message: str, error: str) -> RunState:
    return RunState(
        review_id=state.review_id,
        status=RunStatus.FAILED,
        phase=RunPhase.FAILED,
        repository_path=state.repository_path,
        base_revision=state.base_revision,
        head_revision=state.head_revision,
        message=message,
        resolved_base_revision=state.resolved_base_revision,
        resolved_head_revision=state.resolved_head_revision,
        artifacts=dict(state.artifacts),
        errors=[*state.errors, error],
    )


def run_state_to_dict(state: RunState) -> dict[str, Any]:
    return {
        "review_id": state.review_id,
        "status": state.status.value,
        "phase": state.phase.value,
        "repository_path": state.repository_path,
        "base_revision": state.base_revision,
        "head_revision": state.head_revision,
        "resolved_base_revision": state.resolved_base_revision,
        "resolved_head_revision": state.resolved_head_revision,
        "message": state.message,
        "artifacts": dict(state.artifacts),
        "errors": list(state.errors),
    }


def run_state_from_dict(payload: dict[str, Any]) -> RunState:
    return RunState(
        review_id=str(payload["review_id"]),
        status=RunStatus(str(payload["status"])),
        phase=RunPhase(str(payload["phase"])),
        repository_path=str(payload["repository_path"]),
        base_revision=str(payload["base_revision"]),
        head_revision=str(payload["head_revision"]),
        message=str(payload["message"]),
        resolved_base_revision=(
            str(payload["resolved_base_revision"])
            if payload.get("resolved_base_revision") is not None
            else None
        ),
        resolved_head_revision=(
            str(payload["resolved_head_revision"])
            if payload.get("resolved_head_revision") is not None
            else None
        ),
        artifacts={str(key): str(value) for key, value in dict(payload.get("artifacts", {})).items()},
        errors=[str(item) for item in list(payload.get("errors", []))],
    )
